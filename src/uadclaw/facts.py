"""One APK's manifest turned into typed facts. Pure: no database, no filesystem beyond the
file it is handed.

androguard rather than aapt2, per `docs/pipeline-design.md` §3: aapt2 resolves everything we
need but drags the Android SDK and a JRE into the worker image, and androguard is pure Python.
Measured 2026-08-11 over all 312 APKs of `oriole-cp2a.260705.006.a1`: 312 parsed, zero errors.

Two measured shapes this module encodes rather than rediscovers:

- **`coreApp` is un-namespaced.** AOSP parses it with a null namespace
  (`parser.getAttributeBooleanValue(null, "coreApp", false)`), so a reader that only walks
  `android:`-prefixed attributes misses all 23 of them on a Pixel and computes a floor of
  `Recommended` for packages the platform will not boot without.
- **A missing `android:label` is a missing attribute, not a resolution failure.** On the same
  corpus `get_app_name()` resolves 177 of 225 non-overlay packages and 18 of 87 overlays, and
  every miss spot-checked (`com.google.android.hardwareinfo`, `com.google.android.iwlan`,
  `com.android.imsserviceentitlement`) simply never declared one. So the answer is
  `LABEL_UNKNOWN` and there is no ARSC fallback to build. The `@7f0…`-style unresolved id is
  still handled — it fired zero times on this corpus — and is recorded as
  `label_unresolved` rather than presented as a name, because an id is not something a
  triage screen or a model may read as a human-facing title.

androguard 4.x logs through **loguru**, which stdlib `logging.disable()` does not reach: eight
APKs emitted 123 KB of DEBUG in the baseline run. `logger.disable("androguard")` at import
silences that path by module name, so the worker does not flood its own logs, and it is scoped
to androguard rather than removing every loguru sink the process might own.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger as _loguru_logger

# Before androguard is imported, not after: its module-level loggers bind at import time.
_loguru_logger.disable("androguard")

from androguard.core.apk import APK  # noqa: E402

ANDROID_NS = "{http://schemas.android.com/apk/res/android}"

# What a package with no declared (or unresolvable) label is called downstream. A real word
# rather than an empty string, because this reaches a triage card and a model prompt.
LABEL_UNKNOWN = "unknown"

# Service/receiver intent actions that make a package a member of a class whose removal has a
# specific, known consequence (`docs/research/apk-analysis-signals.md` §14).
INPUT_METHOD_ACTION = "android.view.InputMethod"
ACCESSIBILITY_SERVICE_ACTION = "android.accessibilityservice.AccessibilityService"
CARRIER_SERVICE_ACTION = "android.service.carrier.CarrierService"
DEVICE_ADMIN_ACTION = "android.app.action.DEVICE_ADMIN_ENABLED"

# Manifest elements that can carry an `<intent-filter>`.
_FILTERABLE_COMPONENTS = ("activity", "activity-alias", "service", "receiver", "provider")


class FactsError(RuntimeError):
    """Base for this module's failures."""


class ApkParseError(FactsError):
    """This APK could not be turned into facts.

    Its own type, distinct from a bug in this module: the input is a file out of a vendor
    firmware image, and a deliberately or accidentally malformed AXML is a known androguard
    failure mode. The caller's policy is to record it and carry on, which it can only do if
    bad input is distinguishable from a genuine defect.
    """


@dataclass(frozen=True, slots=True)
class IntentFilterFact:
    component: str  # activity | activity-alias | service | receiver | provider
    component_class: str
    actions: tuple[str, ...]
    categories: tuple[str, ...]
    # AOSP's default when the attribute is absent, and what a non-integer (resource-reference)
    # priority also reads as: this is evidence for "sole high-priority handler", so an
    # unreadable value must not sort above a declared one.
    priority: int = 0

    def as_json(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "component_class": self.component_class,
            "actions": list(self.actions),
            "categories": list(self.categories),
            "priority": self.priority,
        }


@dataclass(frozen=True, slots=True)
class LibraryFact:
    name: str
    version: int | None = None

    def as_json(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True, slots=True)
class ApkFacts:
    """Everything `docs/pipeline-design.md` §3 asks for, off one APK.

    Frozen because these are observations: nothing downstream may adjust a fact in place, and
    the merge in `uadclaw.factstore` builds new values rather than mutating these.
    """

    package: str
    label: str
    label_unresolved: bool
    version_code: int | None
    partition: str
    device_path: str
    priv_app: bool
    sha256: str
    cert_issuer: str | None
    cert_subject: str | None
    core_app: bool
    shared_user_id: str | None
    persistent: bool
    has_code: bool
    overlay_target: str | None
    overlay_static: bool
    overlay_priority: int | None
    libraries: tuple[LibraryFact, ...] = ()
    static_libraries: tuple[LibraryFact, ...] = ()
    uses_libraries_required: tuple[str, ...] = ()
    uses_libraries_optional: tuple[str, ...] = ()
    protected_broadcasts: tuple[str, ...] = ()
    provider_authorities: tuple[str, ...] = ()
    # `<queries><package name="X">`. Deliberately NOT a dependency edge: a caller is expected
    # to handle the queried package being absent, so this is evidence for the human and the
    # model only (`docs/pipeline-design.md` §4).
    queries_packages: tuple[str, ...] = ()
    intent_filters: tuple[IntentFilterFact, ...] = field(default_factory=tuple)
    is_input_method: bool = False
    is_device_admin: bool = False
    is_accessibility_service: bool = False
    is_carrier_service: bool = False


def _parse_int(value: str | None) -> int | None:
    """Manifest integers arrive as text and may be hex; a resource reference (`@7f01…`) has no
    integer meaning here and reads as absent rather than as zero."""
    if value is None:
        return None
    text = value.strip()
    try:
        return int(text, 0)
    except ValueError:
        return None


def _is_true(value: str | None) -> bool:
    return value == "true"


def _attr(element: Any, name: str) -> str | None:
    if element is None:
        return None
    value = element.get(f"{ANDROID_NS}{name}")
    return value if value else None


def _child_names(parent: Any, tag: str, attribute: str = "name") -> tuple[str, ...]:
    if parent is None:
        return ()
    seen: list[str] = []
    for element in parent.iterfind(tag):
        value = _attr(element, attribute)
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)


def _libraries(parent: Any, tag: str) -> tuple[LibraryFact, ...]:
    if parent is None:
        return ()
    found: list[LibraryFact] = []
    for element in parent.iterfind(tag):
        name = _attr(element, "name")
        if name is None:
            continue
        found.append(LibraryFact(name=name, version=_parse_int(_attr(element, "version"))))
    return tuple(found)


def _uses_libraries(parent: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """`required` defaults to `"true"` per the AOSP manifest docs, so an omitted attribute is a
    hard dependency and must not fall into the optional bucket — that bucket is explicitly not
    emitted as a graph edge in task 5."""
    required: list[str] = []
    optional: list[str] = []
    if parent is None:
        return (), ()
    for element in parent.iterfind("uses-library"):
        name = _attr(element, "name")
        if name is None:
            continue
        bucket = optional if _attr(element, "required") == "false" else required
        if name not in bucket:
            bucket.append(name)
    return tuple(required), tuple(optional)


def _provider_authorities(application: Any) -> tuple[str, ...]:
    """A provider may declare several authorities in one semicolon-separated attribute."""
    if application is None:
        return ()
    found: list[str] = []
    for element in application.iterfind("provider"):
        raw = _attr(element, "authorities")
        if not raw:
            continue
        for authority in raw.split(";"):
            name = authority.strip()
            if name and name not in found:
                found.append(name)
    return tuple(found)


def _intent_filters(application: Any) -> tuple[IntentFilterFact, ...]:
    if application is None:
        return ()
    filters: list[IntentFilterFact] = []
    for component in _FILTERABLE_COMPONENTS:
        for element in application.iterfind(component):
            component_class = _attr(element, "name") or ""
            for intent_filter in element.iterfind("intent-filter"):
                filters.append(
                    IntentFilterFact(
                        component=component,
                        component_class=component_class,
                        actions=_child_names(intent_filter, "action"),
                        categories=_child_names(intent_filter, "category"),
                        priority=_parse_int(_attr(intent_filter, "priority")) or 0,
                    )
                )
    return tuple(filters)


def _declares(filters: tuple[IntentFilterFact, ...], component: str, action: str) -> bool:
    return any(f.component == component and action in f.actions for f in filters)


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _certificate_names(apk: APK) -> tuple[str | None, str | None]:
    """Issuer and subject of the signing certificate, sorted-first when an APK carries more
    than one, so the same APK always yields the same pair (an unordered pick would make the
    cross-device conflict check fire on nothing but iteration order).

    The issuer is the seedable vendor mapping — it is how Gamba et al. attributed
    pre-installed software, and it decides `list` far more reliably than a name prefix.
    """
    try:
        certificates = apk.get_certificates()
    except Exception as exc:  # androguard raises assorted types on a broken signature block
        raise ApkParseError(
            f"_certificate_names: signature block of {apk.get_filename()} could not be read "
            f"({type(exc).__name__}: {exc}); the APK is malformed or uses a scheme androguard "
            "does not parse"
        ) from exc
    pairs = sorted(
        (str(cert.issuer.human_friendly), str(cert.subject.human_friendly)) for cert in certificates
    )
    return pairs[0] if pairs else (None, None)


def _label(apk: APK, application: Any) -> tuple[str, bool]:
    """`(label, unresolved)`. `get_app_name()` resolves the id against the APK's own resource
    table; a `@…` coming back out means the id points somewhere this APK cannot reach (a
    framework string, or one an RRO supplies at runtime), which is not a name."""
    if _attr(application, "label") is None:
        return LABEL_UNKNOWN, False
    try:
        name = apk.get_app_name()
    except Exception:  # a broken ARSC must not cost the other 20-odd signals
        return LABEL_UNKNOWN, True
    if not name:
        return LABEL_UNKNOWN, False
    if name.startswith("@"):
        return LABEL_UNKNOWN, True
    return name, False


def parse_apk(path: Path, *, partition: str, device_path: str) -> ApkFacts:
    """Turn one extracted APK into facts.

    `partition` and `device_path` come from the extraction (`uadclaw.unpack`), not from the
    manifest: where a package sits is a fact about the image, and the canonical device path is
    what collapses the nested-`system/` root difference between partitions.

    Blocking and CPU-bound. Callers on the event loop run it through `asyncio.to_thread`.
    """
    try:
        apk = APK(str(path))
    except Exception as exc:
        raise ApkParseError(
            f"parse_apk: androguard could not open {device_path} ({path.name}): "
            f"{type(exc).__name__}: {exc}. The APK is malformed or is not an APK; re-extract "
            "the partition if this is unexpected, otherwise let the stage's failure budget "
            "record it and carry on."
        ) from exc

    root = apk.get_android_manifest_xml()
    if root is None:
        raise ApkParseError(
            f"parse_apk: {device_path} has no readable AndroidManifest.xml; androguard opened "
            "the zip but the manifest did not decode"
        )
    package = apk.get_package()
    if not package:
        raise ApkParseError(
            f"parse_apk: {device_path} declares no package name. Every downstream fact is "
            "keyed on it, so an unnamed package cannot be merged or rated."
        )

    application = root.find("application")
    overlay = root.find("overlay")
    label, label_unresolved = _label(apk, application)
    cert_issuer, cert_subject = _certificate_names(apk)
    uses_required, uses_optional = _uses_libraries(application)
    filters = _intent_filters(application)

    return ApkFacts(
        package=package,
        label=label,
        label_unresolved=label_unresolved,
        version_code=_parse_int(root.get(f"{ANDROID_NS}versionCode")),
        partition=partition,
        device_path=device_path,
        priv_app="/priv-app/" in device_path,
        sha256=sha256_file(path),
        cert_issuer=cert_issuer,
        cert_subject=cert_subject,
        # Null namespace on purpose; see the module docstring.
        core_app=_is_true(root.get("coreApp")),
        shared_user_id=_attr(root, "sharedUserId"),
        persistent=_is_true(_attr(application, "persistent")),
        # AOSP's default is true; only an explicit "false" makes a package resource-only.
        has_code=_attr(application, "hasCode") != "false",
        overlay_target=_attr(overlay, "targetPackage"),
        overlay_static=_is_true(_attr(overlay, "isStatic")),
        overlay_priority=_parse_int(_attr(overlay, "priority")),
        libraries=_libraries(application, "library"),
        static_libraries=_libraries(application, "static-library"),
        uses_libraries_required=uses_required,
        uses_libraries_optional=uses_optional,
        protected_broadcasts=_child_names(root, "protected-broadcast"),
        provider_authorities=_provider_authorities(application),
        queries_packages=_child_names(root.find("queries"), "package"),
        intent_filters=filters,
        is_input_method=_declares(filters, "service", INPUT_METHOD_ACTION),
        is_device_admin=_declares(filters, "receiver", DEVICE_ADMIN_ACTION),
        is_accessibility_service=_declares(filters, "service", ACCESSIBILITY_SERVICE_ACTION),
        is_carrier_service=_declares(filters, "service", CARRIER_SERVICE_ACTION),
    )
