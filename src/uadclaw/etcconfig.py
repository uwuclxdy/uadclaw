"""The rule-ladder inputs that live in `/etc` and in no APK.

`unpack` keeps `/etc/permissions/*.xml`, `/etc/sysconfig/*`, `/etc/default-permissions/*.xml`
and `roles.xml` past retention precisely because three ladder inputs are only stated there:

- **privileged-permission allowlists.** `<privapp-permissions package="X">` — read as an
  INTEGRATION score, never as a boot-risk flag. AOSP refuses to boot when a package that is
  still present asks for a privileged permission nobody allowlisted; that is a ROM-build
  error, and removing the app removes the request. Inverting this clause is the single
  easiest correctness mistake in the ladder, so it is stated here as well as in the ladder.
- **static roles.** `<role name="…" static="true" defaultHolders="a;b">` — a static role is
  always assigned to its default holders, so a sole holder has no user-selectable
  alternative. Absent from both local corpora (no `roles.xml` in either), which is normal;
  its absence is not an extraction failure.
- **platform-provided Java shared libraries.** `<library name="android.test.base" …>` in
  `/etc/permissions/*.xml`. These are why most `<uses-library required="true">` names resolve
  to no package in the corpus: the provider is the platform, not a removable APK. Recorded so
  that "0 library edges" reads as a measured fact rather than as a broken lookup.

Two `/etc/sysconfig` signals were considered and deliberately not consumed:
`allow-in-power-save` (a Doze exemption, so removing such a package changes power management,
not just app availability) and `system-user-blacklisted-app` (a multi-user disable
declaration, not a removal blocklist — conflating it with one would misread an OEM disable
list as a debloat signal). The extracted config files still carry both; this note is what
stops a future reader from wiring them in as ladder inputs.

Dispatch is on the ELEMENT, never on the filename: AOSP states device implementers may choose
their own file layout as long as every `priv-app` package is allowlisted, so a vendor naming
its allowlist something unexpected must not silently produce an empty one.

Two structural guards, because these files come out of a vendor firmware image:

- a per-file byte ceiling, and
- a hard refusal of any document carrying a DTD. `xml.etree` expands internal entities, which
  is the whole "billion laughs" class; no legitimate AOSP config XML has a `<!DOCTYPE>`
  (verified across all 246 config files of a real Pixel build and the emulator's), so
  refusing one costs nothing and removes the class without a new dependency.
"""

import logging
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

logger = logging.getLogger(__name__)

# Real files are 1-200 KB; `privapp-permissions-google.xml` on a Pixel is ~90 KB.
MAX_CONFIG_XML_BYTES = 8 * 1024**2

# Enough of the head to carry a `<?xml?>` declaration, comments and a DTD.
_DTD_SNIFF_BYTES = 8 * 1024
_DTD_PATTERN = re.compile(rb"<!DOCTYPE|<!ENTITY", re.IGNORECASE)

# Separator AOSP uses inside `defaultHolders` when a role is not exclusive.
_ROLE_HOLDER_SEPARATOR = ";"


class ConfigParseError(RuntimeError):
    """One config XML could not be read. Bad input from a firmware image, not a bug here:
    the caller records it and carries on, the same policy an unreadable APK gets."""


@dataclass(frozen=True, slots=True)
class ConfigInputs:
    """One device's `/etc` rule inputs, or several devices' unioned.

    Every field is additive across devices, so the union of two devices can never withdraw a
    signal one of them stated — which is the same reason `factstore` merges danger flags
    sticky-true: a merge that could drop a signal could lower a floor.
    """

    # package -> the privileged permissions explicitly granted to it. Denies are not counted:
    # the score being built is "how deeply is this wired into the platform".
    privapp_permissions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # role name -> default holders, for STATIC roles only. A non-static role is user
    # reassignable and therefore not a floor input.
    static_role_holders: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    # Shared-library names the platform itself provides through /etc/permissions.
    platform_libraries: frozenset[str] = frozenset()
    files_read: int = 0
    files_failed: tuple[str, ...] = ()

    def privapp_allowlisted(self, package: str) -> bool:
        return package in self.privapp_permissions

    def permission_count(self, package: str) -> int:
        return len(self.privapp_permissions.get(package, ()))

    def holders_of_static_role(self, package: str) -> tuple[str, ...]:
        """Every static role `package` is a default holder of."""
        return tuple(
            sorted(role for role, holders in self.static_role_holders.items() if package in holders)
        )

    def as_json(self) -> dict[str, Any]:
        return {
            "privapp_permissions": {
                package: list(permissions)
                for package, permissions in sorted(self.privapp_permissions.items())
            },
            "static_role_holders": {
                role: list(holders) for role, holders in sorted(self.static_role_holders.items())
            },
            "platform_libraries": sorted(self.platform_libraries),
            "files_read": self.files_read,
            "files_failed": list(self.files_failed),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any] | None) -> "ConfigInputs":
        if not payload:
            return cls()
        return cls(
            privapp_permissions={
                str(package): tuple(str(item) for item in permissions)
                for package, permissions in (payload.get("privapp_permissions") or {}).items()
            },
            static_role_holders={
                str(role): tuple(str(item) for item in holders)
                for role, holders in (payload.get("static_role_holders") or {}).items()
            },
            platform_libraries=frozenset(
                str(name) for name in (payload.get("platform_libraries") or ())
            ),
            files_read=int(payload.get("files_read") or 0),
            files_failed=tuple(str(name) for name in (payload.get("files_failed") or ())),
        )


def merge_config_inputs(items: Iterable[ConfigInputs]) -> ConfigInputs:
    """Union several devices' inputs. Order-independent and idempotent: the same devices in
    any order produce the same value, so a corpus-wide ladder run does not depend on which
    device was scanned first."""
    permissions: dict[str, set[str]] = {}
    roles: dict[str, set[str]] = {}
    libraries: set[str] = set()
    files_read = 0
    files_failed: set[str] = set()
    for item in items:
        for package, granted in item.privapp_permissions.items():
            permissions.setdefault(package, set()).update(granted)
        for role, holders in item.static_role_holders.items():
            roles.setdefault(role, set()).update(holders)
        libraries.update(item.platform_libraries)
        files_read += item.files_read
        files_failed.update(item.files_failed)
    return ConfigInputs(
        privapp_permissions={
            package: tuple(sorted(granted)) for package, granted in sorted(permissions.items())
        },
        static_role_holders={
            role: tuple(sorted(holders)) for role, holders in sorted(roles.items())
        },
        platform_libraries=frozenset(libraries),
        files_read=files_read,
        files_failed=tuple(sorted(files_failed)),
    )


def config_xml_paths(artifacts_dir: Path) -> list[Path]:
    """Every `.xml` under an extracted artifacts tree, sorted.

    `os.walk(followlinks=False)`, like the APK walk: an image's own symlink must not walk the
    host filesystem, and on `rglob` that is an interpreter default rather than a statement.
    """
    found: list[Path] = []
    for root, _dirs, files in os.walk(artifacts_dir, followlinks=False):
        for filename in files:
            if not filename.endswith(".xml"):
                continue
            path = Path(root) / filename
            if path.is_symlink() or not path.is_file():
                continue
            found.append(path)
    return sorted(found)


def _read_document(path: Path) -> ElementTree.Element:
    size = path.stat().st_size
    if size > MAX_CONFIG_XML_BYTES:
        raise ConfigParseError(
            f"_read_document: {path} is {size} bytes, over the {MAX_CONFIG_XML_BYTES}-byte "
            "ceiling for a config XML; refusing to parse it"
        )
    raw = path.read_bytes()
    if _DTD_PATTERN.search(raw[:_DTD_SNIFF_BYTES]):
        raise ConfigParseError(
            f"_read_document: {path} declares a DTD. No AOSP config XML does, and entity "
            "expansion is the whole billion-laughs class, so it is refused unparsed."
        )
    try:
        return ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        raise ConfigParseError(f"_read_document: {path} is not well-formed XML: {exc}") from exc


def _collect(root: ElementTree.Element, permissions, roles, libraries) -> None:
    """Pull every recognised element out of one document, wherever it sits.

    `iter()` rather than a fixed path: the allowlist elements are nested under `<permissions>`
    on every build measured, but AOSP lets an implementer choose the file structure, and a
    layout assumption that silently matches nothing is exactly the failure mode this stage
    cannot afford.
    """
    for element in root.iter("privapp-permissions"):
        package = element.get("package")
        if not package:
            continue
        granted = permissions.setdefault(package, set())
        for permission in element.iterfind("permission"):
            name = permission.get("name")
            if name:
                granted.add(name)
    for element in root.iter("role"):
        # Absent `static` means reassignable, which is not a floor input.
        if element.get("static") != "true":
            continue
        name = element.get("name")
        holders = element.get("defaultHolders")
        if not name or not holders:
            continue
        roles.setdefault(name, set()).update(
            holder.strip() for holder in holders.split(_ROLE_HOLDER_SEPARATOR) if holder.strip()
        )
    for element in root.iter("library"):
        name = element.get("name")
        if name:
            libraries.add(name)


def parse_config_inputs(artifacts_dir: Path, paths: Sequence[Path] | None = None) -> ConfigInputs:
    """Read one device's extracted `/etc` XMLs into ladder inputs.

    A file that cannot be read is recorded and skipped rather than failing the device: these
    are vendor-authored files out of a firmware image, and the only floor a missing allowlist
    entry can cost is `Advanced`, which the package's own `priv-app` location already
    supplies. A file that fails is still named, so "the allowlist looks empty" is always
    distinguishable from "the allowlist did not parse".

    Blocking IO; callers on the event loop run it through `asyncio.to_thread`.
    """
    permissions: dict[str, set[str]] = {}
    roles: dict[str, set[str]] = {}
    libraries: set[str] = set()
    failed: list[str] = []
    candidates = list(paths) if paths is not None else config_xml_paths(artifacts_dir)

    for path in candidates:
        try:
            _collect(_read_document(path), permissions, roles, libraries)
        except (ConfigParseError, OSError):
            # Full traceback before it is reduced to a name in a list, never swallowed.
            logger.exception("config XML %s could not be read", path)
            failed.append(str(path.relative_to(artifacts_dir)))

    return ConfigInputs(
        privapp_permissions={
            package: tuple(sorted(granted)) for package, granted in sorted(permissions.items())
        },
        static_role_holders={
            role: tuple(sorted(holders)) for role, holders in sorted(roles.items())
        },
        platform_libraries=frozenset(libraries),
        files_read=len(candidates) - len(failed),
        files_failed=tuple(failed),
    )
