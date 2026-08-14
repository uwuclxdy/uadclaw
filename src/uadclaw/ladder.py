"""The removal-rating floor (`docs/pipeline-design.md` §6). Safety-critical.

`removal` reaches real phones through Canta, AppManager and android-debloat-list as well as
uad-ng, so a wrong `Recommended` is the failure this whole pipeline is shaped around. The
ladder answers one question — **how dangerous is removing this package, at minimum** — and the
answer is a lower bound. A later stage (the model, a human) may decide something is MORE
dangerous than the floor. Nothing may decide it is less.

That is enforced by shape rather than by review:

- `Removal` is ordered by `danger_rank()`, never by its own string comparison
  (`"Expert" < "Recommended"` lexicographically, which would invert the whole ladder).
- `compute_floors()` is the ONLY constructor of a `RemovalFloor`, it takes the whole corpus,
  and there is deliberately no per-package variant: three of the ten rules are corpus-wide
  ("sole provider", "sole holder", "sole handler"), and a function that could be called
  without the corpus would silently return a floor that is too low.
- `RemovalFloor` is frozen and validates on construction that its `floor` really is the
  strictest rule that fired, so even a hand-built one cannot understate its own evidence.
- `raise_to_floor()` is the only function that combines a floor with a proposal, and it
  returns `max(proposal, floor)`. There is no expressible operation that returns less than a
  floor: lowering is not "forbidden", it is absent from the API.

Two clauses are stated here as well as in `docs/domain-knowledge.md`, because both are easy to
get backwards:

- **The privileged-permission allowlist is an integration score, never a boot-risk flag.**
  AOSP refuses to boot when a package that is STILL PRESENT requests a privileged permission
  nobody allowlisted. That is a ROM-build error; removing the app removes the request. So
  allowlist membership contributes the same `Advanced` floor that living in `priv-app` does,
  and the permission count travels alongside as context — it never reaches `Unsafe`.
- **The emergency-alert / cellbroadcast / SAR deny-list is hardcoded and is not a model
  call.** The wiki defines `Unsafe` as including packages "illegal to remove in your country
  (e.g. emergency alerts, or SAR certification law in EU)", and no static analysis answers a
  jurisdiction question.
"""

import enum
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from uadclaw.corpus import CorpusPackage
from uadclaw.etcconfig import ConfigInputs

# The shared user id that puts a package inside the platform's own uid.
SYSTEM_SHARED_USER_ID = "android.uid.system"


class Removal(enum.StrEnum):
    """The upstream `removal` tiers, spelled exactly as `uad_lists.json` carries them.

    Definitions, from the upstream wiki (`docs/domain-knowledge.md`):
    `Recommended` safe to uninstall · `Advanced` breaks obscure or minor functionality,
    overlays, or replaceable apps · `Expert` breaks widespread or important functionality but
    nothing vital to the OS · `Unsafe` can break vital parts of the OS, or is illegal to
    remove in your country.
    """

    RECOMMENDED = "Recommended"
    ADVANCED = "Advanced"
    EXPERT = "Expert"
    UNSAFE = "Unsafe"


# Least to most dangerous. The ONLY ordering of `Removal` in this codebase: the enum's own
# comparison is lexicographic on the value, where "Expert" sorts below "Recommended".
DANGER_ORDER: tuple[Removal, ...] = (
    Removal.RECOMMENDED,
    Removal.ADVANCED,
    Removal.EXPERT,
    Removal.UNSAFE,
)
_DANGER_RANK: dict[Removal, int] = {value: index for index, value in enumerate(DANGER_ORDER)}


def danger_rank(removal: Removal) -> int:
    """Position in `DANGER_ORDER`. Higher is more dangerous to remove."""
    return _DANGER_RANK[Removal(removal)]


def strictest(*removals: Removal) -> Removal:
    """The most dangerous of several tiers. Combining floors can only ever move up."""
    if not removals:
        raise ValueError("strictest: needs at least one tier")
    return max(removals, key=danger_rank)


class FloorRule(enum.StrEnum):
    """Which clause of the design's ladder table set a floor. Recorded because the triage card
    shows it: a reviewer has to be able to see why a package is pinned where it is."""

    CORE_APP = "core_app"
    SOLE_LIBRARY_PROVIDER = "sole_library_provider"
    SOLE_STATIC_ROLE_HOLDER = "sole_static_role_holder"
    DENY_LIST = "deny_list"
    SYSTEM_SHARED_UID = "system_shared_uid"
    PERSISTENT = "persistent"
    SOLE_CORE_INTENT_HANDLER = "sole_core_intent_handler"
    PRIVILEGED = "privileged"
    OVERLAY_TARGET = "overlay_target"
    DEFAULT = "default"


# Table order from the design doc, used only to break a tie between two rules that set the
# same floor, so the rule reported for a package is stable across runs.
_RULE_ORDER: dict[FloorRule, int] = {rule: index for index, rule in enumerate(FloorRule)}


@dataclass(frozen=True, slots=True)
class DenyRule:
    """One hardcoded jurisdiction entry. A regex plus the reason it exists, because in a year
    the reason is the only thing that makes the pattern reviewable."""

    pattern: re.Pattern[str]
    reason: str


# Matched against the lower-cased package name. Over-matching here is safe by construction:
# every hit only RAISES a floor, and the tiers it can raise are `Recommended`/`Advanced`.
# Under-matching is the failure that matters, which is why the alert families are substring
# matches while the acronyms are segment-anchored — `wea` and `amber` as bare substrings hit
# "weather", "wearable" and a Motorola wallpaper across the 5372 live upstream entries, and
# `sar` hits the Hungarian word in `telekom.hu.android.mobilvasarlas`.
DENY_LIST: tuple[DenyRule, ...] = (
    DenyRule(
        pattern=re.compile(r"cellbroadcast"),
        reason="cell broadcast is the transport for statutory emergency alerts",
    ),
    DenyRule(
        pattern=re.compile(r"emergency"),
        reason="emergency calling and alerting",
    ),
    DenyRule(
        pattern=re.compile(r"(^|\.)(cmas|etws|wea)(\.|$)"),
        reason="statutory alert systems (CMAS/WEA in the US, ETWS in Japan)",
    ),
    DenyRule(
        pattern=re.compile(r"(^|\.)amber(\.|$)"),
        reason="AMBER alerts",
    ),
    DenyRule(
        pattern=re.compile(r"(^|\.)(sar[a-z0-9]*|[a-z0-9]*sar)(\.|$)"),
        reason="SAR/specific-absorption-rate control, certification law in the EU",
    ),
)


def deny_list_reason(package: str) -> str | None:
    """The reason this package is legally risky to remove, or None."""
    name = package.lower()
    for rule in DENY_LIST:
        if rule.pattern.search(name):
            return rule.reason
    return None


@dataclass(frozen=True, slots=True)
class FiredRule:
    rule: FloorRule
    floor: Removal
    detail: str

    def as_json(self) -> dict[str, Any]:
        return {"rule": str(self.rule), "floor": str(self.floor), "detail": self.detail}


@dataclass(frozen=True, slots=True)
class RemovalFloor:
    """The lower bound on one package's `removal`, and every rule behind it.

    Validated on construction: `floor` must equal the strictest tier in `fired`. A floor that
    understates its own evidence is therefore not a value this type can hold, whoever builds
    it and however they build it.
    """

    package: str
    floor: Removal
    rule: FloorRule
    fired: tuple[FiredRule, ...]
    privapp_allowlisted: bool = False
    privapp_permission_count: int = 0

    def __post_init__(self) -> None:
        if not self.fired:
            raise ValueError(
                f"RemovalFloor({self.package}): a floor with no rule behind it cannot be "
                "audited; the default rule always fires, so an empty tuple is a bug"
            )
        implied = strictest(*(item.floor for item in self.fired))
        if self.floor != implied:
            raise ValueError(
                f"RemovalFloor({self.package}): floor {self.floor} does not match the "
                f"strictest rule that fired ({implied} via "
                f"{[str(item.rule) for item in self.fired]}). A floor below its own evidence "
                "is a safety regression, not a tuning choice."
            )

    def as_json(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "floor": str(self.floor),
            "rule": str(self.rule),
            "fired": [item.as_json() for item in self.fired],
            "privapp_allowlisted": self.privapp_allowlisted,
            "privapp_permission_count": self.privapp_permission_count,
        }


def raise_to_floor(proposal: Removal, floor: RemovalFloor) -> Removal:
    """The only way a proposed rating and a computed floor combine.

    Returns the more dangerous of the two, so a model or a human may raise a rating and can
    never lower one. There is deliberately no inverse: nothing in this module returns a tier
    below a floor, which is what makes "the LLM cannot land below its floor" a property of the
    API rather than a rule someone has to remember.
    """
    return strictest(Removal(proposal), floor.floor)


def is_below_floor(proposal: Removal, floor: RemovalFloor) -> bool:
    """Whether a proposal would have to be raised. The classification stage rejects rather
    than clamps, so it needs to see the difference."""
    return danger_rank(Removal(proposal)) < danger_rank(floor.floor)


def _base_rules(
    item: CorpusPackage,
    *,
    config: ConfigInputs,
    sole_library_consumers: dict[str, tuple[str, ...]],
    sole_static_roles: dict[str, tuple[str, ...]],
    sole_intent_surfaces: dict[str, tuple[str, ...]],
) -> list[FiredRule]:
    """Every rule except the overlay clause, which needs the other packages' base floors."""
    fired: list[FiredRule] = [
        FiredRule(rule=FloorRule.DEFAULT, floor=Removal.RECOMMENDED, detail="no rule fired")
    ]

    if item.core_app:
        fired.append(
            FiredRule(
                rule=FloorRule.CORE_APP,
                floor=Removal.UNSAFE,
                detail='coreApp="true": AOSP puts it in the minimalist boot environment',
            )
        )
    consumers = sole_library_consumers.get(item.package, ())
    if consumers:
        fired.append(
            FiredRule(
                rule=FloorRule.SOLE_LIBRARY_PROVIDER,
                floor=Removal.UNSAFE,
                detail=f"sole corpus provider of a required library for {', '.join(consumers)}",
            )
        )
    roles = sole_static_roles.get(item.package, ())
    if roles:
        fired.append(
            FiredRule(
                rule=FloorRule.SOLE_STATIC_ROLE_HOLDER,
                floor=Removal.UNSAFE,
                detail=f"sole corpus holder of static role(s) {', '.join(roles)}",
            )
        )
    reason = deny_list_reason(item.package)
    if reason is not None:
        fired.append(FiredRule(rule=FloorRule.DENY_LIST, floor=Removal.UNSAFE, detail=reason))
    # `sharedUserMaxSdkVersion` is deliberately ignored: above its bound the app behaves as if
    # the sharedUserId were never defined, but a device still running an SDK at or below it
    # keeps the full shared-uid privilege, so the floor cannot depend on the attribute. Never
    # measured whether real firmware uses it.
    if item.shared_user_id == SYSTEM_SHARED_USER_ID:
        fired.append(
            FiredRule(
                rule=FloorRule.SYSTEM_SHARED_UID,
                floor=Removal.EXPERT,
                detail=f'sharedUserId="{SYSTEM_SHARED_USER_ID}"',
            )
        )
    if item.persistent:
        fired.append(
            FiredRule(
                rule=FloorRule.PERSISTENT,
                floor=Removal.EXPERT,
                detail='persistent="true": the platform keeps it running',
            )
        )
    surfaces = sole_intent_surfaces.get(item.package, ())
    if surfaces:
        fired.append(
            FiredRule(
                rule=FloorRule.SOLE_CORE_INTENT_HANDLER,
                floor=Removal.EXPERT,
                detail=f"sole corpus handler of {', '.join(surfaces)}",
            )
        )
    allowlisted = config.privapp_allowlisted(item.package)
    if item.priv_app or allowlisted:
        where = "priv-app" if item.priv_app else "privapp-permissions allowlist"
        detail = f"privileged ({where})"
        if allowlisted:
            # Integration score, not a boot-risk flag; see the module docstring.
            detail += f", {config.permission_count(item.package)} allowlisted permission(s)"
        fired.append(FiredRule(rule=FloorRule.PRIVILEGED, floor=Removal.ADVANCED, detail=detail))
    return fired


def _sole_library_consumers(corpus: Sequence[CorpusPackage]) -> dict[str, tuple[str, ...]]:
    """Provider -> the consumers that would break, for libraries it is the only source of."""
    providers: dict[str, list[str]] = {}
    for item in corpus:
        for library in item.provided_libraries:
            providers.setdefault(library, []).append(item.package)
    consumers: dict[str, list[str]] = {}
    for item in corpus:
        for library in item.uses_libraries_required:
            sources = [name for name in providers.get(library, ()) if name != item.package]
            if len(sources) == 1:
                consumers.setdefault(sources[0], []).append(item.package)
    return {provider: tuple(sorted(set(names))) for provider, names in consumers.items()}


def _sole_static_roles(
    corpus: Sequence[CorpusPackage], config: ConfigInputs
) -> dict[str, tuple[str, ...]]:
    """Package -> static roles it holds with no alternative holder in the corpus."""
    names = {item.package for item in corpus}
    sole: dict[str, list[str]] = {}
    for role, holders in config.static_role_holders.items():
        present = sorted(name for name in holders if name in names)
        if len(present) == 1:
            sole.setdefault(present[0], []).append(role)
    return {package: tuple(sorted(roles)) for package, roles in sole.items()}


def _sole_intent_surfaces(corpus: Sequence[CorpusPackage]) -> dict[str, tuple[str, ...]]:
    """Package -> the core intent surfaces it is the corpus's only handler of."""
    surfaces = {
        "HOME": [item.package for item in corpus if item.handles_home],
        "DIALER": [item.package for item in corpus if item.handles_dialer],
        "SMS_DELIVER": [item.package for item in corpus if item.handles_sms_deliver],
        "IME": [item.package for item in corpus if item.is_input_method],
    }
    sole: dict[str, list[str]] = {}
    for surface, handlers in surfaces.items():
        unique = sorted(set(handlers))
        if len(unique) == 1:
            sole.setdefault(unique[0], []).append(surface)
    return {package: tuple(sorted(found)) for package, found in sole.items()}


def _ordered(fired: Sequence[FiredRule]) -> tuple[FiredRule, ...]:
    return tuple(sorted(fired, key=lambda item: (-danger_rank(item.floor), _RULE_ORDER[item.rule])))


def compute_floors(
    corpus: Sequence[CorpusPackage], *, config: ConfigInputs | None = None
) -> dict[str, RemovalFloor]:
    """The floor for every package in the corpus, keyed by package name.

    Takes the whole corpus because three rules are corpus-wide and cannot be answered from one
    manifest: "sole provider of a required library", "sole holder of a static role" and "sole
    handler of a core intent" are all statements about what else is installed.

    The overlay clause runs as a second pass over the first pass's base floors: an overlay
    inherits its target's floor capped at `Advanced`, and reading the target's BASE floor
    (rather than its final one) is what keeps an overlay-of-an-overlay from chasing a chain or
    a cycle. The cap can only ever raise this package's floor, never lower it, because floors
    combine with `strictest`.
    """
    config = config or ConfigInputs()
    sole_library_consumers = _sole_library_consumers(corpus)
    sole_static_roles = _sole_static_roles(corpus, config)
    sole_intent_surfaces = _sole_intent_surfaces(corpus)

    base: dict[str, list[FiredRule]] = {}
    for item in corpus:
        base[item.package] = _base_rules(
            item,
            config=config,
            sole_library_consumers=sole_library_consumers,
            sole_static_roles=sole_static_roles,
            sole_intent_surfaces=sole_intent_surfaces,
        )
    base_floor = {
        package: strictest(*(rule.floor for rule in rules)) for package, rules in base.items()
    }

    floors: dict[str, RemovalFloor] = {}
    for item in corpus:
        fired = list(base[item.package])
        target = item.overlay_target
        if target and target in base_floor and target != item.package:
            inherited = min(base_floor[target], Removal.ADVANCED, key=danger_rank)
            fired.append(
                FiredRule(
                    rule=FloorRule.OVERLAY_TARGET,
                    floor=inherited,
                    detail=f"overlay of {target} (its floor {base_floor[target]}, capped at "
                    f"{Removal.ADVANCED})",
                )
            )
        ordered = _ordered(fired)
        floors[item.package] = RemovalFloor(
            package=item.package,
            floor=ordered[0].floor,
            rule=ordered[0].rule,
            fired=ordered,
            privapp_allowlisted=config.privapp_allowlisted(item.package),
            privapp_permission_count=config.permission_count(item.package),
        )
    return floors
