"""Facts into Postgres: one observation per APK, one merged row per package name.

The merge is the part with teeth. `docs/pipeline-design.md` §3 keys facts by package name and
merges them across devices, and the count of distinct devices is the primary triage ranking
signal — so the interesting failures are not parse errors but silent ones: two devices'
identical package collapsing into two rows (the ranking then reads 1 and 1 where it should
read 2), or two genuinely different packages sharing a name and collapsing into one row whose
signing certificate is whichever device happened to be scanned last.

Both are addressed structurally rather than by care:

- the merged row is **recomputed from every observation** of that package, never folded
  incrementally into whatever was already there. Re-scanning a build is then an upsert onto
  the same observation rows and a recompute to the same merged values, which is what makes
  "the same build parsed twice yields identical facts" a property rather than a hope;
- every merge rule is either order-independent (sticky-true, union, max) or explicitly ordered
  by `(device_key, build, device_path)`. None of them is "the last writer wins";
- a disagreement on any `CONFLICT_FIELDS` field sets `has_conflict` and is recorded with every
  value and the devices that carried it.
"""

import json
import logging
import uuid
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from uadclaw.facts import LABEL_UNKNOWN, ApkFacts
from uadclaw.models import DeviceScan, PackageFact, PackageObservation

logger = logging.getLogger(__name__)


class FactMergeError(RuntimeError):
    """The merge was asked for something it cannot produce, e.g. a package with no
    observations. A bug here, not bad firmware."""


# Danger signals: one device declaring it is enough. Merging these any other way could lower a
# package's rule-ladder floor below what some device's own manifest justifies, which the
# repo's first rule forbids. `has_code` joins them because "has code" is the cautious reading.
STICKY_TRUE_FIELDS: tuple[str, ...] = (
    "priv_app",
    "core_app",
    "persistent",
    "has_code",
    "overlay_static",
    "is_input_method",
    "is_device_admin",
    "is_accessibility_service",
    "is_carrier_service",
)

# Evidence that is additive: a protected broadcast declared on one device is still evidence
# about the package even if another device's build dropped it.
UNION_FIELDS: tuple[str, ...] = (
    "libraries",
    "static_libraries",
    "uses_libraries_required",
    "uses_libraries_optional",
    "protected_broadcasts",
    "provider_authorities",
    "queries_packages",
    "intent_filters",
)

# Identity scalars: one value, taken from the lowest `(device_key, build, device_path)`.
FIRST_WINS_FIELDS: tuple[str, ...] = (
    "cert_issuer",
    "cert_subject",
    "shared_user_id",
    "overlay_target",
)

# Where two devices disagreeing means the merged row must not be trusted as one package.
# Certificates and `sharedUserId` are identity claims — a different signer is a different
# package wearing a taken name. `coreApp`, `persistent` and the overlay target are
# rule-ladder floor inputs, so a disagreement means the floor itself is device-dependent and
# a single rating cannot be honest about it.
#
# A field is conflicting only when two devices state DIFFERENT values. One device stating
# nothing is absence of evidence, not contradiction, and does not raise the flag.
#
# Measured on the two local corpora: 134 of the 147 packages shared by a Pixel 6 and the
# Android 16 emulator disagree on the certificate, every one of them APK signature v3 key
# rotation or the AOSP test key facing Google's production key. The flag is therefore a
# "show the reviewer both values" marker, not a suspicion score, and the triage screen renders it
# rather than filter on it.
CONFLICT_FIELDS: tuple[str, ...] = (
    "cert_issuer",
    "cert_subject",
    "shared_user_id",
    "core_app",
    "persistent",
    "overlay_target",
)

# The launcher icon. Merged first-wins by `(device_key, build, device_path)` like the identity
# scalars above, with one difference: a device that shipped the package without a renderable
# icon must not hide the icon another device shipped. That is the rule `label` already follows
# and for the same reason — strict first-wins would let one silent device blank the column.
#
# "First" is the LOWEST-sorting build, so re-scanning a phone at a newer build leaves the older
# build's artwork in place: `package_observations` never deletes, and the older row keeps
# winning. Deliberate rather than overlooked — `label` and every FIRST_WINS_FIELDS entry behave
# identically, and making the icon rank by recency would make it the one field on the row that
# disagrees with the rest.
ICON_FIELDS: tuple[str, ...] = ("icon_bytes", "icon_mime")

_SCALAR_OBSERVATION_FIELDS: tuple[str, ...] = (
    "label",
    "label_unresolved",
    "version_code",
    "partition",
    "device_path",
    "sha256",
    "cert_issuer",
    "cert_subject",
    "shared_user_id",
    "overlay_target",
    "overlay_priority",
    *ICON_FIELDS,
    *STICKY_TRUE_FIELDS,
)


def observation_row(
    facts: ApkFacts, *, device_key: str, build: str, observed_at: datetime
) -> dict[str, Any]:
    """One APK's facts as a `package_observations` row."""
    return {
        "device_key": device_key,
        "build": build,
        "package": facts.package,
        "observed_at": observed_at,
        "label": facts.label,
        "label_unresolved": facts.label_unresolved,
        "version_code": facts.version_code,
        "partition": facts.partition,
        "device_path": facts.device_path,
        "priv_app": facts.priv_app,
        "sha256": facts.sha256,
        "cert_issuer": facts.cert_issuer,
        "cert_subject": facts.cert_subject,
        "core_app": facts.core_app,
        "shared_user_id": facts.shared_user_id,
        "persistent": facts.persistent,
        "has_code": facts.has_code,
        "overlay_target": facts.overlay_target,
        "overlay_static": facts.overlay_static,
        "overlay_priority": facts.overlay_priority,
        "icon_bytes": facts.icon_bytes,
        "icon_mime": facts.icon_mime,
        "is_input_method": facts.is_input_method,
        "is_device_admin": facts.is_device_admin,
        "is_accessibility_service": facts.is_accessibility_service,
        "is_carrier_service": facts.is_carrier_service,
        "libraries": [library.as_json() for library in facts.libraries],
        "static_libraries": [library.as_json() for library in facts.static_libraries],
        "uses_libraries_required": list(facts.uses_libraries_required),
        "uses_libraries_optional": list(facts.uses_libraries_optional),
        "protected_broadcasts": list(facts.protected_broadcasts),
        "provider_authorities": list(facts.provider_authorities),
        "queries_packages": list(facts.queries_packages),
        "intent_filters": [item.as_json() for item in facts.intent_filters],
    }


def _sort_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (str(row["device_key"]), str(row["build"]), str(row["device_path"]))


def _canonical(value: Any) -> str:
    """A stable text key for a JSON-able value, so a union of dicts dedups and sorts the same
    way on every run and on every machine."""
    return json.dumps(value, sort_keys=True, default=str)


def _union(rows: Sequence[Mapping[str, Any]], field: str) -> list[Any]:
    seen: dict[str, Any] = {}
    for row in rows:
        for value in row.get(field) or ():
            seen.setdefault(_canonical(value), value)
    return [seen[key] for key in sorted(seen)]


def _conflicts(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for field in CONFLICT_FIELDS:
        carriers: dict[str, tuple[Any, set[str]]] = {}
        for row in rows:
            value = row.get(field)
            if value is None:
                continue
            key = _canonical(value)
            carriers.setdefault(key, (value, set()))[1].add(str(row["device_key"]))
        if len(carriers) < 2:
            continue
        found.append(
            {
                "field": field,
                "values": [
                    {"value": carriers[key][0], "devices": sorted(carriers[key][1])}
                    for key in sorted(carriers)
                ],
            }
        )
    return found


def merge_observations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Fold every observation of one package into its `package_facts` row.

    Pure and total over its input: hand it the same observations in any order and it returns
    the same row, which is exactly the property the reproducibility and merge tests pin.
    """
    if not rows:
        raise FactMergeError("merge_observations: a package with no observations has no facts")
    packages = {str(row["package"]) for row in rows}
    if len(packages) != 1:
        raise FactMergeError(
            f"merge_observations: observations for {sorted(packages)} were handed in together; "
            "the merge is per package name and cannot fold two of them into one row"
        )

    ordered = sorted(rows, key=_sort_key)
    first = ordered[0]
    devices = sorted({str(row["device_key"]) for row in rows})
    version_codes = [row["version_code"] for row in rows if row.get("version_code") is not None]
    # A declared label beats "unknown" wherever it appears, then the ordering decides. The
    # alternative (strict first-wins) would let one device that never declared a label hide a
    # name every other device carries.
    labelled = [row for row in ordered if row.get("label") not in (None, LABEL_UNKNOWN)]
    label_source = labelled[0] if labelled else first
    iconed = [row for row in ordered if row.get("icon_bytes")]
    conflicts = _conflicts(rows)

    merged: dict[str, Any] = {
        "package": next(iter(packages)),
        "device_count": len(devices),
        "devices": devices,
        "first_seen_at": min(row["observed_at"] for row in rows),
        "last_seen_at": max(row["observed_at"] for row in rows),
        "label": label_source.get("label", LABEL_UNKNOWN),
        "label_unresolved": bool(label_source.get("label_unresolved")),
        "version_code": max(version_codes) if version_codes else None,
        "partitions": sorted({str(row["partition"]) for row in rows}),
        "has_conflict": bool(conflicts),
        "conflicts": conflicts,
    }
    for field in ICON_FIELDS:
        merged[field] = iconed[0].get(field) if iconed else None
    for field in STICKY_TRUE_FIELDS:
        merged[field] = any(bool(row.get(field)) for row in rows)
    for field in FIRST_WINS_FIELDS:
        merged[field] = first.get(field)
    for field in UNION_FIELDS:
        merged[field] = _union(ordered, field)
    return merged


async def record_device_scan(
    session: AsyncSession,
    *,
    job_id: uuid.UUID | None,
    device_key: str,
    build: str,
    scanned_at: datetime,
    apk_total: int,
    parsed_ok: int,
    failures: Sequence[Mapping[str, Any]],
) -> None:
    session.add(
        DeviceScan(
            job_id=job_id,
            device_key=device_key,
            build=build,
            scanned_at=scanned_at,
            apk_total=apk_total,
            parsed_ok=parsed_ok,
            parse_failed=len(failures),
            failures=[dict(failure) for failure in failures],
        )
    )


async def store_device_facts(
    session: AsyncSession,
    *,
    device_key: str,
    build: str,
    facts: Iterable[ApkFacts],
    observed_at: datetime,
) -> list[str]:
    """Upsert this device's observations, then recompute every package they touch.

    Returns the package names whose merged row was rewritten.
    """
    rows = [
        observation_row(item, device_key=device_key, build=build, observed_at=observed_at)
        for item in facts
    ]
    if not rows:
        return []

    statement = pg_insert(PackageObservation).values(rows)
    statement = statement.on_conflict_do_update(
        constraint="uq_package_observations_device_path",
        # `observed_at` is deliberately absent: it is when this APK was FIRST seen, so a
        # re-scan of the same build rewrites the facts (a parser fix must land) without
        # moving a timestamp, and the merged row comes out byte-identical.
        set_={
            column: statement.excluded[column]
            for column in (*_SCALAR_OBSERVATION_FIELDS, *UNION_FIELDS, "package")
        },
    )
    await session.execute(statement)

    packages = sorted({row["package"] for row in rows})
    await recompute_package_facts(session, packages)
    return packages


async def recompute_package_facts(session: AsyncSession, packages: Sequence[str]) -> None:
    """Rebuild the merged row of each named package from all of its observations."""
    if not packages:
        return
    result = await session.execute(
        select(PackageObservation).where(PackageObservation.package.in_(list(packages)))
    )
    by_package: dict[str, list[dict[str, Any]]] = {}
    for observation in result.scalars():
        by_package.setdefault(observation.package, []).append(
            {
                "package": observation.package,
                "device_key": observation.device_key,
                "build": observation.build,
                "observed_at": observation.observed_at,
                "partition": observation.partition,
                "device_path": observation.device_path,
                "label": observation.label,
                "label_unresolved": observation.label_unresolved,
                "version_code": observation.version_code,
                "cert_issuer": observation.cert_issuer,
                "cert_subject": observation.cert_subject,
                "shared_user_id": observation.shared_user_id,
                "overlay_target": observation.overlay_target,
                **{field: getattr(observation, field) for field in ICON_FIELDS},
                **{field: getattr(observation, field) for field in STICKY_TRUE_FIELDS},
                **{field: getattr(observation, field) for field in UNION_FIELDS},
            }
        )

    merged_rows = [merge_observations(rows) for _, rows in sorted(by_package.items())]
    if not merged_rows:
        return
    statement = pg_insert(PackageFact).values(merged_rows)
    await session.execute(
        statement.on_conflict_do_update(
            index_elements=[PackageFact.package],
            set_={
                column: statement.excluded[column]
                for column in merged_rows[0]
                if column != "package"
            },
        )
    )
    conflicted = [row["package"] for row in merged_rows if row["has_conflict"]]
    if conflicted:
        logger.warning(
            "%d package(s) merged with a cross-device conflict and are flagged for review: %s",
            len(conflicted),
            ", ".join(conflicted[:10]),
        )
