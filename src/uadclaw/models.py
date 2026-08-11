"""Job, scratch-lease, stage-run and extracted-fact tables.

The job substrate every later pipeline stage runs on top of, plus the facts task 4 extracts
and the survivors of retention: firmware and APKs are deleted after a successful analysis, so
`package_observations` and `package_facts` are the only thing left of a device once its job
finishes.
"""

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from uadclaw.db import Base


class JobKind(enum.StrEnum):
    FIRMWARE_ANALYSIS = "firmware_analysis"


# Whether a job kind occupies the scratch lease. FIRMWARE_ANALYSIS unpacks firmware onto
# disk and needs the single-occupant lease for its whole life; classification (task 7) and
# corroboration (task 8) only touch the DB and the network, never scratch. The requirement
# lives on the kind, not on the worker loop, so the pool stays meaningful (able to run
# scratch-free kinds concurrently) the moment one of those kinds lands.
JOB_KIND_NEEDS_SCRATCH: dict[JobKind, bool] = {
    JobKind.FIRMWARE_ANALYSIS: True,
}


class JobState(enum.StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# Terminal states a claim/reclaim sweep must never touch.
TERMINAL_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED})

# Ordered pipeline stages from docs/pipeline-design.md § Pipeline. A job's `stage` records
# the furthest point it reached; resuming restarts from that stage, never from the start.
PIPELINE_STAGES: tuple[str, ...] = (
    "acquire",
    "unpack",
    "extract_facts",
    "corpus_graph",
    "filter",
    "rule_ladder",
    "llm",
    "corroborate",
    "triage",
    "branch",
)

# Bound the log tail kept on the job row; older lines fall off rather than growing the row
# without limit over a long-running or oft-retried job.
LOG_TAIL_MAX_CHARS = 8_000


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    # What this job is FOR, shaped by its kind — for FIRMWARE_ANALYSIS, which driver, device
    # and build to acquire (see `FirmwareJobParams`). JSONB rather than a firmware-specific
    # column set: every later job kind (classification, corroboration) carries a different
    # target, and none of them wants the others' columns nullable on its rows. Validated
    # against a per-kind model at creation, so nothing here is trusted at read time either.
    params: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    state: Mapped[JobState] = mapped_column(
        SAEnum(JobState, name="job_state", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=JobState.QUEUED,
    )
    # Furthest pipeline stage reached; NULL means "not started yet". Resume restarts here.
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attempt: Mapped[int] = mapped_column(nullable=False, default=0)
    worker_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    log_tail: Mapped[str] = mapped_column(Text, nullable=False, default="")
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Refreshed by the running worker; a job stuck CLAIMED/RUNNING past the stale threshold
    # with no heartbeat is reclaimable (its dead worker's crash is the whole point).
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Set once this job's scratch artifacts are gone from disk — on success (retention:
    # facts-only) or once the failure-ceiling sweep evicts it. NULL means artifacts, if any,
    # are still on disk.
    artifacts_deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("ix_jobs_state_created_at", "state", "created_at"),
        Index("ix_jobs_state_heartbeat_at", "state", "heartbeat_at"),
    )


class JobStageRun(Base):
    """One row per (job, stage) attempt. Exists because per-stage duration cannot be
    derived from the four whole-job timestamps on `Job` — those cover the job, not each
    stage inside it."""

    __tablename__ = "job_stage_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    outcome: Mapped[str | None] = mapped_column(String(16), nullable=True)  # succeeded | failed

    __table_args__ = (Index("ix_job_stage_runs_job_id", "job_id"),)


class ScratchLease(Base):
    """Single-row table: scratch is a singleton resource, so the lease is a singleton row
    (id fixed at 1) rather than a set of rows to query for "the" holder."""

    __tablename__ = "scratch_lease"

    id: Mapped[int] = mapped_column(primary_key=True)
    holder_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    holder_worker_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    acquired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (CheckConstraint("id = 1", name="ck_scratch_lease_singleton"),)


class ScratchLeaseEvent(Base):
    """Append-only history the mutable `ScratchLease` row cannot answer questions from once
    overwritten: occupancy over time and per-job wait duration both need every past
    acquire/release/reclaim, not just the current holder."""

    __tablename__ = "scratch_lease_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    worker_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # requested | acquired | released | reclaimed
    event: Mapped[str] = mapped_column(String(16), nullable=False)
    at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_scratch_lease_events_job_id", "job_id"),
        Index("ix_scratch_lease_events_at", "at"),
    )


class DeviceScan(Base):
    """One fact-extraction pass over one device's APKs, append-only.

    Exists because retention deletes the APKs: without this row, "312 APKs went in, 312 came
    out" is unprovable after the job finishes, and a run that quietly parsed 40 of 312 looks
    exactly like a device with 40 packages. `parse_failed` and `failures` are the only home
    for an APK that produced no observation, since a failure has no package name to key on.
    """

    __tablename__ = "device_scans"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    # `<driver>:<device>`, deliberately without the build: two builds of one phone are one
    # device, and `package_facts.device_count` is a count of phones, not of firmware images.
    device_key: Mapped[str] = mapped_column(String(255), nullable=False)
    build: Mapped[str] = mapped_column(String(255), nullable=False)
    scanned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    apk_total: Mapped[int] = mapped_column(nullable=False, default=0)
    parsed_ok: Mapped[int] = mapped_column(nullable=False, default=0)
    parse_failed: Mapped[int] = mapped_column(nullable=False, default=0)
    # [{"device_path": ..., "error": ...}] for every APK that produced no observation.
    failures: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )

    __table_args__ = (Index("ix_device_scans_device_key", "device_key"),)


class PackageObservation(Base):
    """One APK on one device+build, as its manifest declared it.

    One row per APK rather than per package: a device that ships the same package name twice
    (two partitions, two certificates) then produces two rows that flow through the ordinary
    cross-device conflict check instead of needing a special case, and nothing is collapsed
    before it has been compared.

    The unique key is `(device_key, build, device_path)`, which is what makes re-parsing a
    build an upsert onto the same rows — the reproducibility claim in task 4's verify line.
    """

    __tablename__ = "package_observations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    device_key: Mapped[str] = mapped_column(String(255), nullable=False)
    build: Mapped[str] = mapped_column(String(255), nullable=False)
    package: Mapped[str] = mapped_column(String(255), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    label: Mapped[str] = mapped_column(Text, nullable=False)
    label_unresolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    version_code: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    partition: Mapped[str] = mapped_column(String(64), nullable=False)
    device_path: Mapped[str] = mapped_column(Text, nullable=False)
    priv_app: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    cert_issuer: Mapped[str | None] = mapped_column(Text, nullable=True)
    cert_subject: Mapped[str | None] = mapped_column(Text, nullable=True)
    core_app: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    shared_user_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    persistent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    has_code: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    overlay_target: Mapped[str | None] = mapped_column(String(255), nullable=True)
    overlay_static: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    overlay_priority: Mapped[int | None] = mapped_column(nullable=True)
    is_input_method: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_device_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_accessibility_service: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_carrier_service: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # List-valued signals. JSONB rather than side tables: nothing queries an individual
    # protected broadcast, they travel together into the evidence bundle, and a side table per
    # signal would be six joins to rebuild one package's card.
    libraries: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    static_libraries: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    uses_libraries_required: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    uses_libraries_optional: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    protected_broadcasts: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    provider_authorities: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    intent_filters: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)

    __table_args__ = (
        UniqueConstraint(
            "device_key", "build", "device_path", name="uq_package_observations_device_path"
        ),
        Index("ix_package_observations_package", "package"),
    )


class PackageFact(Base):
    """One row per package name, merged across every device that shipped it.

    `device_count` is a stored column, not a join computed at read time: it is the primary
    triage ranking signal (design §3, task 9's queue order), and a ranking that is recomputed
    by whoever happens to read it is a ranking that will eventually be computed differently in
    two places.

    Merge rules, chosen so the merge can never lower a rule-ladder floor:

    - danger booleans (`core_app`, `persistent`, `priv_app`, `has_code`, the four service
      classes) are sticky-true — one device declaring `coreApp` is enough;
    - list-valued signals are the union across devices, deduplicated and sorted;
    - `version_code` keeps the highest seen;
    - identity scalars (`cert_issuer`, `cert_subject`, `shared_user_id`, `overlay_target`,
      `label`) take the value from the lowest `(device_key, build, device_path)`, which is
      deterministic and, unlike last-write, does not depend on scan order.

    A disagreement on any field in `factstore.CONFLICT_FIELDS` sets `has_conflict` and lands
    in `conflicts` with every value and the devices that carried it. The row survives, because
    dropping it would hide the disagreement instead of surfacing it.

    `has_conflict` is a review signal, not a verdict, and the difference is measured rather
    than assumed: across the Pixel 6 and Android 16 emulator corpora, 134 of the 147 shared
    packages disagree on the signing certificate, all of it APK signature v3 key rotation
    (`com_google_android_gms-rotation-2020`) or the AOSP test key facing Google's production
    key. So triage shows a conflict; it must not filter on one.
    """

    __tablename__ = "package_facts"

    package: Mapped[str] = mapped_column(String(255), primary_key=True)
    device_count: Mapped[int] = mapped_column(nullable=False, default=0)
    devices: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    label: Mapped[str] = mapped_column(Text, nullable=False)
    label_unresolved: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    version_code: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    partitions: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    priv_app: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    cert_issuer: Mapped[str | None] = mapped_column(Text, nullable=True)
    cert_subject: Mapped[str | None] = mapped_column(Text, nullable=True)
    core_app: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    shared_user_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    persistent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    has_code: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    overlay_target: Mapped[str | None] = mapped_column(String(255), nullable=True)
    overlay_static: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_input_method: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_device_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_accessibility_service: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_carrier_service: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    libraries: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    static_libraries: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    uses_libraries_required: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    uses_libraries_optional: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    protected_broadcasts: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    provider_authorities: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    intent_filters: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)

    has_conflict: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    conflicts: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )

    __table_args__ = (
        Index("ix_package_facts_device_count", "device_count"),
        Index("ix_package_facts_has_conflict", "has_conflict"),
    )
