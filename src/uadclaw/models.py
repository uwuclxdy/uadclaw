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
    LargeBinary,
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
    CLASSIFICATION = "classification"
    BRANCH_EMISSION = "branch_emission"


# Whether a job kind occupies the scratch lease. FIRMWARE_ANALYSIS unpacks firmware onto
# disk and needs the single-occupant lease for its whole life; classification only touches
# the DB and the network, never scratch. The requirement lives on the kind, not on the
# worker loop, so the pool stays meaningful — a classification job runs concurrently with a
# firmware job rather than queueing behind its lease.
#
# BRANCH_EMISSION writes into the OPERATOR'S git clone, which is a bind mount of its own and
# not scratch at all. Taking the lease would queue a branch behind a multi-hour Samsung
# unpack for a directory that unpack never touches.
JOB_KIND_NEEDS_SCRATCH: dict[JobKind, bool] = {
    JobKind.FIRMWARE_ANALYSIS: True,
    JobKind.CLASSIFICATION: False,
    JobKind.BRANCH_EMISSION: False,
}


class JobState(enum.StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# Terminal states a claim/reclaim sweep must never touch.
TERMINAL_STATES = frozenset({JobState.SUCCEEDED, JobState.FAILED})

# Every stage name the design's pipeline has, in design order. This is the vocabulary, NOT
# the walk: nothing may iterate it to decide what a job runs next (see JOB_KIND_STAGES).
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

# The stages each KIND actually walks. A job's `stage` records the furthest point it
# reached; resuming restarts from the stage after it, never from the beginning.
#
# Kind-scoped rather than one global walk, and this is a safety property rather than tidiness.
# The worker no-ops any stage with no registered handler, so a single global list plus a
# registered `llm` handler means **every firmware job classifies the entire corpus against a
# paid API the moment it finishes unpacking**, with nobody having asked for it. Classification
# costs money and is a separate decision, so it is a separate kind that a human queues.
#
# The consequence is deliberate and visible: FIRMWARE_ANALYSIS now ENDS at `rule_ladder`.
# It used to no-op its way through `llm`, `corroborate`, `triage` and `branch` and finish at
# `branch`, which recorded four stage runs that never did anything.
JOB_KIND_STAGES: dict[JobKind, tuple[str, ...]] = {
    JobKind.FIRMWARE_ANALYSIS: (
        "acquire",
        "unpack",
        "extract_facts",
        "corpus_graph",
        "filter",
        "rule_ladder",
    ),
    # `corroborate` APPENDS to this walk rather than forming a kind of its own: corroboration
    # has no input without a classification to corroborate, and upstream's review bar asks for
    # the check on every AI-written description rather than on a subset somebody remembered to
    # queue. It stays out of the FIRMWARE_ANALYSIS walk for the reason `llm` does — it spends
    # money, on a search quota and on the judge both.
    JobKind.CLASSIFICATION: ("llm", "corroborate"),
    # Its own kind rather than a tail on either walk, for the reason `llm` is one: emission
    # commits into somebody else's repository, so a human decides that a vendor batch is ready
    # and queues it. Appending `branch` to the classification walk would cut a branch the
    # moment a corroboration run finished, out of whatever happened to be approved at that
    # instant, with nobody having looked.
    JobKind.BRANCH_EMISSION: ("branch",),
}

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
    # This device's `/etc` rule-ladder inputs (privapp allowlists, static roles), parsed by
    # the `rule_ladder` stage out of the config XMLs the unpack stage kept. They live here
    # rather than in scratch because scratch dies with the job while the ladder is
    # corpus-wide: without this column, device B's ladder run cannot see device A's
    # allowlist. Shape: `uadclaw.etcconfig.ConfigInputs.as_json()`.
    config_inputs: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    __table_args__ = (
        Index("ix_device_scans_device_key", "device_key"),
        Index("ix_device_scans_job_id", "job_id"),
    )


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
    queries_packages: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    content_uri_authorities: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    intent_filters: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)

    # The launcher icon this device's copy of the package shipped, capped at 64 KB by
    # `icons.MAX_ICON_BYTES`. It lives on the observation and not only on the merged row
    # because the merge is RECOMPUTED from every observation: an icon written straight onto
    # `package_facts` would be erased by the next re-scan of any device that ships the package.
    icon_bytes: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    icon_mime: Mapped[str | None] = mapped_column(String(32), nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "device_key", "build", "device_path", name="uq_package_observations_device_path"
        ),
        Index("ix_package_observations_package", "package"),
        CheckConstraint(
            "(icon_bytes IS NULL) = (icon_mime IS NULL)", name="ck_package_observations_icon_pair"
        ),
    )


class PackageFact(Base):
    """One row per package name, merged across every device that shipped it.

    `device_count` is a stored column, not a join computed at read time: it is the primary
    triage ranking signal (design §3, the triage queue's order), and a ranking that is recomputed
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
    queries_packages: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    content_uri_authorities: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    intent_filters: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)

    has_conflict: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    conflicts: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )

    # What `GET /icons/{package}` serves. Merged with the same first-wins ordering the identity
    # scalars use, except that an observation carrying no icon never hides one that does — the
    # rule `label` already follows, for the same reason.
    #
    # **`has_icon` is `icon_mime is not None`**, and every screen builds it that way. Named
    # here rather than left for each lane to invent, because the screen decides whether to
    # emit an `<img>` from one column while the route serves the other: a third predicate
    # anywhere is a broken-image glyph on somebody's screen. The CHECK below is what keeps the
    # two agreeing; the route's own `icon_mime in ICON_MIMES` gate is a separate question (a
    # paired column can still name `text/html`) and does not fold into it.
    icon_bytes: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    icon_mime: Mapped[str | None] = mapped_column(String(32), nullable=True)

    __table_args__ = (
        Index("ix_package_facts_device_count", "device_count"),
        Index("ix_package_facts_has_conflict", "has_conflict"),
        # Every screen decides whether to emit an `<img src="/icons/…">` from `icon_mime`
        # while the route serves `icon_bytes`, so a row carrying one without the other renders
        # a broken-image glyph in place of the monogram that is the DESIGNED answer for "no
        # icon" — strictly worse than what it displaced. `extract_icon` returns both or
        # neither, which makes the pair consistent by convention; a partial backfill, a manual
        # fix-up or a future writer setting one column does not. This repo's rule is that an
        # "X cannot happen" is a guarantee only when the counterexample is rejected, so it is
        # rejected here rather than promised in prose.
        CheckConstraint(
            "(icon_bytes IS NULL) = (icon_mime IS NULL)", name="ck_package_facts_icon_pair"
        ),
    )


class PackageAnalysis(Base):
    """What the deterministic core (design §4-§6) concluded about one package.

    Nothing on this row is ever model output. `dependencies` and `needed_by` come from the
    corpus graph and `floor`/`floor_rule` from the rule ladder, so every column here carries
    `graph:` or `rule:` provenance by construction — which is the point: a re-run of the LLM
    stage rewrites nothing here, and a wrong edge cannot arrive from a model.

    One row per package in the corpus, not per candidate: a package already in `uad_lists.json`
    still gets its edges and its floor, because a mechanically derived `neededBy` on an
    existing entry and a floor above an existing `removal` are corrections worth proposing.
    `queued` is the only column that answers "does this reach the additions queue".

    The three stages write disjoint column sets and each is nullable until its stage has run:
    `corpus_graph` writes the edges and evidence, `filter` writes `upstream_present`/`queued`,
    `rule_ladder` writes the floor. NULL therefore means "that stage has not run for this
    package", which is a different thing from `false` and must stay distinguishable.
    """

    __tablename__ = "package_analysis"

    package: Mapped[str] = mapped_column(String(255), primary_key=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # --- corpus_graph -----------------------------------------------------------------
    # The two emitted fields, in the upstream schema's meaning. Stored rather than derived
    # from `edges` on read, for the same reason `device_count` is stored: a value that ships
    # upstream must have exactly one spelling.
    dependencies: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    needed_by: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    # Why each of the above exists: [{"kind", "dependent", "provider", "detail"}].
    edges: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    # Relations recorded but deliberately NOT emitted as edges (declared package queries,
    # required libraries no corpus package provides). Shape: `uadclaw.corpus.PackageEvidence`.
    evidence: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    # --- filter -----------------------------------------------------------------------
    upstream_present: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    queued: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    # `uadclaw.filters.FilterVerdict`: why it is or is not in the queue.
    filter_verdict: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Which bytes of `uad_lists.json` decided `upstream_present` — path, sha256, entry count
    # and when that copy was obtained. Recorded per row because the list is an external input
    # that changes what gets proposed, so "it was already upstream" has to name its evidence.
    upstream_provenance: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    # --- rule_ladder ------------------------------------------------------------------
    # The floor, never a rating: the model may raise it and never lower it.
    floor: Mapped[str | None] = mapped_column(String(16), nullable=True)
    floor_rule: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Every rule that fired, strictest first: [{"rule", "floor", "detail"}]. The triage card
    # shows this, so a reviewer can see why a package is pinned where it is.
    floor_reasons: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    # Privileged-permission allowlist membership, an INTEGRATION score and never a boot-risk
    # flag: AOSP's "device won't boot" clause fires when a package that is still present
    # requests a permission that is not allowlisted, which is a ROM-build error. Removing the
    # app removes the request. See `docs/domain-knowledge.md` § AOSP semantics.
    privapp_allowlisted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    privapp_permission_count: Mapped[int | None] = mapped_column(nullable=True)

    __table_args__ = (
        Index("ix_package_analysis_queued", "queued"),
        Index("ix_package_analysis_floor", "floor"),
    )


class PackageClassification(Base):
    """What the model proposed for one package, and what it cost.

    A separate table from `package_analysis` on purpose, sharing not one column with it.
    That table's contract is "nothing on this row is ever model output", and the cheapest way
    to keep a contract like that true is to leave the model no column to write. A re-run of
    the classification stage therefore cannot touch an edge or a floor even by mistake: the
    two live in different tables and are written by different stages.

    One row per package, keyed on the package name rather than on (package, model, run):
    triage reviews the CURRENT proposal, and a history table nothing reads is a table that
    silently grows. What is kept instead is enough to tell one proposal from another —
    `bundle_sha256` pins the exact evidence it answered, `model` and `thinking` pin what
    answered it, and `attempts` records how many calls it took.

    `usage` is the API's raw envelope, unreshaped and per row. That is deliberate: the cost
    and cache measurement the design still owes itself reads `prompt_cache_hit_tokens` and
    `prompt_cache_miss_tokens` off real production rows rather than off a throwaway script,
    and a field nobody thought to break out today is still there tomorrow.

    A parked row is a recorded terminal state, never an exception that kills the job: one
    package the model cannot answer must not cost the other 47. `parked_reason` carries the
    field and the reason the validator refused, because a park nobody can read is a park
    nobody can clear.
    """

    __tablename__ = "package_classification"

    package: Mapped[str] = mapped_column(String(255), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    # Which evidence this answers. The model output is not reproducible and the bundle is, so
    # this column is the whole reproducibility claim: same hash means same question asked.
    bundle_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    # Whether thinking mode was on. Its reasoning tokens bill as output at 20x the
    # thinking-off spend, so the cost comparison needs to know which mode produced each row.
    thinking: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # --- the proposal (NULL on a parked row: nothing was accepted) ----------------------
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Upstream's key is `list`; the attribute is not, and cannot be. A `list:` mapped_column
    # BINDS the name inside this class body, so every `Mapped[list[Any]]` annotation after it
    # resolves `list` to the column and raises `Operator 'getitem' is not supported` at import
    # time. `package_analysis` already renames upstream's `neededBy` to `needed_by`, so a
    # column name being ours rather than theirs is the established shape here.
    uad_list: Mapped[str | None] = mapped_column(String(16), nullable=True)
    removal: Mapped[str | None] = mapped_column(String(16), nullable=True)
    confidence: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Fields the model declared it could not determine. A valid, expected outcome.
    unknown_fields: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    reasoning_brief: Mapped[str | None] = mapped_column(Text, nullable=True)
    # field name -> `rule:` / `graph:` / `llm:<model>` / `human:`. Read on a re-run: a field
    # tagged `human:` survives untouched, which is what makes a bad model run re-runnable
    # without walking over an edit somebody made in triage.
    provenance: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    # --- what it cost ------------------------------------------------------------------
    usage: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    attempts: Mapped[int] = mapped_column(nullable=False, default=0)

    # --- park ---------------------------------------------------------------------------
    parked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    parked_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_package_classification_parked", "parked"),
        Index("ix_package_classification_bundle_sha256", "bundle_sha256"),
    )


class PackageTriageDecision(Base):
    """One human verdict on one proposal. **Append-only: nothing here is ever updated.**

    A status column on `package_classification` would answer "what is the state of this
    package" and destroy the only measurement this pipeline has of whether its own funnel is
    improving. Rejections in particular are the raw material for prompt work: what the model
    proposed, what a human said about it, and how often that changed, is a series rather than
    a current value, and a series cannot be recovered from a field that was overwritten.

    `bundle_sha256` is what makes the log self-expiring without anything deleting from it. A
    decision is about ONE proposal, so it is recorded against the evidence hash that proposal
    answered; re-classifying changes that hash, the old decisions stop matching the current
    one on their own, and the package returns to the queue carrying its whole history. That
    is also why there is no unique constraint on `(package, bundle_sha256)`: deciding twice on
    one proposal is a real thing a reviewer does (defer, come back, approve), and the second
    decision supersedes the first by being later rather than by replacing it.

    `action` is one of `triagestore.ACTIONS`. `reason` is required for `reject` and the
    requirement is a CHECK constraint rather than only a code path, because "a rejection
    without a reason" is exactly the row that looks fine until somebody tries to learn
    something from the log a month later.
    """

    __tablename__ = "package_triage_decision"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    package: Mapped[str] = mapped_column(String(255), nullable=False)
    # Which proposal was being decided on, not which package. See the class docstring.
    bundle_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    # What an `edit` changed: {field: {"from": ..., "to": ...}}. Kept beside the decision
    # rather than only on the classification row, because the row carries the value that won
    # and this carries the value it replaced.
    edited_fields: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        Index("ix_package_triage_decision_package_decided_at", "package", "decided_at"),
        CheckConstraint(
            "action <> 'reject' OR (reason IS NOT NULL AND btrim(reason) <> '')",
            name="ck_package_triage_decision_reject_reason",
        ),
    )


class BranchEmission(Base):
    """One emission run: a vendor batch committed onto a branch in the operator's own clone.

    The only row in this schema that describes something OUTSIDE this project, which is what
    shapes it. Nothing here can be recovered by re-reading this database — the branch lives in
    a clone this pipeline does not own, a human pushes it, and once the PR is open the only
    record of what went into it is upstream's diff. So the row is written for the question
    asked a month later: which packages, at what ratings, against which floors, from which
    base commit, disclosed with which body.

    **`commit_oid IS NULL` means the emission's outcome is not known**, and that state is a
    designed one rather than a corrupt row. The row is committed BEFORE the git commit is
    attempted, so a worker killed between the two leaves exactly this: an intent whose outcome
    is still readable off the clone. `stages.branch_stage` resolves it on the next run — see
    that function for why the write goes first and how the outcome is recovered.

    `reconciled` is how the row says which of the two ways it reached its commit oid: `false`
    is the oid `emit_branch` returned, `true` is one a later run read back off a branch a
    crashed run had already cut. They are not the same claim and a reader has to be able to
    tell them apart.

    `list_sha256` is the digest of the bytes handed to `emit_branch`, and it is the whole
    reconciliation key: a branch found on disk carrying exactly those bytes is this emission's
    branch, whatever the approved set has done since. Comparing against a re-derived batch
    instead would refuse a perfectly good branch whenever a reviewer approved one more package
    in the meantime.
    """

    __tablename__ = "branch_emission"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    # Unique: one emission per job, which is what makes the branch name derivable from the job
    # and a retry idempotent. SET NULL rather than CASCADE — the job row is operational and
    # prunable, while this row describes bytes that left the box.
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    vendor: Mapped[str] = mapped_column(String(64), nullable=False)
    branch: Mapped[str] = mapped_column(String(255), nullable=False)
    # Where the clone was and which file inside it, recorded rather than re-read from settings:
    # a later reader has to know which checkout this happened in, and the setting can move.
    repo_path: Mapped[str] = mapped_column(Text, nullable=False)
    list_path: Mapped[str] = mapped_column(Text, nullable=False)
    # The object id the branch was cut from, never the ref name. A ref moves; an oid does not,
    # and the whole emission is pinned to this one so a fetch mid-run cannot rebase it.
    base_commit: Mapped[str] = mapped_column(String(64), nullable=False)
    list_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    pipeline_version: Mapped[str] = mapped_column(String(64), nullable=False)
    pipeline_commit_sha: Mapped[str] = mapped_column(String(64), nullable=False)
    package_count: Mapped[int] = mapped_column(nullable=False)
    # The disclosure, stored rather than written to disk. The web container has a read-only
    # rootfs and a second on-disk copy is somewhere this row can drift out of agreement with.
    pr_body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    commit_oid: Mapped[str | None] = mapped_column(String(64), nullable=True)
    committed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reconciled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("job_id", name="uq_branch_emission_job_id"),
        Index("ix_branch_emission_vendor", "vendor"),
        # The pair, held in the schema for the reason the icon pair is: a row carrying a commit
        # oid with no timestamp, or a timestamp with no oid, is a half-written outcome, and the
        # recovery path reads `commit_oid IS NULL` as "the outcome is unknown". A future writer
        # setting one column would make that read answer wrongly for a row that is actually
        # finished, which is the one mistake here that ends in a second branch.
        CheckConstraint(
            "(commit_oid IS NULL) = (committed_at IS NULL)", name="ck_branch_emission_commit_pair"
        ),
        # A reconciled row is one that RECOVERED an outcome, so it must carry one. Without this
        # the flag could be set on a pending row, where it would claim a recovery that found
        # nothing.
        CheckConstraint(
            "NOT reconciled OR commit_oid IS NOT NULL", name="ck_branch_emission_reconciled_commit"
        ),
    )


class BranchEmissionPackage(Base):
    """One package that shipped in one emission, as it shipped.

    A copy of values that also live on `package_classification` and `package_analysis`, and
    that duplication is the point: those two rows are CURRENT state and get overwritten by the
    next classification run, while this one has to answer "what exactly went into that PR"
    after the corpus has moved on. `floor` in particular is the number the rating was checked
    against at emission time, which is the only way to audit the check later.
    """

    __tablename__ = "branch_emission_package"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    emission_id: Mapped[int] = mapped_column(
        ForeignKey("branch_emission.id", ondelete="CASCADE"), nullable=False
    )
    package: Mapped[str] = mapped_column(String(255), nullable=False)
    bundle_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    # Upstream's key is `list` and the attribute cannot be: a `list:` mapped_column rebinds the
    # name for the rest of the class body. Same rename `PackageClassification` carries.
    uad_list: Mapped[str] = mapped_column(String(16), nullable=False)
    removal: Mapped[str] = mapped_column(String(16), nullable=False)
    floor: Mapped[str] = mapped_column(String(16), nullable=False)

    __table_args__ = (
        UniqueConstraint("emission_id", "package", name="uq_branch_emission_package"),
        Index("ix_branch_emission_package_package", "package"),
    )


class PackageSearchResult(Base):
    """One web-search result for one package, with the page body that was fetched for it.

    **Keyed on the package NAME, never on a bundle or description hash.** The query is the
    package name and nothing else, so a re-classification after a prompt change asks the same
    search question and must re-use these rows rather than spend the search quota again. That
    is also what makes a `judge_failed` re-run free: the judge is re-asked against rows that
    are already here.

    What is kept is what makes a verdict auditable without a re-fetch: the url, the title, the
    Brave snippet, and the extracted page TEXT truncated to `PAGE_TEXT_MAX_CHARS`. Not raw
    HTML — nothing reads it back as markup, and a megabyte of hostile HTML per result is a
    liability rather than evidence.

    `page_text IS NULL` with a `fetch_error` beside it means the judge saw the snippet for
    this source instead of its body. Recorded rather than collapsed, because a verdict reached
    on snippets alone is a weaker verdict and a reviewer has to be able to see that.

    `fetched_at` is the TTL clock: a package whose newest row is inside
    `CORROBORATION_SEARCH_TTL_DAYS` is re-used with no HTTP call at all.
    """

    __tablename__ = "package_search_results"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    package: Mapped[str] = mapped_column(String(255), nullable=False)
    # Rank in the merged `web` + `discussions` order, 1-based. Stored because the merge order
    # is what the judge was shown, and it cannot be recomputed from the rows alone.
    position: Mapped[int] = mapped_column(nullable=False)
    url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    snippet: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # `web` or `discussions`. Kept because they are different source classes, and whether a
    # corroboration came off a forum thread or a vendor page is a triage-relevant fact.
    block: Mapped[str] = mapped_column(String(16), nullable=False)
    page_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Capped by `corroborate.FETCH_ERROR_MAX_CHARS` on the way in rather than by the column.
    # This quotes attacker-controlled response data — a status line, a `Content-Type` — and h11
    # caps one header near 16 KiB, so uncapped it is ~16 KiB x 10 sources x 500 packages of
    # growth per job. Held in code because `SourceEvidence.with_error` is the single funnel
    # every fetch failure passes, while a `varchar(n)` here would turn that growth into a
    # failed INSERT partway through a job that had already spent its search quota.
    fetch_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (
        UniqueConstraint("package", "url", name="uq_package_search_results_url"),
        Index("ix_package_search_results_package", "package"),
    )


class PackageCorroboration(Base):
    """Whether an independent source supports the model's description for one package.

    Upstream's stated review bar, per package — see `docs/domain-knowledge.md` § Upstream.

    A separate table from `package_classification` for the reason that one is separate from
    `package_analysis`: the corroboration stage writes only here, so "the classification row
    is what the model proposed" stays true by construction and a corroboration re-run cannot
    touch a proposal.

    `status` carries FOUR values and the difference between them is the whole point.
    `uncorroborated` (we looked, nothing supports it) and `search_failed` (we could not look)
    and `judge_failed` (we looked but could not judge) are three different facts with three
    different retry costs, and collapsing any two of them tells triage a package lacks support
    when nobody actually asked. Only the first two of the four can be a model's answer; the
    other two are set by code, and a response claiming one is refused.

    `description_sha256` is the idempotence key over exactly what was judged (the package name
    and the description, canonically serialised by `corroborate.description_digest`). A verdict
    is a function of the claim being checked, so a re-classification that changes the
    description re-opens the question and one that does not, does not.
    """

    __tablename__ = "package_corroboration"

    package: Mapped[str] = mapped_column(String(255), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    description_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    # `uadclaw.corroborate.CorroborationStatus`.
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    # NULL when no model was asked: a search that failed, or a search that returned nothing to
    # judge. A stored model id on such a row would credit a call that never happened.
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    thinking: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # The urls the judge cited as supporting the description, with the title taken off the
    # SEARCH RESULT rather than out of the model: [{"url", "title"}]. Empty for every status
    # but `corroborated`, and every entry was in the set handed to the judge — a response
    # citing anything else is rejected whole rather than having the url dropped.
    sources: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Why a `search_failed` or `judge_failed` row is one. Distinct from `reasoning`, which is
    # the judge's own note: this one is the pipeline's.
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    # field name -> `llm:<model>` for a judged verdict, `rule:corroborate` for one code
    # reached, `search:brave` for the sources.
    provenance: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )

    usage: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    attempts: Mapped[int] = mapped_column(nullable=False, default=0)

    __table_args__ = (
        Index("ix_package_corroboration_status", "status"),
        Index("ix_package_corroboration_description_sha256", "description_sha256"),
    )
