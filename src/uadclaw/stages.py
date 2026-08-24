"""The real pipeline stage handlers, wired to the worker's `StageHandler` signature.

Three job kinds run through here and they share nothing but the signature. `firmware_analysis`
walks `acquire` → `unpack` → `extract_facts` → `corpus_graph` → `filter` → `rule_ladder` and
lives on disk; `classification` walks `llm` → `corroborate`, holds no scratch lease, and is
the only kind in this repo that spends money — the model on both stages, plus a search quota
on the second; `branch_emission` walks `branch` alone and is the only one that writes outside
this project, into a git clone the operator supplies. Which stages a kind walks is
`models.JOB_KIND_STAGES`, and it is per kind precisely so a firmware job cannot wander into
`llm` — nor, now that `branch` has a real handler, into somebody else's repository.

Every firmware stage writes only inside the job's own `ctx.scratch_dir`, and none of them
writes a job row: job state is the worker's, fenced on `(job_id, worker_id, attempt)`. The
handoff between stages is a small JSON file in scratch rather than a database column, because
it describes files on disk and dies with them.

Retention is enforced as the stages go, not only at the end: `unpack` deletes the firmware
archive and every multi-GB intermediate the moment the files worth keeping are out, and
`extract_facts` deletes the APKs the moment their facts are in Postgres. A job that fails
later must not be sitting on 3.5 GB of zip it no longer needs, and decision 5 (facts-only
retention) is what makes the Samsung and Oppo scope affordable at all.
"""

import asyncio
import hashlib
import importlib.metadata
import json
import logging
import os
import shutil
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, ValidationError

from uadclaw.brave import BraveClient, BraveError, PageFetcher, require_brave_key, source_links
from uadclaw.bundle import EvidenceBundle, PackageIdentity, build_bundles
from uadclaw.classify import (
    SYSTEM_PROMPT,
    Classification,
    ClassificationJobParams,
    ClassificationRejected,
    ListDerivation,
    derive_list,
    user_prompt,
    validate_response,
)
from uadclaw.classifystore import (
    existing_bundle_hashes,
    load_identities,
    park_package,
    require_queue,
    select_candidates,
    store_classification,
)
from uadclaw.corpus import EDGE_LIBRARY, EDGE_OVERLAY, build_graph
from uadclaw.corpusstore import (
    load_config_inputs,
    record_config_inputs,
    require_corpus,
    store_filter_verdicts,
    store_floors,
    store_graph,
)
from uadclaw.corroborate import SYSTEM_PROMPT as JUDGE_SYSTEM_PROMPT
from uadclaw.corroborate import (
    Corroboration,
    CorroborationRejected,
    CorroborationStatus,
    SourceEvidence,
    code_verdict,
    validate_verdict,
)
from uadclaw.corroborate import user_prompt as judge_user_prompt
from uadclaw.corroboratestore import (
    cached_sources,
    existing_verdicts,
    record_failure,
    require_proposals,
    store_search_results,
    store_verdict,
)
from uadclaw.corroboratestore import select_candidates as select_corroboration_candidates
from uadclaw.emission import (
    BranchEmissionJobParams,
    already_carried,
    branch_name,
    insert_entries,
    render_pr_body,
)
from uadclaw.emissionstore import (
    EmissionRecord,
    forget_intent,
    load_approved,
    load_emission,
    record_commit,
    record_intent,
)
from uadclaw.etcconfig import parse_config_inputs
from uadclaw.facts import ApkFacts, ApkParseError, parse_apk
from uadclaw.factstore import record_device_scan, store_device_facts
from uadclaw.filters import FilterVerdict, queue_verdicts, survival
from uadclaw.firmware import (
    FirmwareInputError,
    FirmwareJobParams,
    FirmwareRef,
    get_driver,
    select_ref,
)
from uadclaw.ladder import RemovalFloor, compute_floors
from uadclaw.llm import (
    ChatResult,
    DeepSeekAuthError,
    DeepSeekBalanceError,
    DeepSeekError,
    DeepSeekMalformedError,
    DeepSeekUnavailableError,
    LlmBudgetError,
    LlmClient,
)
from uadclaw.models import Job
from uadclaw.settings import Settings, get_settings
from uadclaw.unpack import canonical_device_path, extract_artifacts, unpack_to_partitions
from uadclaw.upstream import load_upstream_list
from uadclaw.upstreamrepo import (
    RepoState,
    branch_exists,
    emit_branch,
    inspect_repo,
    verify_emitted_branch,
)
from uadclaw.worker import StageContext, StageHandler

logger = logging.getLogger(__name__)

FIRMWARE_DIRNAME = "firmware"
UNPACK_DIRNAME = "unpack"
ARTIFACTS_DIRNAME = "artifacts"
STATE_FILENAME = "pipeline-state.json"


class PipelineState(BaseModel):
    """What one stage tells the next about the files it left in scratch."""

    ref: FirmwareRef
    # None once `unpack` has deleted the archive it no longer needs.
    archive_path: str | None = None
    # The digest of what actually arrived, and whether anything could be checked against it.
    # False means the source published no checksum: the bytes every later stage trusts were
    # never provable, and that travels with the facts rather than being forgotten here.
    archive_sha256: str | None = None
    integrity_verified: bool = False
    partitions: list[str] = []
    artifact_count: int = 0
    apk_count: int = 0
    # Set by `extract_facts`. `apk_count` above is what came out of the images; these two are
    # what survived parsing, and a gap between them is the whole point of recording both.
    package_count: int = 0
    parse_failed_count: int = 0


class StageInputError(RuntimeError):
    """A stage's own inputs (its scratch directory, a previous stage's output) are not
    where they must be. Distinct from a firmware/unpack failure: nothing external is
    broken, the job simply cannot run from here."""


class StageRecordError(RuntimeError):
    """A per-package stage could not record what happened to one of its packages, its own
    recovery write included.

    Job-level, and raised once at the END of the stage rather than out of the failing task.
    A database nothing can be written to is a fact about the job, but the per-item boundary
    inside a `TaskGroup` is not where that gets decided: an exception raised there cancels
    every sibling, so the group loses answers already paid for in order to report a failure
    the last package could have reported on its own. Every sibling records itself first and
    the stage then fails, naming what went unrecorded.
    """


def _require_scratch(ctx: StageContext) -> Path:
    if ctx.scratch_dir is None:
        raise StageInputError(
            f"job {ctx.job_id}: this stage unpacks firmware onto disk but the job kind was "
            "claimed without a scratch lease; mark the kind as needing scratch in "
            "JOB_KIND_NEEDS_SCRATCH"
        )
    return ctx.scratch_dir


def read_state(scratch_dir: Path) -> PipelineState:
    path = scratch_dir / STATE_FILENAME
    if not path.is_file():
        raise StageInputError(
            f"read_state: {path} is missing, so the acquire stage's output is gone. A reclaimed "
            "job resumes under a NEW attempt with a fresh, empty scratch directory — requeue it "
            "from the acquire stage rather than the stage it stalled on."
        )
    return PipelineState.model_validate_json(path.read_text(encoding="utf-8"))


def write_state(scratch_dir: Path, state: PipelineState) -> None:
    scratch_dir.mkdir(parents=True, exist_ok=True)
    (scratch_dir / STATE_FILENAME).write_text(
        json.dumps(state.model_dump(mode="json"), indent=2), encoding="utf-8"
    )


async def _job_params(ctx: StageContext) -> FirmwareJobParams:
    """Read-only: a stage never writes a job row (the worker owns that, fenced)."""
    async with ctx.session_factory() as session:
        job = await session.get(Job, ctx.job_id)
        params = dict(job.params) if job is not None else None
    if params is None:
        raise StageInputError(f"_job_params: job {ctx.job_id} no longer exists")
    try:
        return FirmwareJobParams.model_validate(params)
    except ValidationError as exc:
        raise FirmwareInputError(
            f"_job_params: job {ctx.job_id} carries no usable firmware target in its params "
            f"({params!r}). Create the job with at least {{'driver': ..., 'device': ...}}: {exc}"
        ) from exc


async def acquire_stage(ctx: StageContext) -> None:
    """Resolve the job's target against its driver and download the archive into scratch."""
    settings = get_settings()
    scratch = _require_scratch(ctx)
    params = await _job_params(ctx)
    driver = get_driver(params.driver, settings)

    ref = params.as_ref()
    if ref is None:
        refs = await driver.list_available()
        ref = select_ref(refs, device=params.device, build=params.build)
    logger.info(
        "job %s acquiring %s %s build %s from %s",
        ctx.job_id,
        driver.name,
        ref.device,
        ref.build,
        ref.url,
    )
    archive = await driver.fetch(ref, scratch / FIRMWARE_DIRNAME)
    if not archive.integrity_verified:
        logger.warning(
            "job %s: %s published no checksum for %s, so the archive is integrity-unverified",
            ctx.job_id,
            driver.name,
            ref.build,
        )
    await asyncio.to_thread(
        write_state,
        scratch,
        PipelineState(
            ref=ref,
            archive_path=str(archive.path),
            archive_sha256=archive.sha256,
            integrity_verified=archive.integrity_verified,
        ),
    )


async def unpack_stage(ctx: StageContext) -> None:
    """Open the archive down to filesystem images, pull the APKs and config inputs out of
    them, then delete everything that was only ever a means to that end."""
    settings = get_settings()
    scratch = _require_scratch(ctx)
    state = await asyncio.to_thread(read_state, scratch)
    if state.archive_path is None:
        raise StageInputError(
            f"unpack_stage: job {ctx.job_id} already unpacked and released its archive; requeue "
            "it from the acquire stage to run unpack again"
        )
    archive = Path(state.archive_path)
    if not archive.resolve().is_relative_to(scratch.resolve()):
        raise StageInputError(
            f"unpack_stage: recorded archive {archive} is outside this job's scratch directory "
            f"{scratch}; refusing to read it"
        )
    if not archive.is_file():
        raise StageInputError(
            f"unpack_stage: {archive} is gone. A reclaimed job resumes under a new attempt with "
            "an empty scratch directory — requeue from the acquire stage."
        )

    workdir = scratch / UNPACK_DIRNAME
    partitions = await unpack_to_partitions(archive, workdir, settings=settings)
    artifacts = await extract_artifacts(partitions, scratch / ARTIFACTS_DIRNAME)

    # Retention, the moment the bytes stop being needed: the archive and every partition
    # image are gigabytes each, and nothing downstream reads them again.
    await asyncio.to_thread(shutil.rmtree, workdir, True)
    await asyncio.to_thread(archive.unlink, True)

    apk_count = sum(1 for artifact in artifacts if artifact.image_path.endswith(".apk"))
    await asyncio.to_thread(
        write_state,
        scratch,
        state.model_copy(
            update={
                "archive_path": None,
                "partitions": [partition.name for partition in partitions],
                "artifact_count": len(artifacts),
                "apk_count": apk_count,
            }
        ),
    )
    logger.info(
        "job %s unpacked %s %s: %d APKs plus %d config files from %d partitions",
        ctx.job_id,
        state.ref.device,
        state.ref.build,
        apk_count,
        len(artifacts) - apk_count,
        len(partitions),
    )


class TooManyApkParseFailuresError(RuntimeError):
    """This device's APKs failed to parse at a rate that makes the remaining facts
    untrustworthy. Distinct from a single `ApkParseError`: one bad APK is normal firmware,
    a fifth of them is a broken extraction wearing a smaller number."""


def apk_paths(artifacts_dir: Path) -> list[Path]:
    """Every extracted APK, deepest-first-sorted for a deterministic parse order.

    `os.walk(followlinks=False)` rather than `rglob`: not descending into a symlinked
    directory is what keeps an image's own symlink from walking the host filesystem, and on
    `rglob` that is an interpreter default rather than something this code states.
    """
    found: list[Path] = []
    for root, _dirs, files in os.walk(artifacts_dir, followlinks=False):
        for filename in files:
            if not filename.endswith(".apk"):
                continue
            path = Path(root) / filename
            if path.is_symlink() or not path.is_file():
                continue
            found.append(path)
    return sorted(found)


def artifact_location(artifacts_dir: Path, path: Path) -> tuple[str, str]:
    """`(partition, device_path)` for an extracted file, from the layout `extract_artifacts`
    wrote: `<artifacts>/<partition>/<path-inside-the-image>`. The canonical device path is
    rebuilt with `unpack.canonical_device_path`, the same function the extractor used, so the
    nested-`system/` root difference between partitions cannot come back as two spellings."""
    relative = path.relative_to(artifacts_dir).as_posix()
    partition, _, image_path = relative.partition("/")
    if not image_path:
        raise StageInputError(
            f"artifact_location: {relative} sits directly in the artifacts directory with no "
            "partition directory above it; extract_artifacts always writes "
            "<partition>/<path-inside-the-image>"
        )
    return partition, canonical_device_path(partition, image_path)


def parse_device_apks(
    artifacts_dir: Path, paths: list[Path]
) -> tuple[list[ApkFacts], list[dict[str, str]]]:
    """Parse every APK, keeping the failures instead of raising on the first one.

    One unreadable APK out of a vendor image is a normal event and must not cost the other
    three hundred; the caller applies the failure budget that decides whether the rate itself
    is the failure. Blocking and CPU-bound — run through `asyncio.to_thread`.
    """
    parsed: list[ApkFacts] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        partition, device_path = artifact_location(artifacts_dir, path)
        try:
            parsed.append(parse_apk(path, partition=partition, device_path=device_path))
        except ApkParseError as exc:
            # Full traceback before it is reduced to a row, never swallowed.
            logger.exception("APK %s could not be parsed", device_path)
            failures.append({"device_path": device_path, "error": str(exc)})
    return parsed, failures


async def extract_facts_stage(ctx: StageContext) -> None:
    """Parse every extracted APK into facts, merge them into the corpus by package name, and
    delete the APKs.

    The config XMLs extracted alongside them stay: they are `corpus_graph`/`rule_ladder`
    inputs (privapp allowlists, sysconfig, roles) measured in kilobytes, and they are not what
    facts-only retention is about — the gigabytes are the APKs, and those go here.
    """
    settings = get_settings()
    scratch = _require_scratch(ctx)
    state = await asyncio.to_thread(read_state, scratch)
    artifacts_dir = scratch / ARTIFACTS_DIRNAME
    if not artifacts_dir.is_dir():
        raise StageInputError(
            f"extract_facts_stage: job {ctx.job_id} has no {artifacts_dir} directory, so the "
            "unpack stage's output is gone. A reclaimed job resumes under a NEW attempt with "
            "an empty scratch directory — requeue it from the acquire stage."
        )

    paths = await asyncio.to_thread(apk_paths, artifacts_dir)
    if not paths:
        raise StageInputError(
            f"extract_facts_stage: {artifacts_dir} holds no APK. unpack refuses to finish "
            "without one, so this scratch directory belongs to a different attempt or was "
            "emptied underneath the job; requeue from the acquire stage."
        )

    parsed, failures = await asyncio.to_thread(parse_device_apks, artifacts_dir, paths)
    _check_parse_budget(len(paths), failures, settings.max_apk_parse_failure_ratio)

    device_key = f"{state.ref.driver}:{state.ref.device}"
    observed_at = datetime.now(UTC)
    async with ctx.session_factory() as session, session.begin():
        await record_device_scan(
            session,
            job_id=ctx.job_id,
            device_key=device_key,
            build=state.ref.build,
            scanned_at=observed_at,
            apk_total=len(paths),
            parsed_ok=len(parsed),
            failures=failures,
        )
        packages = await store_device_facts(
            session,
            device_key=device_key,
            build=state.ref.build,
            facts=parsed,
            observed_at=observed_at,
        )

    # Retention, decision 5: the facts are in Postgres, so the APKs stop existing here rather
    # than at the end of the job. A later stage failing must not leave gigabytes of firmware
    # on a scratch volume sized for one device.
    await asyncio.to_thread(_delete_apks, paths)
    await asyncio.to_thread(
        write_state,
        scratch,
        state.model_copy(
            update={"package_count": len(packages), "parse_failed_count": len(failures)}
        ),
    )
    logger.info(
        "job %s extracted facts for %s %s: %d package(s) from %d APK(s), %d unparseable",
        ctx.job_id,
        state.ref.device,
        state.ref.build,
        len(packages),
        len(paths),
        len(failures),
    )


def _check_parse_budget(total: int, failures: list[dict[str, str]], max_ratio: float) -> None:
    if not failures:
        return
    ratio = len(failures) / total
    if ratio <= max_ratio:
        logger.warning(
            "%d of %d APK(s) failed to parse (%.1f%%, within the %.1f%% budget): %s",
            len(failures),
            total,
            ratio * 100,
            max_ratio * 100,
            ", ".join(failure["device_path"] for failure in failures[:10]),
        )
        return
    raise TooManyApkParseFailuresError(
        f"extract_facts: {len(failures)} of {total} APKs failed to parse ({ratio:.1%}), over "
        f"the {max_ratio:.1%} budget, so the facts this device would record are not a "
        "description of it. First failure: "
        f"{failures[0]['device_path']}: {failures[0]['error']}. Re-run the unpack stage if the "
        "extraction looks wrong, or raise MAX_APK_PARSE_FAILURE_RATIO if this vendor genuinely "
        "ships that many unreadable APKs."
    )


def _delete_apks(paths: list[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


async def corpus_graph_stage(ctx: StageContext) -> None:
    """Derive the two high-confidence dependency edge classes over the whole corpus.

    Corpus-wide rather than per device, and re-run by every job: an edge is a statement about
    what else exists, so a second device arriving can create edges for packages a previous job
    already analysed.

    This is also where the `/etc` config XMLs are read and parked on the device's scan row —
    the first stage of the deterministic core that needs them, and the last one that can still
    see the job's scratch directory. The graph needs the platform-declared shared libraries to
    tell "no package provides this library" from "the platform does", and the rule ladder
    needs the privapp allowlists and static roles two stages later, by which point this job's
    scratch is the only copy in existence.
    """
    scratch = _require_scratch(ctx)
    artifacts_dir = scratch / ARTIFACTS_DIRNAME
    if not artifacts_dir.is_dir():
        raise StageInputError(
            f"corpus_graph_stage: job {ctx.job_id} has no {artifacts_dir} directory, so the "
            "config XMLs the deterministic core reads are gone. A reclaimed job resumes under "
            "a NEW attempt with an empty scratch directory — requeue it from the acquire stage."
        )
    device_config = await asyncio.to_thread(parse_config_inputs, artifacts_dir)
    if device_config.files_failed:
        logger.warning(
            "job %s: %d config XML(s) did not parse and contribute no rule input: %s",
            ctx.job_id,
            len(device_config.files_failed),
            ", ".join(device_config.files_failed[:10]),
        )

    at = datetime.now(UTC)
    async with ctx.session_factory() as session, session.begin():
        await record_config_inputs(session, job_id=ctx.job_id, config=device_config)
        corpus = await require_corpus(session)
        config = await load_config_inputs(session)
        graph = build_graph(corpus, platform_libraries=config.platform_libraries)
        written = await store_graph(session, corpus, graph, at=at)
    counts = graph.counts_by_kind()
    logger.info(
        "job %s corpus graph: %d package(s), %d edge(s) (%d overlay, %d library) from %d "
        "config file(s)",
        ctx.job_id,
        written,
        len(graph.edges),
        counts.get(EDGE_OVERLAY, 0),
        counts.get(EDGE_LIBRARY, 0),
        device_config.files_read,
    )


async def filter_stage(ctx: StageContext) -> None:
    """Decide which packages reach the additions queue.

    The upstream list is loaded first and on its own: if it is missing or empty the stage
    fails here, before anything is written, rather than marking all 393 packages as new.
    """
    settings = get_settings()
    upstream = await asyncio.to_thread(load_upstream_list, settings.upstream_list_path)
    at = datetime.now(UTC)
    async with ctx.session_factory() as session, session.begin():
        corpus = await require_corpus(session)
        verdicts = queue_verdicts(corpus, upstream=upstream)
        await store_filter_verdicts(session, verdicts, upstream=upstream, at=at)
    counts = survival(verdicts)
    logger.info(
        "job %s filter: %d of %d package(s) queued (%d already upstream, %d auto-generated "
        "RRO, %d emulator-only) against %s sha256 %s",
        ctx.job_id,
        counts[str(FilterVerdict.QUEUED)],
        counts["total"],
        counts[str(FilterVerdict.ALREADY_UPSTREAM)],
        counts[str(FilterVerdict.AUTO_GENERATED_RRO)],
        counts[str(FilterVerdict.EMULATOR_ONLY)],
        upstream.path,
        upstream.sha256[:12],
    )


async def rule_ladder_stage(ctx: StageContext) -> None:
    """Compute every package's removal floor.

    Database to database: the `/etc` inputs it needs were parked on the device scan rows by
    `corpus_graph`, and the union across every device is what makes a floor a property of the
    package rather than of whichever device is being scanned right now.
    """
    at = datetime.now(UTC)
    async with ctx.session_factory() as session, session.begin():
        corpus = await require_corpus(session)
        config = await load_config_inputs(session)
        floors = compute_floors(corpus, config=config)
        written = await store_floors(session, floors, at=at)

    by_floor: dict[str, int] = {}
    for floor in floors.values():
        by_floor[str(floor.floor)] = by_floor.get(str(floor.floor), 0) + 1
    logger.info(
        "job %s rule ladder: %d floor(s) over %d privapp allowlist entries and %d static "
        "role(s); %s",
        ctx.job_id,
        written,
        len(config.privapp_permissions),
        len(config.static_role_holders),
        ", ".join(f"{tier} {count}" for tier, count in sorted(by_floor.items())),
    )


async def _classification_params(ctx: StageContext) -> ClassificationJobParams:
    async with ctx.session_factory() as session:
        job = await session.get(Job, ctx.job_id)
        params = dict(job.params) if job is not None else None
    if params is None:
        raise StageInputError(f"_classification_params: job {ctx.job_id} no longer exists")
    try:
        return ClassificationJobParams.model_validate(params)
    except ValidationError as exc:
        raise StageInputError(
            f"_classification_params: job {ctx.job_id} carries params a classification job "
            f"cannot run from ({params!r}): {exc}"
        ) from exc


def _require_provider(
    settings: Settings, params: ClassificationJobParams, *, job_id: uuid.UUID
) -> str:
    """The provider this job's model calls go to, or a fail-fast refusal naming the job.

    Creation already refused an unknown id, so reaching this with one means the provider
    table changed between creation and run — or the row was written by hand, which the stage
    guard exists for anyway. Either way the refusal names the job, the bad id and the valid
    ids, before a single request is made.
    """
    provider_id = params.provider if params.provider is not None else settings.llm_default_provider
    if provider_id not in settings.llm_providers:
        valid = ", ".join(sorted(settings.llm_providers)) or "(none configured)"
        raise StageInputError(
            f"job {job_id}: provider {provider_id!r} is not in the provider table "
            f"(configured ids: {valid}). Add it to LLM_PROVIDERS (or mount "
            "secrets/llm_providers), or re-create the job with a provider param naming one "
            "of the configured ids."
        )
    return provider_id


async def _classify_one(
    client: LlmClient,
    bundle: EvidenceBundle,
    *,
    floor: RemovalFloor,
    derivation: ListDerivation,
    max_calls: int,
    spent: list[int],
) -> tuple[Classification | None, str | None, ChatResult | None, int]:
    """One package: call, validate, re-prompt while the budget lasts, then give up.

    Returns `(classification, park_reason, last_result, calls)`, where `calls` is the number
    of requests that actually reached the wire. Giving up returns a reason rather than
    raising, because one package the model cannot answer must not cost the other 47 — the
    caller records a park row and moves on.

    **One budget, counted in requests, and it is the only ceiling.** A package can fail two
    ways: the WIRE (429/500/503, a transport error, a body that is not JSON, the documented
    empty-content bug), retried inside the client and invisible here; and the ANSWER (a
    `removal` under the floor, a four-character description), retried here by re-prompting.
    Giving each its own cap multiplies them — a `503, 503, below-floor` cycle spends three
    requests on one re-prompt, so two caps of three is a real ceiling of nine. So this loop
    spends `max_calls` requests total, tells the client how many of them it may use per call,
    and charges itself what each call actually cost, on the failure path too.

    An exhausted CLIENT is terminal for this package rather than a re-prompt: the client has
    already retried the wire and the budget is spent, and re-entering the loop would send the
    identical prompt into a transport that just refused it three times.

    A `LlmBudgetError`, a bad credential and an empty balance still abort the whole job:
    none of them is about this package, and burning 48 packages' budgets against a dead
    account is not a diagnosis. Every OTHER failure is contained by `_classify_and_store`,
    which reads `spent` — a one-element list rather than the return value, because the count
    has to survive the exception that ends this call — to charge the parked row what the
    package really cost.
    """
    user = user_prompt(bundle)
    last_result: ChatResult | None = None
    reason = "no attempt was made"
    calls = 0
    while calls < max_calls:
        try:
            result = await client.complete_json(
                system=SYSTEM_PROMPT, user=user, max_calls=max_calls - calls
            )
        except (DeepSeekMalformedError, DeepSeekUnavailableError) as exc:
            # Terminal, and charged for what it really spent — `exc.attempts`, not one.
            calls += exc.attempts
            spent[0] = calls
            reason = f"after {calls} request(s): {exc}"
            logger.warning(
                "classification gave up on the wire: package=%s %s", bundle.package, reason
            )
            break
        calls += result.attempts
        spent[0] = calls
        last_result = result
        try:
            payload = result.json_object()
            classification = validate_response(
                payload, bundle=bundle, floor=floor, derivation=derivation, model=result.model
            )
        except (ClassificationRejected, DeepSeekMalformedError) as exc:
            reason = f"after {calls} request(s): {exc}"
            logger.warning("classification rejected: package=%s %s", bundle.package, reason)
            continue
        return classification, None, result, calls
    return None, reason, last_result, calls


async def llm_stage(ctx: StageContext) -> None:
    """Classify every candidate the filter queued, one validated proposal at a time.

    Database and network only — no scratch, which is why `classification` is registered with
    `JOB_KIND_NEEDS_SCRATCH[CLASSIFICATION] = False` and runs alongside a firmware job rather
    than queueing behind its single-occupant lease.

    The deterministic view is RECOMPUTED here rather than read back out of `package_analysis`.
    That is deliberate on two counts: `ladder.compute_floors` is the only constructor of a
    `RemovalFloor` and reviving one from JSONB would be a second one, weakening the type that
    makes a below-floor value unrepresentable; and a floor read from a row could be stale
    against a corpus that grew since, so the bundle would attest evidence that no longer
    holds.
    """
    settings = get_settings()
    params = await _classification_params(ctx)
    provider_id = _require_provider(settings, params, job_id=ctx.job_id)
    provider = settings.llm_providers[provider_id]
    limit = min(
        params.limit or settings.classification_max_packages, settings.classification_max_packages
    )
    upstream = await asyncio.to_thread(load_upstream_list, settings.upstream_list_path)

    async with ctx.session_factory() as session:
        corpus = await require_corpus(session)
        config = await load_config_inputs(session)
        identities = await load_identities(session)
        queue = await require_queue(session)
        existing = await existing_bundle_hashes(session)

    graph = build_graph(corpus, platform_libraries=config.platform_libraries)
    floors = compute_floors(corpus, config=config)
    bundles = build_bundles(
        corpus, floors=floors, identities=identities, graph=graph, upstream=upstream
    )
    candidates = select_candidates(
        queued=queue,
        bundles=bundles,
        existing=existing,
        packages=params.packages,
        reclassify=params.reclassify,
        limit=limit,
    )
    if not candidates:
        logger.info(
            "job %s llm: nothing to classify — %d queued package(s), all already answered "
            "against their current evidence bundle. Pass reclassify=true to ask again.",
            ctx.job_id,
            len(queue),
        )
        return

    logger.info(
        "job %s llm: classifying %d of %d queued package(s) with provider %s model %s "
        "(thinking=%s)",
        ctx.job_id,
        len(candidates),
        len(queue),
        provider_id,
        provider.model,
        provider.thinking,
    )
    counts = {"classified": 0, "parked": 0}
    # Packages whose own recovery write failed. Collected rather than raised where it happens:
    # see `_classify_and_store` and `StageRecordError`.
    unrecorded: list[str] = []
    # A TaskGroup rather than `asyncio.gather`: the failures that reach here at all are the
    # ones that abort the whole job (an empty balance, a rejected key, a budget too small for
    # any package), and `gather` propagates the first one while leaving every sibling running
    # — so 47 more packages would keep spending their retry cap against a dead account, and
    # would then be writing rows through a session and an HTTP client this block has already
    # closed. A TaskGroup cancels the siblings and waits for them before it raises.
    async with (
        LlmClient.from_settings(settings, provider_id=provider_id) as client,
        asyncio.TaskGroup() as group,
    ):
        for package in candidates:
            group.create_task(
                _classify_and_store(
                    ctx,
                    client,
                    bundles[package],
                    floor=floors[package],
                    identity=identities.get(package),
                    max_calls=settings.classification_max_calls_per_package,
                    counts=counts,
                    unrecorded=unrecorded,
                )
            )
    logger.info(
        "job %s llm: %d classified, %d parked, %d unrecorded out of %d candidate(s)",
        ctx.job_id,
        counts["classified"],
        counts["parked"],
        len(unrecorded),
        len(candidates),
    )
    if unrecorded:
        raise StageRecordError(
            f"job {ctx.job_id} llm: {len(unrecorded)} of {len(candidates)} package(s) were "
            f"paid for and could not be written, their own recovery row included: "
            f"{', '.join(sorted(unrecorded))}. The other packages in this run were written "
            "and are not re-asked. Check the database is reachable and writable, then re-run."
        )


async def _classify_and_store(
    ctx: StageContext,
    client: LlmClient,
    bundle: EvidenceBundle,
    *,
    floor: RemovalFloor,
    identity: PackageIdentity | None,
    max_calls: int,
    counts: dict[str, int],
    unrecorded: list[str],
) -> None:
    """One package end to end, in its own transaction. **Raises only for a job-level abort.**

    Per package rather than one transaction for the batch: a database error on package 30
    must not discard 29 answers that were already paid for.

    This runs inside a `TaskGroup`, so anything that escapes here cancels every sibling. It
    used to: a 400, a 404 or a proxy's 502 is a bare `DeepSeekError`, which is neither of the
    two classes `_classify_one` catches and which the client does not retry. Measured on a
    four-package group with one 502 in it: four requests reached the wire, all four were paid
    for, and `package_classification` came out EMPTY — three answers lost to a sibling and no
    park row for the fourth, so every future run asked all four again. So every failure that
    is ABOUT THIS PACKAGE, the wire and the write alike, becomes this package's own parked
    row naming what happened.

    Three exceptions still take the whole job down, unchanged and deliberately, exactly as
    `_corroborate_and_store` and this stage's own docstring have it: `DeepSeekAuth`,
    `DeepSeekBalance` and `LlmBudget` are facts about the ACCOUNT or the configuration
    rather than about this package, and burning 47 more packages' budgets against a dead
    account is not a diagnosis.

    **A failure of the park write itself is still job-level, and still does not raise here.**
    It used to, and that was the same defect one level down: a stale triage edit made
    `park_package` refuse, so the recovery call written to contain a per-package failure
    became the thing that cancelled every sibling. It is counted instead, and `llm_stage`
    raises `StageRecordError` once the group has finished and every other package has its row.
    """
    # One element rather than a return value, because the count has to survive the exception
    # that ends the call it is charged in.
    spent = [0]
    try:
        await _classify_and_store_one(
            ctx,
            client,
            bundle,
            floor=floor,
            identity=identity,
            max_calls=max_calls,
            counts=counts,
            spent=spent,
        )
    except (DeepSeekAuthError, DeepSeekBalanceError, LlmBudgetError):
        raise
    except Exception as exc:
        logger.exception("classification failed for %s", bundle.package)
        attempts = spent[0] + (exc.attempts if isinstance(exc, DeepSeekError) else 0)
        try:
            async with ctx.session_factory() as session, session.begin():
                await park_package(
                    session,
                    bundle.package,
                    bundle_sha256=bundle.sha256,
                    model=client.model,
                    thinking=client.thinking,
                    reason=f"after {attempts} request(s): {type(exc).__name__}: {exc}",
                    usage={},
                    attempts=attempts,
                    at=datetime.now(UTC),
                )
        except Exception:
            logger.exception("recording the failure of %s failed too", bundle.package)
            unrecorded.append(bundle.package)
            return
        counts["parked"] += 1


async def _classify_and_store_one(
    ctx: StageContext,
    client: LlmClient,
    bundle: EvidenceBundle,
    *,
    floor: RemovalFloor,
    identity: PackageIdentity | None,
    max_calls: int,
    counts: dict[str, int],
    spent: list[int],
) -> None:
    """The work `_classify_and_store` wraps. Every failure it does not turn into a park
    reason of its own is contained by that wrapper, which reads `spent` for the true cost."""
    item = bundle.payload.get("facts", {})
    derivation = derive_list(
        bundle.package,
        cert_issuer=identity.cert_issuer if identity else None,
        partitions=tuple(item.get("partitions", ())),
    )
    classification, reason, result, attempts = await _classify_one(
        client, bundle, floor=floor, derivation=derivation, max_calls=max_calls, spent=spent
    )
    at = datetime.now(UTC)
    usage = result.usage if result is not None else {}
    async with ctx.session_factory() as session, session.begin():
        if classification is None:
            await park_package(
                session,
                bundle.package,
                bundle_sha256=bundle.sha256,
                model=client.model,
                thinking=client.thinking,
                reason=reason or "the model produced nothing usable",
                usage=usage,
                attempts=attempts,
                at=at,
            )
        else:
            await store_classification(
                session,
                classification,
                model=result.model if result is not None else client.model,
                thinking=client.thinking,
                usage=usage,
                attempts=attempts,
                at=at,
            )
    # Counted after the transaction commits, never inside it: a write that fails at commit
    # would otherwise be counted as an answer here and as a park in the wrapper, and the two
    # numbers the stage logs would add up to more packages than it had.
    counts["parked" if classification is None else "classified"] += 1


class _QueryBudget:
    """How many Brave queries this job has left.

    A plain counter rather than a semaphore: the ceiling is a SPEND, not a concurrency limit,
    and it must be decremented once per query taken rather than released afterwards. Safe
    across the TaskGroup below because there is no await between the check and the decrement,
    so no other task can observe the intermediate state.
    """

    __slots__ = ("ceiling", "remaining")

    def __init__(self, ceiling: int) -> None:
        self.ceiling = ceiling
        self.remaining = ceiling

    def take(self) -> bool:
        if self.remaining < 1:
            return False
        self.remaining -= 1
        return True


async def _judge_one(
    client: LlmClient,
    *,
    package: str,
    description: str,
    sources: Sequence[SourceEvidence],
    max_calls: int,
) -> tuple[Corroboration | None, str | None, ChatResult | None, int]:
    """One package's verdict: call, validate, re-prompt while the budget lasts, then give up.

    The same single-budget shape as `_classify_one`, counted in REQUESTS rather than in
    re-prompts, because the wire retries live inside the client and giving each layer its own
    cap turns a documented 3 into a real 9. Returns `(corroboration, reason, last_result,
    calls)`; giving up returns a reason so the caller records `judge_failed` with it, rather
    than raising and costing every other package its verdict.

    A rejected verdict is re-prompted rather than repaired. That matters most for the
    fabricated-citation case: the answer cited a source it was never given, so dropping the bad
    url and keeping the rest would keep whatever reasoning produced it.
    """
    user = judge_user_prompt(package, description, sources)
    last_result: ChatResult | None = None
    reason = "no attempt was made"
    calls = 0
    while calls < max_calls:
        try:
            result = await client.complete_json(
                system=JUDGE_SYSTEM_PROMPT, user=user, max_calls=max_calls - calls
            )
        except (DeepSeekMalformedError, DeepSeekUnavailableError) as exc:
            # Terminal, and charged for what it really spent — `exc.attempts`, not one.
            calls += exc.attempts
            reason = f"after {calls} request(s): {exc}"
            logger.warning("corroboration gave up on the wire: package=%s %s", package, reason)
            break
        calls += result.attempts
        last_result = result
        try:
            payload = result.json_object()
            corroboration = validate_verdict(
                payload,
                package=package,
                description=description,
                sources=sources,
                model=result.model,
            )
        except (CorroborationRejected, DeepSeekMalformedError) as exc:
            reason = f"after {calls} request(s): {exc}"
            logger.warning("corroboration verdict rejected: package=%s %s", package, reason)
            continue
        return corroboration, None, result, calls
    return None, reason, last_result, calls


async def _search_sources(
    ctx: StageContext,
    *,
    brave: BraveClient,
    fetcher: PageFetcher,
    package: str,
    settings: Settings,
    budget: _QueryBudget,
) -> tuple[list[SourceEvidence] | None, str]:
    """This package's evidence, from the cache when it is fresh and from Brave otherwise.

    Returns `(sources, "")`, or `(None, why the search failed)` — a failure is a value here
    rather than an exception, because a search error must never fail the job: the stage is
    per-package resumable and one flaky request must not discard the run's completed work.

    The search rows are committed BEFORE the judge is asked anything. That ordering is what
    makes `judge_failed` free to retry: a re-run finds them inside the TTL and spends no
    search quota at all.

    **A query that was PAID FOR reaches the database whatever the fetch phase then does.** The
    rows used to be written only after `fetch_all` returned, so a package that died in the
    fetch phase cached nothing and recorded no status — and then re-poisoned itself and burned
    a fresh Brave query on every future run, permanently. `fetch_all` no longer raises, and
    this is the second half of that guarantee: if it ever does again, the hits are still stored
    with the failure recorded per source, and the retry costs zero quota.
    """
    fresh_after = datetime.now(UTC) - timedelta(days=settings.corroboration_search_ttl_days)
    async with ctx.session_factory() as session:
        cached = await cached_sources(session, package, fresh_after=fresh_after)
    if cached:
        logger.debug("corroboration re-used %d cached source(s) for %s", len(cached), package)
        return cached, ""
    if not budget.take():
        return None, (
            f"this job's Brave query budget of {budget.ceiling} is spent, so the search was "
            "never made. Re-run the job to search the remaining packages, or raise "
            "CORROBORATION_MAX_QUERIES_PER_JOB."
        )
    try:
        hits = await brave.search(package, limit=settings.corroboration_sources_per_package)
    except BraveError as exc:
        # Full traceback before it is reduced to a row, never swallowed.
        logger.exception("corroboration search failed for %s", package)
        return None, f"{type(exc).__name__}: {exc}"
    try:
        sources = await fetcher.fetch_all(hits)
    except Exception as exc:
        logger.exception("corroboration fetch phase failed for %s", package)
        detail = f"the fetch phase failed: {type(exc).__name__}: {exc}"
        sources = [hit.with_error(detail) for hit in hits]
    async with ctx.session_factory() as session, session.begin():
        await store_search_results(session, package, sources, at=datetime.now(UTC))
    return sources, ""


async def _corroborate_and_store(
    ctx: StageContext,
    *,
    brave: BraveClient,
    fetcher: PageFetcher,
    judge: LlmClient,
    package: str,
    description: str,
    settings: Settings,
    budget: _QueryBudget,
    counts: dict[str, int],
    unrecorded: list[str],
) -> None:
    """One package end to end, in its own transactions. **Raises only for a job-level abort.**

    Per package rather than one transaction for the batch, for `_classify_and_store`'s reason:
    a database error on package 30 must not discard 29 verdicts that were already paid for.

    This runs inside a `TaskGroup`, so anything that escapes here cancels every sibling — and a
    measured run lost all six packages in a group, judge calls already paid for included, to a
    single unhandled exception from one package's fetch phase. So every failure that is ABOUT
    THIS PACKAGE becomes this package's own recorded status, at the phase it reached: before
    the sources are in hand it is `search_failed` (a retry re-searches and spends a query),
    after them it is `judge_failed` (a retry re-reads the cached rows and spends none).

    Three exceptions still take the whole job down, unchanged and deliberately: `DeepSeekAuth`,
    `DeepSeekBalance` and `LlmBudget` are facts about the ACCOUNT rather than about this
    package, exactly as `llm_stage` documents, and letting 499 more packages burn their budgets
    against a dead account is not a diagnosis.

    **A failure of the recovery write itself is still job-level and still does not raise here**,
    for `_classify_and_store`'s reason: the one call written to contain a per-package failure
    must not be the one that cancels the group. It is collected and `corroborate_stage` raises
    `StageRecordError` after every other package has recorded itself.
    """
    # One element rather than a return value, because the phase has to survive the exception
    # that ends the call.
    phase = [CorroborationStatus.SEARCH_FAILED]
    try:
        await _corroborate_one(
            ctx,
            brave=brave,
            fetcher=fetcher,
            judge=judge,
            package=package,
            description=description,
            settings=settings,
            budget=budget,
            counts=counts,
            phase=phase,
        )
    except (DeepSeekAuthError, DeepSeekBalanceError, LlmBudgetError):
        raise
    except Exception as exc:
        logger.exception("corroboration failed for %s", package)
        try:
            async with ctx.session_factory() as session, session.begin():
                await record_failure(
                    session,
                    package,
                    description=description,
                    status=phase[0],
                    reason=f"{type(exc).__name__}: {exc}",
                    at=datetime.now(UTC),
                )
        except Exception:
            logger.exception("recording the failure of %s failed too", package)
            unrecorded.append(package)
            return
        counts[str(phase[0])] += 1


async def _corroborate_one(
    ctx: StageContext,
    *,
    brave: BraveClient,
    fetcher: PageFetcher,
    judge: LlmClient,
    package: str,
    description: str,
    settings: Settings,
    budget: _QueryBudget,
    counts: dict[str, int],
    phase: list[CorroborationStatus],
) -> None:
    """The work `_corroborate_and_store` wraps. Every failure it does not turn into a value is
    contained by that wrapper, which reads `phase` to decide what this package is recorded as."""
    sources, failure = await _search_sources(
        ctx, brave=brave, fetcher=fetcher, package=package, settings=settings, budget=budget
    )
    at = datetime.now(UTC)
    if sources is None:
        async with ctx.session_factory() as session, session.begin():
            await record_failure(
                session,
                package,
                description=description,
                status=CorroborationStatus.SEARCH_FAILED,
                reason=failure,
                at=at,
            )
        counts[str(CorroborationStatus.SEARCH_FAILED)] += 1
        return

    # The sources are in hand and stored, so a failure past this line is free to retry: the
    # cached rows answer the same query for `corroboration_search_ttl_days` at zero quota.
    phase[0] = CorroborationStatus.JUDGE_FAILED

    if not sources:
        # Nothing to judge, so nothing is asked. Not a failure and not a model call: with zero
        # sources, "no independent source supports this" is arithmetic rather than a
        # judgement, and the row says so by carrying `rule:` provenance.
        verdict = code_verdict(
            package,
            description,
            status=CorroborationStatus.UNCORROBORATED,
            reasoning="the search returned no result to judge",
        )
        async with ctx.session_factory() as session, session.begin():
            await store_verdict(
                session,
                verdict,
                model=None,
                thinking=None,
                sources=[],
                usage={},
                attempts=0,
                at=at,
            )
        counts[str(CorroborationStatus.UNCORROBORATED)] += 1
        return

    corroboration, reason, result, calls = await _judge_one(
        judge,
        package=package,
        description=description,
        sources=sources,
        max_calls=settings.corroboration_max_calls_per_package,
    )
    at = datetime.now(UTC)
    usage = result.usage if result is not None else {}
    async with ctx.session_factory() as session, session.begin():
        if corroboration is None:
            await record_failure(
                session,
                package,
                description=description,
                status=CorroborationStatus.JUDGE_FAILED,
                reason=reason or "the judge produced nothing usable",
                model=judge.model,
                thinking=judge.thinking,
                usage=usage,
                attempts=calls,
                at=at,
            )
            counts[str(CorroborationStatus.JUDGE_FAILED)] += 1
            return
        await store_verdict(
            session,
            corroboration,
            model=result.model if result is not None else judge.model,
            thinking=judge.thinking,
            sources=source_links(sources, corroboration.sources),
            usage=usage,
            attempts=calls,
            at=at,
        )
        counts[str(corroboration.status)] += 1


async def corroborate_stage(ctx: StageContext) -> None:
    """Check every live proposal against an independent source, one package at a time.

    Upstream's stated review bar, implemented: search the package name, fetch the top results'
    bodies, and have a judge decide whether any of them supports the description the model
    wrote. Database and network only, like `llm` — no scratch.

    Appended to the CLASSIFICATION walk rather than made its own kind: corroboration has no
    input without a classification to corroborate. The cost of that is stated rather than
    hidden — finishing a classification job now needs a Brave key, and a box without one fails
    here with `require_brave_key` naming the setting instead of failing silently.
    """
    settings = get_settings()
    params = await _classification_params(ctx)
    # The same provider the job's `llm` stage used: the judge is a model call of the SAME
    # job, and the job's params are what picked the provider. Resolved here too, fail-fast,
    # before a single query or call.
    provider_id = _require_provider(settings, params, job_id=ctx.job_id)
    limit = min(
        params.limit or settings.corroboration_max_packages, settings.corroboration_max_packages
    )
    # Before a single query or call: the failure names the setting rather than arriving as an
    # auth error partway through a job that has already spent the search quota.
    require_brave_key(settings)

    async with ctx.session_factory() as session:
        proposals = await require_proposals(session)
        existing = await existing_verdicts(session)
    candidates = select_corroboration_candidates(
        proposals=proposals, existing=existing, packages=params.packages, limit=limit
    )
    if not candidates:
        logger.info(
            "job %s corroborate: nothing to check — %d live proposal(s), all already judged "
            "against their current description.",
            ctx.job_id,
            len(proposals),
        )
        return

    logger.info(
        "job %s corroborate: checking %d of %d live proposal(s) against %s",
        ctx.job_id,
        len(candidates),
        len(proposals),
        settings.brave_search_url,
    )
    budget = _QueryBudget(settings.corroboration_max_queries_per_job)
    counts = {str(status): 0 for status in CorroborationStatus}
    # See `_corroborate_and_store` and `StageRecordError`: a recovery write that fails is the
    # job's problem, not the group's.
    unrecorded: list[str] = []
    # A TaskGroup for `llm_stage`'s reason: the failures that reach this far are the ones that
    # abort the whole job (an empty balance, a rejected key), and `gather` would propagate the
    # first while leaving every sibling running against a dead account and writing rows through
    # a session this block has already closed.
    async with (
        BraveClient.from_settings(settings) as brave,
        PageFetcher.from_settings(settings) as fetcher,
        LlmClient.from_settings(settings, provider_id=provider_id) as judge,
        asyncio.TaskGroup() as group,
    ):
        for package in candidates:
            group.create_task(
                _corroborate_and_store(
                    ctx,
                    brave=brave,
                    fetcher=fetcher,
                    judge=judge,
                    package=package,
                    description=proposals[package],
                    settings=settings,
                    budget=budget,
                    counts=counts,
                    unrecorded=unrecorded,
                )
            )
    logger.info(
        "job %s corroborate: %s over %d candidate(s), %d unrecorded, %d of %d search "
        "quer(ies) spent",
        ctx.job_id,
        ", ".join(f"{status} {count}" for status, count in sorted(counts.items())),
        len(candidates),
        len(unrecorded),
        budget.ceiling - budget.remaining,
        budget.ceiling,
    )
    if unrecorded:
        raise StageRecordError(
            f"job {ctx.job_id} corroborate: {len(unrecorded)} of {len(candidates)} package(s) "
            f"were searched or judged and could not be written, their own failure row "
            f"included: {', '.join(sorted(unrecorded))}. Check the database is reachable and "
            "writable, then re-run — the cached search rows mean the retry spends no quota."
        )


async def _branch_params(ctx: StageContext) -> BranchEmissionJobParams:
    async with ctx.session_factory() as session:
        job = await session.get(Job, ctx.job_id)
        params = dict(job.params) if job is not None else None
    if params is None:
        raise StageInputError(f"_branch_params: job {ctx.job_id} no longer exists")
    try:
        return BranchEmissionJobParams.model_validate(params)
    except ValidationError as exc:
        raise StageInputError(
            f"_branch_params: job {ctx.job_id} carries params a branch_emission job cannot run "
            f"from ({params!r}). It needs the vendor to emit, e.g. {{'vendor': 'pixel'}}: {exc}"
        ) from exc


def _pipeline_version() -> str:
    """This package's own version, for the disclosure. Read from the installed distribution
    because there is no `__version__` in the tree and a second copy of the number is a second
    thing to forget to bump."""
    return importlib.metadata.version("uadclaw")


async def _reconcile_emission(ctx: StageContext, *, repo: Path, recorded: EmissionRecord) -> None:
    """Recover the outcome of an emission whose recording never landed.

    Reached when this job already wrote an intent, that intent carries no commit, and the
    branch it named is present in the clone: the git commit succeeded and the write after it
    did not.

    **The branch is verified, not merely recognised, and bounding the FILE does not bound the
    BRANCH.** An earlier revision compared the list blob's digest and adopted the branch tip,
    which is wrong in the exact shape this repo already found and fixed inside `emit_branch`:
    a successful emission leaves the clone checked out ON the emission branch, so a human
    working in their own clone during the crash window commits onto it, their commit touches
    nothing this pipeline wrote, the blob still matches, and their work is adopted as this
    batch's outcome and disclosed to upstream as such. `verify_emitted_branch` makes the same
    three read-backs `emit_branch` makes about its own commit — parent, touched paths, bytes —
    so a recovery can never accept what the emission itself would have rejected.

    The digest compared is the one the INTENT recorded, never a freshly derived batch: a
    reviewer approving one more package between the crash and the retry must not make a
    perfectly good branch look wrong.
    """
    verified = await asyncio.to_thread(
        verify_emitted_branch,
        repo,
        branch=recorded.branch,
        base_commit=recorded.base_commit,
        list_path=recorded.list_path,
        list_sha256=recorded.list_sha256,
    )
    if verified.root != recorded.repo_path:
        # Belt and braces rather than the safety gate — the checks above already make an
        # accidental adopt from another checkout implausible — but it is the difference
        # between a confusing error and one naming what the operator changed.
        raise StageInputError(
            f"job {ctx.job_id} branch: {recorded.branch} was found in {verified.root}, while "
            f"this job's emission happened in {recorded.repo_path}. UPSTREAM_REPO_PATH has "
            "been repointed since; point it back to record the outcome, or resolve the "
            "emission by hand."
        )
    async with ctx.session_factory() as session, session.begin():
        await record_commit(
            session,
            emission_id=recorded.id,
            commit_oid=verified.commit,
            at=datetime.now(UTC),
            reconciled=True,
        )
    logger.warning(
        "job %s branch: %s was already committed at %s by an earlier attempt whose recording "
        "did not land; verified and recovered the commit rather than cutting a second branch",
        ctx.job_id,
        recorded.branch,
        verified.commit[:12],
    )


async def _forget_if_rolled_back(
    ctx: StageContext, *, repo: Path, emission_id: int, branch: str
) -> None:
    """After a failed `emit_branch`, discard the intent only once the rollback is CONFIRMED.

    `emit_branch` documents that it deletes the branch it created on every failure path, and
    this deliberately reads that back instead of trusting it — the one case where it does not
    hold is the one that matters, a rollback that itself failed, and that error text is prose
    nobody should be matching on. A branch still standing means the intent has to survive so
    the next run's recovery path can verify and adopt it.

    Never raises: it runs on a failure path, and the exception it is unwinding is the one the
    operator needs to see. Anything it cannot do leaves the intent in place, which is the
    conservative direction — the next run refuses rather than re-emitting.
    """
    try:
        if await asyncio.to_thread(branch_exists, repo, branch):
            logger.warning(
                "job %s branch: %s failed but %s still exists in %s, so the intent is kept for "
                "the next run to verify rather than discarded",
                ctx.job_id,
                "emit_branch",
                branch,
                repo,
            )
            return
        async with ctx.session_factory() as session, session.begin():
            await forget_intent(session, emission_id=emission_id)
    except Exception:
        logger.exception(
            "job %s branch: could not confirm whether %s was rolled back; the intent is kept, "
            "so the next run will refuse rather than emit a second time",
            ctx.job_id,
            branch,
        )


# One upstream clone, one work tree, and `worker_pool_size` is a documented tuning knob that
# can exceed 1. Two `branch_emission` jobs driven at once both checkout-and-commit into that
# one clone and fail each other's read-backs, each blaming "a commit hook, or another process"
# without naming the other job — the cause an operator cannot guess. `JOB_KIND_NEEDS_SCRATCH`
# does not serialize them (BRANCH_EMISSION is False by design: emission writes the clone, not
# scratch), so this lock is the seam. The worker runs every job in one asyncio loop, so
# `async with` awaits rather than blocks the loop, exactly as the `asyncio.to_thread` calls do.
_emission_clone_lock = asyncio.Lock()


async def branch_stage(ctx: StageContext) -> None:
    """Commit one vendor's approved entries onto a branch in the operator's own clone.

    The pipeline's last act, and the only one that writes outside this project. Nothing here
    pushes or authenticates — there is no GitHub credential in this stack (design decision 8)
    — so the deliverable is a local branch plus a PR body on the emission row, and a human
    pushes it.

    **Every `upstreamrepo` call goes through `asyncio.to_thread`.** That module is synchronous
    subprocess work by design, `GIT_TIMEOUT_SECONDS` is 120 per git call and `inspect_repo`
    makes several, so an inline call would block the worker's event loop — heartbeat included
    — for minutes against a clone with a wedged hook, and the job would be reclaimed out from
    under itself.

    **The whole emission is pinned to one object id.** `inspect_repo` resolves the base ref
    once and `emit_branch` is then handed `state.base_commit` rather than the ref name, so the
    bytes are spliced from the same blob the branch is cut from. Passing the ref twice would
    let a `git fetch` landing between the two calls rebase the emission onto a newer commit
    while the bytes still came from the older one, which reverts whatever arrived in between
    and looks exactly like a correct branch.

    **The intent is written and COMMITTED before a single git command runs, and this is the
    crash-resumability decision.** The alternative — emit first, record after — was rejected
    because of what each ordering can lose. Recording first can lose only the OUTCOME, and the
    outcome is still readable off the clone: the branch is there or it is not, and its
    committed bytes say whether it is ours. Emitting first loses the BATCH — which packages,
    at which ratings, against which floors, with which body — and nothing in the clone records
    that; worse, `emit_branch` refuses a branch that already exists, so the retry would have no
    way to tell its own previous emission from somebody else's branch of the same name and
    would be wedged on a refusal with no record to resolve it against. So the row goes first,
    `commit_oid IS NULL` marks the window, and the four states a retry can meet are:

    - no row: nothing happened, emit.
    - a row with a commit: this job is done, no-op, and no second branch is cut.
    - a row with no commit and the branch present: the commit landed and the recording did
      not. `_reconcile_emission` VERIFIES the branch and recovers the oid, marking the row
      `reconciled` so it says which of the two ways it got there rather than claiming the
      pipeline watched it happen.
    - a row with no commit and no branch: **refused**, and this is the one state that is
      genuinely ambiguous. It is either an emission that died before committing anything or
      one that committed, got pushed, and had its branch deleted before the recording landed;
      nothing inside the clone separates them, and the two want opposite actions. Re-emitting
      would be the guess that sends a batch upstream twice, so the ambiguity resolves toward
      the refusal. What keeps that from swallowing the ORDINARY failed attempt is
      `_forget_if_rolled_back`: a failing `emit_branch` reads the clone back and, when its
      branch really is gone, deletes the intent, so a retryable failure leaves no row at all
      and reaches the first state instead of this one. The residual cost is stated rather than
      hidden — a worker killed inside the emission itself lands here and needs a human to clear
      the row, which is the price of never double-shipping.
    """
    settings = get_settings()
    params = await _branch_params(ctx)
    vendor = params.vendor

    repo = settings.upstream_repo_dir
    if repo is None:
        raise StageInputError(
            f"job {ctx.job_id} branch: no upstream clone is configured. Branch emission commits "
            "into a local clone of the upstream repo (a fork is fine); clone one and set "
            "UPSTREAM_REPO_PATH to it. Nothing else in this pipeline needs it, which is why it "
            "is blank by default."
        )
    commit_sha = settings.pipeline_commit_sha.strip()
    if not commit_sha:
        raise StageInputError(
            f"job {ctx.job_id} branch: PIPELINE_COMMIT_SHA is unset, so the PR body cannot say "
            "which commit of this pipeline produced these entries. Upstream's CONTRIBUTING "
            "requires the disclosure and the evidence bundles are only reproducible from a "
            "named commit, so this is refused rather than emitted without it. The worker image "
            "carries no .git: set it from the deploying checkout's `git rev-parse HEAD`."
        )
    list_path = settings.upstream_repo_list_path

    async with ctx.session_factory() as session:
        recorded = await load_emission(session, job_id=ctx.job_id)
    if recorded is not None and recorded.commit_oid is not None:
        logger.info(
            "job %s branch: %s was already emitted at %s; nothing to do",
            ctx.job_id,
            recorded.branch,
            recorded.commit_oid[:12],
        )
        return
    if recorded is not None:
        async with _emission_clone_lock:
            if await asyncio.to_thread(branch_exists, repo, recorded.branch):
                await _reconcile_emission(ctx, repo=repo, recorded=recorded)
                return
        raise StageInputError(
            f"job {ctx.job_id} branch: this job recorded an emission of {recorded.branch} into "
            f"{recorded.repo_path} whose outcome was never written, and that branch is not "
            "there now. Two things look identical from here and only one is safe to act on: "
            "the emission may have died before it committed anything, or it may have committed "
            "and been pushed and deleted before the recording landed — in which case emitting "
            "again would send this batch upstream a second time. This pipeline cannot tell "
            "them apart from inside the clone, so it refuses rather than guessing the "
            "expensive way. Check `git reflog` in that clone and whether the branch was "
            f"pushed; if nothing shipped, delete emission {recorded.id} and re-queue the job."
        )

    async with ctx.session_factory() as session:
        approved = await load_approved(session, vendor=vendor)
    if not approved:
        raise StageInputError(
            f"job {ctx.job_id} branch: nothing is approved under vendor {vendor!r}, so there is "
            "no batch to emit. A branch with no diff on it is not the artifact this stage "
            "makes; approve candidates for that vendor in triage first, and check the vendor "
            "against a device_key's driver half rather than against a package-name prefix."
        )

    branch = branch_name(
        prefix=settings.emission_branch_prefix, vendor=vendor, job_id=str(ctx.job_id)
    )
    async with _emission_clone_lock:
        state: RepoState = await asyncio.to_thread(
            inspect_repo, repo, base_ref=settings.upstream_base_ref, list_path=list_path
        )
    # Nothing retires an approved package once it has shipped — the classification row and the
    # human's `approve` both survive emission — so a vendor's SECOND batch still carries its
    # first, and `insert_entries` refuses a batch carrying an already-carried key WHOLE. That
    # refusal is a safety property and stays; what changes is that a well-formed caller never
    # builds such a batch. Decided against the destination's own bytes at the base commit,
    # which is the only copy that can answer it: the operator's `/data` copy is a different
    # file refreshed on a different day.
    carried = already_carried(state.list_bytes, approved)
    packages = tuple(item for item in approved if item.package not in set(carried))
    if carried:
        logger.info(
            "job %s branch: %d of %d approved %s package(s) are already carried at %s and are "
            "left out of this batch: %s",
            ctx.job_id,
            len(carried),
            len(approved),
            vendor,
            state.base_commit[:12],
            ", ".join(carried),
        )
    if not packages:
        raise StageInputError(
            f"job {ctx.job_id} branch: every package approved under vendor {vendor!r} "
            f"({len(carried)} of them) is already carried in {state.list_path} at "
            f"{state.base_commit[:12]}, so there is nothing left to propose. That is what a "
            "vendor looks like once its batch has merged upstream and the clone has been "
            "pulled; approve more candidates in triage before queueing another emission."
        )

    new_bytes = insert_entries(state.list_bytes, packages)
    digest = hashlib.sha256(new_bytes).hexdigest()
    body = render_pr_body(
        vendor=vendor,
        packages=packages,
        pipeline_version=_pipeline_version(),
        commit_sha=commit_sha,
        base_commit=state.base_commit,
        branch=branch,
        bundle_base_url=settings.emission_bundle_url,
    )
    # Upstream's own convention for a package addition, from the merged `pkg(...)` PRs rather
    # than from this repo's commit style: the message is read by that project's maintainers.
    message = f"pkg({vendor}): add {len(packages)} package(s)"

    async with ctx.session_factory() as session, session.begin():
        emission_id = await record_intent(
            session,
            job_id=ctx.job_id,
            vendor=vendor,
            branch=branch,
            repo_path=state.path,
            list_path=state.list_path,
            base_commit=state.base_commit,
            list_sha256=digest,
            pipeline_version=_pipeline_version(),
            pipeline_commit_sha=commit_sha,
            pr_body=body,
            packages=packages,
            at=datetime.now(UTC),
        )

    async with _emission_clone_lock:
        try:
            commit_oid = await asyncio.to_thread(
                emit_branch,
                repo,
                branch=branch,
                # The resolved object id, never the ref: see the docstring. A fetch landing between
                # the inspection above and this call would otherwise move what the branch is cut
                # from.
                base_ref=state.base_commit,
                list_path=state.list_path,
                new_bytes=new_bytes,
                message=message,
            )
        except BaseException:
            # An ordinary failed attempt has to stay retryable, and it is the ROLLBACK — read back,
            # never assumed — that separates it from the crash window this stage refuses. Discard
            # the intent only once the branch is confirmed gone; anything else keeps it.
            await _forget_if_rolled_back(ctx, repo=repo, emission_id=emission_id, branch=branch)
            raise
    async with ctx.session_factory() as session, session.begin():
        await record_commit(
            session,
            emission_id=emission_id,
            commit_oid=commit_oid,
            at=datetime.now(UTC),
            reconciled=False,
        )
    logger.info(
        "job %s branch: emitted %d %s package(s) onto %s at %s in %s, based on %s",
        ctx.job_id,
        len(packages),
        vendor,
        branch,
        commit_oid[:12],
        state.path,
        state.base_commit[:12],
    )


def pipeline_stage_handlers() -> dict[str, StageHandler]:
    """The stages that have real implementations, across every job kind. The worker no-ops
    any stage missing from this mapping, and `models.JOB_KIND_STAGES` decides which of them
    a given job walks — an entry here does NOT put a stage into every kind's pipeline."""
    return {
        "acquire": acquire_stage,
        "unpack": unpack_stage,
        "extract_facts": extract_facts_stage,
        "corpus_graph": corpus_graph_stage,
        "filter": filter_stage,
        "rule_ladder": rule_ladder_stage,
        "llm": llm_stage,
        "corroborate": corroborate_stage,
        "branch": branch_stage,
    }
