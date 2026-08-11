"""The real pipeline stage handlers, wired to the worker's `StageHandler` signature.

Two job kinds run through here and they share nothing but the signature. `firmware_analysis`
walks `acquire` → `unpack` → `extract_facts` → `corpus_graph` → `filter` → `rule_ladder` and
lives on disk; `classification` walks `llm` alone, holds no scratch lease, and is the only
stage in this repo that spends money. Which stages a kind walks is `models.JOB_KIND_STAGES`,
and it is per kind precisely so a firmware job cannot wander into `llm`.

Every stage writes only inside the job's own `ctx.scratch_dir`, and none of them writes a job
row: job state is the worker's, fenced on `(job_id, worker_id, attempt)`. The handoff between
stages is a small JSON file in scratch rather than a database column, because it describes
files on disk and dies with them.

Retention is enforced as the stages go, not only at the end: `unpack` deletes the firmware
archive and every multi-GB intermediate the moment the files worth keeping are out, and
`extract_facts` deletes the APKs the moment their facts are in Postgres. A job that fails
later must not be sitting on 3.5 GB of zip it no longer needs, and decision 5 (facts-only
retention) is what makes the Samsung and Oppo scope affordable at all.
"""

import asyncio
import json
import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

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
from uadclaw.deepseek import ChatResult, DeepSeekClient, DeepSeekMalformedError
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
from uadclaw.models import Job
from uadclaw.settings import get_settings
from uadclaw.unpack import canonical_device_path, extract_artifacts, unpack_to_partitions
from uadclaw.upstream import load_upstream_list
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


async def _classify_one(
    client: DeepSeekClient,
    bundle: EvidenceBundle,
    *,
    floor: RemovalFloor,
    derivation: ListDerivation,
    max_attempts: int,
) -> tuple[Classification | None, str | None, ChatResult | None, int]:
    """One package: call, validate, retry a bounded number of times, then give up.

    Returns `(classification, park_reason, last_result, attempts)`. Giving up returns a
    reason rather than raising, because one package the model cannot answer must not cost
    the other 47 — the caller records a park row and moves on.

    **This loop and the client's are one layer each, not two on top of each other.** The
    client retries the WIRE (429/500/503, a transport error, a body that is not JSON, the
    documented empty-content bug) and never sees a field. This loop retries the SEMANTICS: a
    well-formed JSON object whose `removal` sits below the floor, or whose description is
    four characters long. They are different failures with different fixes, and a validator
    rejection re-sent unchanged is exactly the retry the design asks for.

    A `DeepSeekBudgetError`, a bad credential and an empty balance all abort the whole job
    rather than parking a package: none of them is about this package, and burning the
    retry cap on each of 48 packages against a dead account is not a diagnosis.
    """
    user = user_prompt(bundle)
    last_result: ChatResult | None = None
    reason = "no attempt was made"
    for attempt in range(1, max_attempts + 1):
        result = await client.complete_json(system=SYSTEM_PROMPT, user=user)
        last_result = result
        try:
            payload = result.json_object()
            classification = validate_response(
                payload, bundle=bundle, floor=floor, derivation=derivation, model=result.model
            )
        except (ClassificationRejected, DeepSeekMalformedError) as exc:
            reason = f"attempt {attempt}/{max_attempts}: {exc}"
            logger.warning("classification rejected: package=%s %s", bundle.package, reason)
            continue
        return classification, None, result, attempt
    return None, reason, last_result, max_attempts


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
        "job %s llm: classifying %d of %d queued package(s) with %s (thinking=%s)",
        ctx.job_id,
        len(candidates),
        len(queue),
        settings.deepseek_model,
        settings.deepseek_thinking,
    )
    counts = {"classified": 0, "parked": 0}
    # A TaskGroup rather than `asyncio.gather`: the failures that reach here at all are the
    # ones that abort the whole job (an empty balance, a rejected key, a budget too small for
    # any package), and `gather` propagates the first one while leaving every sibling running
    # — so 47 more packages would keep spending their retry cap against a dead account, and
    # would then be writing rows through a session and an HTTP client this block has already
    # closed. A TaskGroup cancels the siblings and waits for them before it raises.
    async with DeepSeekClient.from_settings(settings) as client, asyncio.TaskGroup() as group:
        for package in candidates:
            group.create_task(
                _classify_and_store(
                    ctx,
                    client,
                    bundles[package],
                    floor=floors[package],
                    identity=identities.get(package),
                    max_attempts=settings.deepseek_max_attempts,
                    counts=counts,
                )
            )
    logger.info(
        "job %s llm: %d classified, %d parked out of %d candidate(s)",
        ctx.job_id,
        counts["classified"],
        counts["parked"],
        len(candidates),
    )


async def _classify_and_store(
    ctx: StageContext,
    client: DeepSeekClient,
    bundle: EvidenceBundle,
    *,
    floor: RemovalFloor,
    identity: PackageIdentity | None,
    max_attempts: int,
    counts: dict[str, int],
) -> None:
    """One package end to end, in its own transaction.

    Per package rather than one transaction for the batch: a database error on package 30
    must not discard 29 answers that were already paid for.
    """
    item = bundle.payload.get("facts", {})
    derivation = derive_list(
        bundle.package,
        cert_issuer=identity.cert_issuer if identity else None,
        partitions=tuple(item.get("partitions", ())),
    )
    classification, reason, result, attempts = await _classify_one(
        client, bundle, floor=floor, derivation=derivation, max_attempts=max_attempts
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
            counts["parked"] += 1
            return
        await store_classification(
            session,
            classification,
            model=result.model if result is not None else client.model,
            thinking=client.thinking,
            usage=usage,
            attempts=attempts,
            at=at,
        )
        counts["classified"] += 1


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
    }
