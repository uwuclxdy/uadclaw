"""The real `acquire`, `unpack` and `extract_facts` pipeline stages, wired to the worker's
`StageHandler` signature.

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

from uadclaw.facts import ApkFacts, ApkParseError, parse_apk
from uadclaw.factstore import record_device_scan, store_device_facts
from uadclaw.firmware import (
    FirmwareInputError,
    FirmwareJobParams,
    FirmwareRef,
    get_driver,
    select_ref,
)
from uadclaw.models import Job
from uadclaw.settings import get_settings
from uadclaw.unpack import canonical_device_path, extract_artifacts, unpack_to_partitions
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


def pipeline_stage_handlers() -> dict[str, StageHandler]:
    """The stages that have real implementations. The worker no-ops any stage missing from
    this mapping, so tasks 5+ each add one entry here and nothing else."""
    return {
        "acquire": acquire_stage,
        "unpack": unpack_stage,
        "extract_facts": extract_facts_stage,
    }
