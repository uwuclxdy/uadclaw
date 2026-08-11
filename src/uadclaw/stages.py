"""The real `acquire` and `unpack` pipeline stages, wired to the worker's `StageHandler`
signature.

Both stages write only inside the job's own `ctx.scratch_dir`, and neither writes a job row:
job state is the worker's, fenced on `(job_id, worker_id, attempt)`. The handoff between the
two is a small JSON file in scratch rather than a database column, because it describes
files on disk and dies with them.

Retention is enforced as the stages go, not only at the end: `unpack` deletes the firmware
archive and every multi-GB intermediate the moment the files worth keeping are out. A job
that fails later must not be sitting on 3.5 GB of zip it no longer needs.
"""

import asyncio
import json
import logging
import shutil
from pathlib import Path

from pydantic import BaseModel, ValidationError

from uadclaw.firmware import (
    FirmwareInputError,
    FirmwareJobParams,
    FirmwareRef,
    get_driver,
    select_ref,
)
from uadclaw.models import Job
from uadclaw.settings import get_settings
from uadclaw.unpack import extract_artifacts, unpack_to_partitions
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


def pipeline_stage_handlers() -> dict[str, StageHandler]:
    """The stages that have real implementations. The worker no-ops any stage missing from
    this mapping, so tasks 4+ each add one entry here and nothing else."""
    return {"acquire": acquire_stage, "unpack": unpack_stage}
