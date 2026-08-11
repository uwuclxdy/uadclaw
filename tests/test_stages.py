"""Stage wiring: job-params validation at the API boundary, the acquire -> unpack handoff,
and the acquire stage driving a driver through the worker's `StageContext`.

The driver here is a stub rather than the Pixel one: what is under test is the stage, and a
stage that only works against one OEM would be the wrong shape for the six drivers task 11
adds behind the same interface.
"""

import hashlib
import io
import shutil
import subprocess
import uuid
import zipfile
from pathlib import Path

import pytest

from conftest import utcnow
from uadclaw import firmware as firmware_module
from uadclaw import jobs as jobs_module
from uadclaw.firmware import (
    DownloadedArchive,
    FirmwareDriver,
    FirmwareRef,
    TermsPosture,
    TermsRisk,
)
from uadclaw.models import Job, JobKind, JobState
from uadclaw.settings import Settings, get_settings
from uadclaw.stages import (
    PipelineState,
    StageInputError,
    acquire_stage,
    pipeline_stage_handlers,
    read_state,
    unpack_stage,
    write_state,
)
from uadclaw.worker import StageContext

# What the fake firmware carries, in the exact shape real firmware does: the `system`
# partition's image has a nested `system/` root, so these are `system/...` inside the image
# and `/system/...` on a device.
FAKE_PARTITION_FILES = {
    "system/priv-app/Foo/Foo.apk": b"not really an apk",
    "system/etc/sysconfig/google.xml": b"<config/>",
    "system/etc/permissions/privapp-permissions-platform.xml": b"<permissions/>",
    "system/lib64/libc.so": b"not an artifact",
}

TOOLCHAIN = ("7z", "mke2fs")
needs_toolchain = pytest.mark.skipif(
    any(shutil.which(tool) is None for tool in TOOLCHAIN),
    reason=f"needs the real unpacking toolchain on PATH: {', '.join(TOOLCHAIN)}",
)


def build_ext4_image(tmp_path: Path) -> bytes:
    """A REAL 1 MB ext4 filesystem, built without root via `mke2fs -d`. A blob carrying only
    the ext4 magic would prove the dispatch and nothing about the extraction."""
    source = tmp_path / "partition-src"
    for relative, content in FAKE_PARTITION_FILES.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    image = tmp_path / "system.img"
    subprocess.run(  # noqa: S603
        ["mke2fs", "-q", "-t", "ext4", "-b", "1024", "-d", str(source), str(image), "1024"],
        check=True,
        capture_output=True,
    )
    return image.read_bytes()


class StubDriver(FirmwareDriver):
    name = "pixel"

    def terms(self) -> TermsPosture:
        return TermsPosture(
            risk=TermsRisk.PUBLIC,
            source_url="https://example.invalid",
            summary="",
            acknowledged=True,
        )

    async def list_available(self) -> list[FirmwareRef]:
        return [
            FirmwareRef(
                driver="pixel", device="comet", build="A.1", url="https://example.invalid/a.zip"
            ),
            FirmwareRef(
                driver="pixel", device="comet", build="A.2", url="https://example.invalid/b.zip"
            ),
        ]

    def __init__(self, settings: Settings, image_bytes: bytes | None = None) -> None:
        self.settings = settings
        self.image_bytes = image_bytes

    async def fetch(self, ref: FirmwareRef, dest_dir: Path) -> DownloadedArchive:
        """The real Pixel layout: outer zip -> nested image zip -> raw ext4 partitions."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as zf:
            zf.writestr("system.img", self.image_bytes or b"")
        archive = dest_dir / ref.archive_filename
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("comet-a2/image-comet-a2.zip", inner.getvalue())
            zf.writestr("comet-a2/bootloader-comet-x.img", b"\x00" * 64)
        return DownloadedArchive(
            path=archive,
            sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            integrity_verified=True,
        )


# --- params validation at the boundary ---------------------------------------------------


def test_a_firmware_job_with_a_misspelt_key_is_refused_at_creation():
    with pytest.raises(jobs_module.JobValidationError):
        jobs_module.validate_job_params(
            JobKind.FIRMWARE_ANALYSIS, {"driver": "pixel", "devise": "comet"}
        )


def test_a_valid_firmware_target_round_trips():
    params = jobs_module.validate_job_params(
        JobKind.FIRMWARE_ANALYSIS, {"driver": "pixel", "device": "comet", "build": "AD1A.1"}
    )

    assert params == {"driver": "pixel", "device": "comet", "build": "AD1A.1"}


def test_a_targetless_firmware_job_is_refused_at_creation():
    """An omitted target and an empty one are the same job and must get the same answer.
    Short-circuiting on None skipped the model entirely, so `extra="forbid"` never ran, the
    omitted form returned 201, and the job claimed the single-occupant scratch lease before
    discovering it had nothing to acquire."""
    for params in (None, {}, {"driver": "pixel"}, {"device": "comet"}):
        with pytest.raises(jobs_module.JobValidationError):
            jobs_module.validate_job_params(JobKind.FIRMWARE_ANALYSIS, params)


def test_a_kind_with_no_params_model_still_takes_none():
    """Only kinds that declare a params model are gated; the mapping is what decides, not a
    blanket rule, so task 7's DB-only kinds do not have to invent a target."""
    assert jobs_module.JOB_KIND_PARAM_MODELS[JobKind.FIRMWARE_ANALYSIS] is not None


# --- the acquire -> unpack handoff --------------------------------------------------------


def test_missing_state_names_the_new_attempt_trap(tmp_path):
    with pytest.raises(StageInputError) as excinfo:
        read_state(tmp_path)

    assert "acquire" in str(excinfo.value)


def test_state_round_trips(tmp_path):
    ref = FirmwareRef(driver="pixel", device="comet", build="A.1", url="https://x/y.zip")
    write_state(tmp_path, PipelineState(ref=ref, archive_path=str(tmp_path / "a.zip")))

    state = read_state(tmp_path)

    assert state.ref == ref
    assert state.archive_path == str(tmp_path / "a.zip")


def test_the_registry_wires_the_stages_the_worker_runs():
    handlers = pipeline_stage_handlers()

    assert set(handlers) == {"acquire", "unpack"}


# --- the stages themselves ----------------------------------------------------------------


@pytest.fixture
def stub_driver(monkeypatch, tmp_path):
    """Registered through the real registry, so `get_driver` (and its enable/disable check)
    is on the path under test rather than bypassed."""
    image_bytes = build_ext4_image(tmp_path) if shutil.which("mke2fs") else b""
    monkeypatch.setattr(
        firmware_module,
        "_driver_factories",
        lambda: {"pixel": lambda settings: StubDriver(settings, image_bytes)},
    )


@needs_toolchain
async def test_acquire_then_unpack_extracts_the_artifacts_and_drops_everything_else(
    db_env, db_session_factory, stub_driver, tmp_path
):
    """The whole leg on a real (tiny) ext4 filesystem: resolve a build, fetch it, walk
    zip -> nested zip -> ext4, pull the artifacts, and delete every intermediate."""
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session,
            kind=JobKind.FIRMWARE_ANALYSIS.value,
            params={"driver": "pixel", "device": "comet"},
        )
        job_id = job.id

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    ctx = StageContext(
        job_id=job_id, attempt=1, scratch_dir=scratch, session_factory=db_session_factory
    )

    await acquire_stage(ctx)

    state = read_state(scratch)
    assert state.ref.build == "A.2"  # newest of the two the stub lists
    assert Path(state.archive_path).is_file()

    await unpack_stage(ctx)

    after = read_state(scratch)
    assert after.partitions == ["system"]
    assert after.apk_count == 1
    assert after.artifact_count == 3  # the .so is not an artifact
    extracted = {
        path.relative_to(scratch / "artifacts").as_posix()
        for path in (scratch / "artifacts").rglob("*")
        if path.is_file()
    }
    assert extracted == {
        "system/system/priv-app/Foo/Foo.apk",
        "system/system/etc/sysconfig/google.xml",
        "system/system/etc/permissions/privapp-permissions-platform.xml",
    }
    assert (scratch / "artifacts/system/system/priv-app/Foo/Foo.apk").read_bytes() == (
        FAKE_PARTITION_FILES["system/priv-app/Foo/Foo.apk"]
    )

    # Retention: the archive and every multi-GB intermediate go the moment they stop being
    # needed, not at the end of the job.
    assert after.archive_path is None
    assert not (scratch / "unpack").exists()
    assert list((scratch / "firmware").iterdir()) == []


async def test_acquire_refuses_a_job_whose_target_is_gone(db_env, db_session_factory, tmp_path):
    """The API can no longer create one, so this writes the row directly: the stage keeps its
    own guard for a row that predates the boundary check or was written by hand, and the
    message has to name what is missing rather than blowing up on a KeyError."""
    job_id = uuid.uuid4()
    async with db_session_factory() as session, session.begin():
        session.add(
            Job(
                id=job_id,
                kind=JobKind.FIRMWARE_ANALYSIS.value,
                params={},
                state=JobState.QUEUED,
                attempt=0,
                log_tail="",
                created_at=utcnow(),
            )
        )
    get_settings.cache_clear()
    ctx = StageContext(
        job_id=job_id, attempt=1, scratch_dir=tmp_path, session_factory=db_session_factory
    )

    with pytest.raises(firmware_module.FirmwareInputError):
        await acquire_stage(ctx)


async def test_a_stage_without_scratch_fails_loudly(db_session_factory):
    ctx = StageContext(
        job_id=uuid.uuid4(), attempt=1, scratch_dir=None, session_factory=db_session_factory
    )

    with pytest.raises(StageInputError):
        await acquire_stage(ctx)
