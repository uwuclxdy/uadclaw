"""The real chain against a real multi-GB Android image, end to end and opt-in.

Everything else in the suite runs against synthetic bytes; this is the only test that proves
`unpack_to_partitions` + `extract_artifacts` reproduce a manually verified package set. It
needs ~14 GB of scratch, the whole unpacking toolchain, and about a minute, so it runs only
under `UADCLAW_HEAVY_TESTS=1`:

    UADCLAW_HEAVY_TESTS=1 UADCLAW_HEAVY_WORKDIR=/var/tmp/uadclaw-heavy \\
      uv run pytest -n0 -m heavy tests/test_unpack_chain_heavy.py

The corpus is the local emulator system image recorded in `docs/research/local-probe.md`: a
GPT disk whose `super` partition holds five dynamic partitions, four ext4 and one EROFS. It
exercises the branch a Pixel factory zip does NOT take (GPT -> lpunpack -> mixed
filesystems), which is exactly why it is worth keeping alongside the real-firmware run.
"""

import asyncio
import os
import shutil
from pathlib import Path

import pytest

from uadclaw.settings import Settings
from uadclaw.unpack import _extract_erofs, extract_artifacts, unpack_to_partitions

REPO_ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH = REPO_ROOT / "docs" / "research" / "emulator-a16-packages.tsv"
DEFAULT_IMAGE = Path.home() / "Android/Sdk/system-images/android-36.1/google_apis/x86_64/system.img"
IMAGE = Path(os.environ.get("UADCLAW_HEAVY_IMAGE", str(DEFAULT_IMAGE)))
# Peak usage is the carved super (4.1 GB) plus its unpacked partitions (3.6 GB), then the
# extracted artifacts (2.3 GB); the margin covers the image being a little larger.
REQUIRED_FREE_BYTES = 14 * 1024**3

pytestmark = [
    pytest.mark.heavy,
    # The suite-wide 30 s ceiling is sized for unit tests; carving 4 GB out of a disk image
    # and unpacking it takes a minute or two on a fast box and much longer on a slow one.
    pytest.mark.timeout(1800),
    pytest.mark.skipif(
        os.environ.get("UADCLAW_HEAVY_TESTS") != "1",
        reason="opt-in: set UADCLAW_HEAVY_TESTS=1 (needs ~14 GB scratch and the toolchain)",
    ),
]


def ground_truth_apk_paths() -> set[str]:
    """`{partition}/{path inside the partition image}` for all 228 APKs, from the manual
    extraction recorded in the local probe."""
    rows = GROUND_TRUTH.read_text(encoding="utf-8").splitlines()
    return {row.split("\t")[1] for row in rows if row.strip()}


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    """One unpack + extract run shared by every assertion below; `asyncio.run` rather than an
    async fixture so the module scope does not collide with the per-test event loop.

    The work directory is deleted afterwards whatever happens: this test writes ~10 GB, and
    leaving that behind on a dev box is the same bug the pipeline's retention rules exist to
    prevent.
    """
    if not IMAGE.is_file():
        pytest.skip(f"heavy corpus image not on this box: {IMAGE}")
    if not GROUND_TRUTH.is_file():
        pytest.skip(f"ground-truth manifest missing (docs/ is gitignored): {GROUND_TRUTH}")

    work = Path(os.environ.get("UADCLAW_HEAVY_WORKDIR", str(tmp_path_factory.mktemp("heavy"))))
    work.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(work).free
    if free < REQUIRED_FREE_BYTES:
        pytest.skip(f"{work} has {free / 1024**3:.1f} GB free, need ~14 GB")

    settings = Settings(
        postgres_password="test-only-password",
        auth_password="test-only-admin-password",
        session_secret="test-only-session-secret",
    )

    async def run():
        partitions = await unpack_to_partitions(IMAGE, work / "unpack", settings=settings)
        artifacts = await extract_artifacts(partitions, work / "artifacts")
        return partitions, artifacts

    try:
        yield (*asyncio.run(run()), work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_the_gpt_super_disk_yields_every_dynamic_partition(chain):
    partitions, _artifacts, _work = chain

    assert {partition.name for partition in partitions} == {
        "system",
        "system_ext",
        "product",
        "vendor",
        "system_dlkm",
    }


def test_extracted_apk_set_matches_the_manual_extraction_exactly(chain):
    _partitions, artifacts, _work = chain
    extracted = {
        f"{artifact.partition}/{artifact.image_path}"
        for artifact in artifacts
        if artifact.image_path.endswith(".apk")
    }
    expected = ground_truth_apk_paths()

    assert len(expected) == 228
    assert extracted - expected == set(), "extracted APKs the manual run did not find"
    assert expected - extracted == set(), "APKs the manual run found and this run missed"


def test_per_partition_apk_counts_match_the_probe(chain):
    _partitions, artifacts, _work = chain
    counts: dict[str, int] = {}
    for artifact in artifacts:
        if artifact.image_path.endswith(".apk"):
            counts[artifact.partition] = counts.get(artifact.partition, 0) + 1

    assert counts == {"product": 139, "system": 64, "system_ext": 19, "vendor": 6}


def test_config_inputs_come_out_of_every_partition_that_has_them(chain):
    """A directory-shaped extraction that silently yields nothing is the measured failure
    mode here (`7z x -r 'sysconfig/*'` extracts zero files while the listing proves dozens
    exist), so the count is asserted per partition rather than in aggregate."""
    _partitions, artifacts, _work = chain
    configs: dict[str, list[str]] = {}
    for artifact in artifacts:
        if not artifact.image_path.endswith(".apk"):
            configs.setdefault(artifact.partition, []).append(artifact.image_path)

    for partition in ("system", "system_ext", "product", "vendor"):
        assert configs.get(partition), f"{partition} yielded no config inputs at all"
    # sysconfig is populated on three of the four partitions of this image (vendor has none).
    for partition in ("system", "system_ext", "product"):
        assert any("etc/sysconfig/" in path for path in configs[partition]), partition
    assert any("privapp-permissions" in path for path in configs["system"])


def test_every_artifact_is_really_on_disk_and_non_empty(chain):
    _partitions, artifacts, _work = chain

    assert artifacts
    for artifact in artifacts:
        assert artifact.local_path.is_file(), artifact.local_path
        assert artifact.local_path.stat().st_size > 0, artifact.local_path


def test_device_paths_collapse_the_nested_system_root(chain):
    _partitions, artifacts, _work = chain
    device_paths = {artifact.device_path for artifact in artifacts}

    assert "/system/framework/framework-res.apk" in device_paths
    assert not [path for path in device_paths if path.startswith("/system/system/")]
    assert not [path for path in device_paths if path.startswith("/product/product/")]


async def test_erofs_partition_is_read_by_fsck_erofs_not_by_7z(chain, tmp_path):
    """7-Zip has no EROFS handler: pointed at this exact image it reports ONE entry
    (`NOTICE.xml`, its gzip reader guessing) where the partition really holds 105 files, 97
    of them kernel modules. Nothing raises — which is why this asserts a count."""
    partitions, _artifacts, _work = chain
    dlkm = next(partition for partition in partitions if partition.name == "system_dlkm")

    modules, listed = await _extract_erofs(dlkm.path, tmp_path / "dlkm", ["*.ko"])

    assert len(modules) == 97
    assert listed == 105  # what the image yields at all, which is the other half of the count
    assert all((tmp_path / "dlkm" / name).is_file() for name in modules)
