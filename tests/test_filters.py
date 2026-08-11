"""The additions-queue filter and the upstream list it reads.

Two failure directions, and they are not symmetric. Dropping too much loses candidates and is
recoverable — the corpus is still on record. Dropping too little pushes noise at the human
gate, which is the pipeline's actual bottleneck, and a filter that silently stops filtering
(an empty list read as "nothing is upstream", a prefix match that finds no suffixes) looks
exactly like a corpus with more to offer.
"""

import json
import os
import time
from datetime import UTC, datetime

import pytest

from uadclaw.corpus import CorpusPackage
from uadclaw.filters import (
    FilterVerdict,
    filter_verdict,
    is_auto_generated_rro,
    is_emulator_device,
    queue_verdicts,
    survival,
)
from uadclaw.upstream import UpstreamListError, load_upstream_list

PIXEL = "pixel:oriole"
EMULATOR = "google:emulator-a16"


@pytest.fixture
def upstream(tmp_path):
    path = tmp_path / "uad_lists.json"
    path.write_text(
        json.dumps({"com.listed.one": {"removal": "Recommended"}, "com.listed.two": {}}),
        encoding="utf-8",
    )
    return load_upstream_list(path)


# --- the upstream list ------------------------------------------------------------------------


def test_the_loaded_list_records_which_bytes_decided(tmp_path, upstream):
    """Provenance is the point: "already upstream" is a claim about one specific 1.6 MB of
    JSON, and the row that carries the verdict has to name it."""
    provenance = upstream.provenance()

    assert upstream.entry_count == 2
    assert upstream.packages == {"com.listed.one", "com.listed.two"}
    assert provenance["sha256"] == upstream.sha256
    assert len(provenance["sha256"]) == 64
    assert provenance["entry_count"] == 2
    assert provenance["path"].endswith("uad_lists.json")


def test_obtained_at_is_the_files_own_mtime_not_the_moment_it_was_read(tmp_path):
    """ "How stale is the list this verdict came from" is a property of the file. Reading it
    again must not make a year-old copy look fresh."""
    path = tmp_path / "uad_lists.json"
    path.write_text(json.dumps({"com.listed.one": {}}), encoding="utf-8")
    stamp = datetime(2025, 1, 2, 3, 4, tzinfo=UTC).timestamp()
    os.utime(path, (stamp, stamp))

    loaded = load_upstream_list(path)

    assert loaded.obtained_at == datetime(2025, 1, 2, 3, 4, tzinfo=UTC)
    assert loaded.loaded_at > loaded.obtained_at
    assert loaded.loaded_at.timestamp() == pytest.approx(time.time(), abs=30)


def test_a_missing_list_is_refused_rather_than_read_as_nothing_being_upstream(tmp_path):
    with pytest.raises(UpstreamListError, match="is not a file"):
        load_upstream_list(tmp_path / "absent.json")


def test_an_empty_list_is_refused(tmp_path):
    """The failure this guard exists for: zero entries marks every package as missing and
    pushes the entire corpus into a queue a human walks by hand."""
    path = tmp_path / "uad_lists.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(UpstreamListError, match="carries zero entries"):
        load_upstream_list(path)


def test_a_list_that_is_not_an_object_is_refused(tmp_path):
    path = tmp_path / "uad_lists.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(UpstreamListError, match="not an object"):
        load_upstream_list(path)


def test_a_truncated_list_is_refused(tmp_path):
    path = tmp_path / "uad_lists.json"
    path.write_text('{"com.listed.one": {', encoding="utf-8")

    with pytest.raises(UpstreamListError, match="not valid JSON"):
        load_upstream_list(path)


def test_an_oversized_file_is_refused_unparsed(tmp_path, monkeypatch):
    import uadclaw.upstream as upstream_module

    monkeypatch.setattr(upstream_module, "MAX_UPSTREAM_LIST_BYTES", 8)
    path = tmp_path / "uad_lists.json"
    path.write_text(json.dumps({"com.listed.one": {}}), encoding="utf-8")

    with pytest.raises(UpstreamListError, match="over the"):
        load_upstream_list(path)


# --- the filters -------------------------------------------------------------------------------


def test_a_package_already_upstream_leaves_the_queue(upstream):
    item = CorpusPackage(package="com.listed.one", devices=(PIXEL,))

    assert filter_verdict(item, upstream=upstream) is FilterVerdict.ALREADY_UPSTREAM


def test_an_auto_generated_rro_is_matched_as_a_suffix_not_a_prefix(upstream):
    """The measured shape on real firmware: `com.android.ons.auto_generated_rro_vendor__`. A
    prefix match finds 0 of the 27 on a Pixel 6 and reads as a corpus that has none."""
    item = CorpusPackage(package="com.android.ons.auto_generated_rro_vendor__", devices=(PIXEL,))

    assert is_auto_generated_rro("com.android.ons.auto_generated_rro_vendor__")
    assert not is_auto_generated_rro("com.android.ons")
    assert filter_verdict(item, upstream=upstream) is FilterVerdict.AUTO_GENERATED_RRO


def test_a_package_seen_only_on_emulators_leaves_the_queue(upstream):
    item = CorpusPackage(package="com.android.sdksetup", devices=(EMULATOR,))

    assert filter_verdict(item, upstream=upstream) is FilterVerdict.EMULATOR_ONLY


def test_a_package_on_a_phone_and_an_emulator_stays(upstream):
    """The rule is "only on emulators", not "on an emulator": a package a real Pixel ships is
    a real candidate however many emulator images also carry it."""
    item = CorpusPackage(package="com.example.real", devices=(EMULATOR, PIXEL))

    assert filter_verdict(item, upstream=upstream) is FilterVerdict.QUEUED


def test_overlays_are_not_dropped_as_a_class(upstream):
    """720 of the 5372 live upstream entries match an overlay or RRO pattern, and a Pixel 6
    ships 87 overlays. Dropping the class would discard most of the corpus."""
    item = CorpusPackage(
        package="com.android.settings.overlay.oriole",
        devices=(PIXEL,),
        overlay_target="com.android.settings",
    )

    assert filter_verdict(item, upstream=upstream) is FilterVerdict.QUEUED


def test_a_package_with_no_recorded_device_is_not_treated_as_emulator_only(upstream):
    """`all()` over an empty tuple is true, which would silently drop every package whose
    device list failed to merge."""
    item = CorpusPackage(package="com.example.real", devices=())

    assert filter_verdict(item, upstream=upstream) is FilterVerdict.QUEUED


@pytest.mark.parametrize(
    ("device_key", "expected"),
    [
        ("google:emulator-a16", True),
        ("google:sdk_gphone64_x86_64", True),
        ("aosp:goldfish", True),
        ("aosp:ranchu", True),
        ("pixel:oriole", False),
        ("samsung:emulatorish-s24", True),
        ("pixel:comet", False),
    ],
)
def test_emulator_device_keys_are_recognised_by_the_device_half(device_key, expected):
    assert is_emulator_device(device_key) is expected


def test_the_verdicts_are_ordered_so_the_funnel_reads_as_a_funnel(upstream):
    """An `auto_generated_rro` package that is ALSO already upstream reports the upstream
    verdict: 26 of a Pixel's 27 auto-generated RROs are in fact already carried, and a drop
    reason that hid that would misdescribe the corpus."""
    item = CorpusPackage(package="com.listed.one", devices=(EMULATOR,))

    assert filter_verdict(item, upstream=upstream) is FilterVerdict.ALREADY_UPSTREAM


def test_a_corpus_of_nothing_but_emulator_devices_queues_nothing(upstream):
    """Worth knowing before it looks like a bug: run the pipeline over an emulator image and
    nothing at all reaches the queue, because every package it carries alone is emulator-only
    by definition. The rule is about provenance, so it needs a phone in the corpus to have
    anything to say."""
    corpus = [
        CorpusPackage(package="com.new.one", devices=(EMULATOR,)),
        CorpusPackage(package="com.new.two", devices=(EMULATOR,)),
    ]

    counts = survival(queue_verdicts(corpus, upstream=upstream))

    assert counts["queued"] == 0
    assert counts["emulator_only"] == 2


def test_survival_counts_every_verdict_including_the_empty_ones(upstream):
    corpus = [
        CorpusPackage(package="com.listed.one", devices=(PIXEL,)),
        CorpusPackage(package="com.new.one", devices=(PIXEL,)),
        CorpusPackage(package="com.new.two", devices=(PIXEL,)),
        CorpusPackage(package="com.emu.only", devices=(EMULATOR,)),
    ]

    counts = survival(queue_verdicts(corpus, upstream=upstream))

    assert counts == {
        "total": 4,
        "queued": 2,
        "already_upstream": 1,
        "auto_generated_rro": 0,
        "emulator_only": 1,
    }
