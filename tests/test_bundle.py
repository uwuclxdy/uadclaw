"""The evidence bundle, which is the project's only reproducibility guarantee.

The model output is explicitly not reproducible — DeepSeek has no seed and says so — so the
claim this project makes to upstream is narrower and has to actually hold: the *question* is
pinned. That means the hash has to survive everything that legitimately varies between two
runs over the same corpus, and the only way to know it does is to vary those things on
purpose here rather than to trust that everything happens to be sorted.

What varies between two runs and must not move the hash: the order rows come back from the
database, the order packages are handed to the graph and the ladder, the insertion order of
the upstream mapping, and dict key order anywhere in the payload. What MUST move it: a fact
about the package changing.
"""

import json
from datetime import UTC, datetime

import pytest

from uadclaw.bundle import (
    BUNDLE_SCHEMA_VERSION,
    PackageIdentity,
    build_bundle,
    build_bundles,
    canonical_json,
    nearest_entries,
)
from uadclaw.corpus import CorpusPackage, build_graph
from uadclaw.ladder import Removal, compute_floors
from uadclaw.upstream import UpstreamEntry, UpstreamList

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)


def package(name: str, **overrides) -> CorpusPackage:
    return CorpusPackage(package=name, **overrides)


CORPUS = [
    package(
        "com.android.settings", core_app=True, partitions=("system",), devices=("pixel:oriole",)
    ),
    package(
        "com.android.settings.overlay.oriole",
        overlay_target="com.android.settings",
        partitions=("product",),
        devices=("pixel:oriole",),
    ),
    package("com.vendor.lib", libraries=("com.vendor.support",)),
    package("com.vendor.app", uses_libraries_required=("com.vendor.support",)),
    package(
        "com.example.notes",
        partitions=("product",),
        devices=("pixel:oriole", "google:emulator-a16"),
        device_count=2,
        queries_packages=("com.android.settings", "com.absent.thing"),
    ),
]


def upstream_list(entries: dict[str, tuple[str, str, str]]) -> UpstreamList:
    return UpstreamList(
        path="/data/uad_lists.json",
        sha256="0" * 64,
        obtained_at=NOW,
        loaded_at=NOW,
        entry_count=len(entries),
        packages=frozenset(entries),
        entries={
            name: UpstreamEntry(
                package=name, list=values[0], removal=values[1], description=values[2]
            )
            for name, values in entries.items()
        },
    )


UPSTREAM = upstream_list(
    {
        "com.example.notes.legacy": ("Oem", "Recommended", "Older notes app, replaced."),
        "com.example.calendar": ("Oem", "Advanced", "Calendar app, syncs to the vendor cloud."),
        "com.example.notes.sync": ("Oem", "Advanced", "Sync adapter for the notes app."),
        "com.other.thing": ("Misc", "Recommended", "Unrelated package from another vendor."),
        "com.example.blank": ("Oem", "Recommended", "   "),
    }
)


def build_all(corpus):
    graph = build_graph(corpus)
    floors = compute_floors(corpus)
    return build_bundles(corpus, floors=floors, graph=graph, upstream=UPSTREAM)


# --- determinism ---------------------------------------------------------------------------


def test_the_same_corpus_in_a_different_order_produces_the_same_hashes():
    """The single property this module exists for. The corpus arrives from a database query
    and is handed to the graph, the ladder and the anchor search; if any of those leaks its
    iteration order into the payload, two runs over one corpus disagree about what was asked
    and the hash in the upstream PR body means nothing."""
    forward = build_all(CORPUS)
    reversed_corpus = build_all(list(reversed(CORPUS)))
    shuffled = build_all([CORPUS[3], CORPUS[0], CORPUS[4], CORPUS[2], CORPUS[1]])

    assert {name: bundle.sha256 for name, bundle in forward.items()} == {
        name: bundle.sha256 for name, bundle in reversed_corpus.items()
    }
    assert {name: bundle.sha256 for name, bundle in forward.items()} == {
        name: bundle.sha256 for name, bundle in shuffled.items()
    }


def test_upstream_insertion_order_does_not_move_the_hash():
    """The anchors are picked out of a 5372-entry mapping. Picking "whichever the mapping
    yields first" among equally-near entries would be invisible until the file is re-fetched
    in a different order."""
    corpus = [CORPUS[4]]
    floors = compute_floors(corpus)
    forward = build_bundles(corpus, floors=floors, upstream=UPSTREAM)
    backwards = build_bundles(
        corpus,
        floors=floors,
        upstream=upstream_list(
            {
                name: (entry.list, entry.removal, entry.description)
                for name, entry in reversed(list(UPSTREAM.entries.items()))
            }
        ),
    )
    assert forward["com.example.notes"].sha256 == backwards["com.example.notes"].sha256


def test_dict_key_order_inside_the_payload_does_not_move_the_hash():
    bundles = build_all(CORPUS)
    bundle = bundles["com.example.notes"]
    shuffled_payload = dict(reversed(list(bundle.payload.items())))
    assert canonical_json(shuffled_payload) == canonical_json(bundle.payload)


def test_the_payload_carries_no_timestamp_and_no_run_scoped_id():
    """A clock or a job id anywhere in the payload makes every bundle unique and the anchor
    worthless, and it would do so silently — every hash would still be a hash."""
    blob = canonical_json(build_all(CORPUS)["com.example.notes"].payload).lower()
    for forbidden in ("2026", "timestamp", "job_id", "created_at", "loaded_at", "obtained_at"):
        assert forbidden not in blob, f"{forbidden!r} leaked into the hashed payload"


def test_a_changed_fact_moves_the_hash():
    """The negative control for every determinism test above: a hash that never changes is
    also stable."""
    before = build_all(CORPUS)["com.android.settings"].sha256
    mutated = [
        package(
            "com.android.settings",
            core_app=True,
            partitions=("system",),
            devices=("pixel:oriole",),
            persistent=True,
        )
        if item.package == "com.android.settings"
        else item
        for item in CORPUS
    ]
    assert build_all(mutated)["com.android.settings"].sha256 != before


def test_the_schema_version_is_part_of_the_hash():
    bundles = build_all(CORPUS)
    payload = dict(bundles["com.example.notes"].payload)
    assert payload["schema"] == BUNDLE_SCHEMA_VERSION
    payload["schema"] = BUNDLE_SCHEMA_VERSION + 1
    assert canonical_json(payload) != canonical_json(bundles["com.example.notes"].payload)


# --- anchors -------------------------------------------------------------------------------


def test_anchors_are_the_longest_shared_dotted_prefix_ties_by_name():
    anchors = nearest_entries("com.example.notes", UPSTREAM, limit=3)
    names = [entry.package for entry in anchors]
    # Three shared segments beats two; `notes.legacy` and `notes.sync` tie at three and are
    # ordered by name, and `com.example.calendar` (two) comes after both.
    assert names == ["com.example.notes.legacy", "com.example.notes.sync", "com.example.calendar"]


def test_an_entry_with_a_blank_description_is_not_an_anchor():
    """64 of the 5372 live entries have an empty description. An anchor is a style sample,
    and an empty one teaches the model that empty is acceptable."""
    anchors = nearest_entries("com.example.blank", UPSTREAM, limit=5)
    assert "com.example.blank" not in [entry.package for entry in anchors]
    assert all(entry.description.strip() for entry in anchors)


def test_the_package_itself_is_never_its_own_anchor():
    anchors = nearest_entries("com.other.thing", UPSTREAM, limit=5)
    assert "com.other.thing" not in [entry.package for entry in anchors]


# --- content -------------------------------------------------------------------------------


def test_the_bundle_carries_the_floor_and_every_rule_that_fired():
    bundle = build_all(CORPUS)["com.android.settings"]
    assert bundle.payload["floor"]["floor"] == str(Removal.UNSAFE)
    rules = [item["rule"] for item in bundle.payload["floor"]["fired"]]
    assert "core_app" in rules


def test_the_bundle_carries_graph_edges_as_evidence():
    bundle = build_all(CORPUS)["com.android.settings.overlay.oriole"]
    assert bundle.payload["graph"]["dependencies"] == ["com.android.settings"]
    edges = bundle.payload["graph"]["edges"]
    assert [edge["provider"] for edge in edges] == ["com.android.settings"]


def test_the_identity_half_reaches_the_payload():
    corpus = [CORPUS[4]]
    floors = compute_floors(corpus)
    bundles = build_bundles(
        corpus,
        floors=floors,
        identities={
            "com.example.notes": PackageIdentity(
                label="Notes", cert_issuer="Organization: Example Corp", version_code=42
            )
        },
    )
    facts = bundles["com.example.notes"].payload["facts"]
    assert facts["label"] == "Notes"
    assert facts["cert_issuer"] == "Organization: Example Corp"
    assert facts["version_code"] == 42


def test_a_bundle_built_against_another_packages_floor_is_refused():
    floors = compute_floors(CORPUS)
    with pytest.raises(ValueError, match="floor is for"):
        build_bundle(CORPUS[0], floor=floors["com.vendor.app"])


def test_a_package_with_no_floor_is_refused_rather_than_skipped():
    floors = compute_floors(CORPUS)
    del floors["com.example.notes"]
    with pytest.raises(ValueError, match="has no computed floor"):
        build_bundles(CORPUS, floors=floors)


def test_the_canonical_form_carries_no_structural_whitespace():
    """`json.dumps` defaults to `", "` and `": "` separators. A hash taken over the default
    form is still a hash and still stable, so this is only ever caught by looking."""
    bundle = build_all(CORPUS)["com.example.notes"]
    text = bundle.canonical_json()
    assert json.loads(text) == json.loads(json.dumps(dict(bundle.payload)))
    assert '", "' not in text
    assert '": "' not in text
