"""Branch emission: the bytes that go upstream, and the disclosure that goes with them.

This is the last gate before this pipeline's output leaves the box and becomes somebody else's
repository, so the tests are shaped around the two ways that goes wrong. The first is a diff
nobody will merge: `uad_lists.json` is not uniformly formatted, and any write path that
re-serializes it emits a multi-thousand-line reformat bundled into a content PR. That is
pinned as a BYTE property — the prefix through the last existing entry is compared literally,
because "the JSON is equivalent" is exactly the assertion a reformat passes. The second is an
entry that should never have shipped: a rating under its computed floor, a description a
maintainer already asked not to receive, a package filed under a vendor nobody scanned it on.

The real 1.6 MB file is the byte test's best input and it is gitignored, so it is used when it
is on the box and skipped when it is not, with a fixture carrying the same quirks (two key
orders, uneven indentation, a `suggestions` key upstream's own struct does not declare)
covering the property on every run.
"""

import json
import os
import re
from pathlib import Path

import pytest

from uadclaw.classify import UadList
from uadclaw.emission import (
    DOMINANT_KEY_ORDER,
    SHARED_VENDOR,
    ApprovedPackage,
    EmissionError,
    build_entry,
    group_by_vendor,
    insert_entries,
    render_pr_body,
    vendor_for,
)
from uadclaw.ladder import Removal

REPO_ROOT = Path(__file__).resolve().parents[1]

SHA_A = "a" * 64
SHA_B = "b" * 64

# Two key orders, three indentations and a key upstream's struct does not declare — every
# quirk the live file actually carries, in the smallest document that carries them.
FIXTURE = b"""{
  "org.lineageos.jelly": {
    "list": "Oem",
    "description": "LineageOS Browser App.",
    "dependencies": [],
    "neededBy": [],
    "labels": [],
    "removal": "Recommended"
  },
    "com.cyanogenmod.filemanager": {
    "description": "Cyanogenmod file manager.",
    "removal": "Advanced",
    "suggestions": ["use another one"],
    "list": "Oem",
    "dependencies": [],
    "neededBy": [],
    "labels": []
  },
  "com.felicanetworks.mfc": {
    "list": "Carrier",
    "description": "FeliCa \\u2122 chip driver.",
    "dependencies": [],
    "neededBy": ["com.felicanetworks.mfw.a.boot", "com.felicanetworks.mfm.main"],
    "labels": [],
    "removal": "Unsafe"
  }
}
"""


def approved(package: str, **overrides: object) -> ApprovedPackage:
    fields: dict[str, object] = {
        "package": package,
        "uad_list": "Oem",
        "description": "A vendor component that does a thing, safe to remove.",
        "dependencies": (),
        "needed_by": (),
        "labels": (),
        "removal": "Recommended",
        "floor": "Recommended",
        "bundle_sha256": SHA_A,
        "model": "deepseek-v4-flash",
        "device_keys": ("pixel:oriole",),
    }
    fields.update(overrides)
    return ApprovedPackage(**fields)  # type: ignore[arg-type]


def prefix_through_last_entry(raw: bytes) -> bytes:
    """Everything up to and including the last existing entry's closing brace.

    Derived here by stripping rather than by asking the module, so the assertion is independent
    of the code it is checking: drop the document's trailing whitespace, drop the top-level
    closing brace, drop the whitespace that preceded it.
    """
    stripped = raw.rstrip()
    assert stripped.endswith(b"}")
    return stripped[:-1].rstrip()


def upstream_list_path() -> Path | None:
    override = os.environ.get("UADCLAW_HEAVY_UPSTREAM_LIST")
    path = Path(override) if override else REPO_ROOT / "data" / "uad_lists.json"
    return path if path.is_file() else None


# --- the vendor comes from device provenance, never from a name ----------------------------


def test_one_driver_across_every_device_key_is_the_vendor():
    assert vendor_for(("samsung:SM-S911U", "samsung:SM-S928B")) == "samsung"


def test_two_drivers_make_the_batch_shared():
    assert vendor_for(("pixel:oriole", "samsung:SM-S911U")) == SHARED_VENDOR


def test_the_vendor_is_read_off_the_device_key_and_never_off_the_package_name():
    """A name-prefix rule swept 70 Xiaomi rows into an OPPO bucket in this repo once; the
    grouping derives from a field that exists instead."""
    grouped = group_by_vendor([approved("com.xiaomi.aiservice", device_keys=("pixel:oriole",))])
    assert set(grouped) == {"pixel"}


def test_a_package_with_no_device_keys_is_refused_by_name():
    with pytest.raises(EmissionError, match="com.example.orphan"):
        group_by_vendor([approved("com.example.orphan", device_keys=())])


def test_vendor_for_refuses_an_empty_sequence():
    with pytest.raises(EmissionError, match="no device keys"):
        vendor_for(())


@pytest.mark.parametrize("key", ["oriole", ":oriole", "pixel:", "pixel"])
def test_a_device_key_that_is_not_driver_colon_device_is_refused(key: str):
    with pytest.raises(EmissionError, match="device key"):
        vendor_for((key,))


def test_group_by_vendor_splits_and_sorts_deterministically():
    grouped = group_by_vendor(
        [
            approved("com.b", device_keys=("samsung:SM-S911U",)),
            approved("com.a", device_keys=("samsung:SM-S928B",)),
            approved("com.c", device_keys=("pixel:oriole", "samsung:SM-S911U")),
            approved("com.d", device_keys=("pixel:oriole",)),
        ]
    )
    assert list(grouped) == ["pixel", "samsung", SHARED_VENDOR]
    assert [item.package for item in grouped["samsung"]] == ["com.a", "com.b"]
    assert [item.package for item in grouped[SHARED_VENDOR]] == ["com.c"]


def test_group_by_vendor_refuses_the_same_package_twice():
    """Two copies can carry different removals and land on two different branches, where
    `insert_entries` never sees them together."""
    with pytest.raises(EmissionError, match="appears twice"):
        group_by_vendor(
            [
                approved("com.example.dupe", removal="Recommended"),
                approved("com.example.dupe", removal="Unsafe", device_keys=("samsung:SM-S911U",)),
            ]
        )


# --- one entry's shape ----------------------------------------------------------------------


def test_build_entry_iterates_in_the_dominant_key_order():
    """Spelled out rather than compared against `DOMINANT_KEY_ORDER` alone: a test that
    asserts a constant against itself stays green however the constant is rewritten, and this
    order is the 4267-entry majority in somebody else's file rather than ours to choose."""
    expected = ("list", "description", "dependencies", "neededBy", "labels", "removal")
    assert tuple(build_entry(approved("com.example.app"))) == expected
    assert expected == DOMINANT_KEY_ORDER


def test_build_entry_uses_upstreams_spelling_for_the_two_renamed_fields():
    entry = build_entry(
        approved("com.example.app", uad_list="Aosp", needed_by=("com.example.other",))
    )
    assert entry["list"] == "Aosp"
    assert entry["neededBy"] == ["com.example.other"]


def test_build_entry_renders_absent_name_lists_as_empty_json_arrays():
    entry = build_entry(approved("com.example.app"))
    assert entry["dependencies"] == []
    assert json.dumps(entry["labels"]) == "[]"


@pytest.mark.parametrize("value", ["Pending", "Unlisted", "oem", "", "Vendor"])
def test_a_list_value_upstream_does_not_declare_is_refused(value: str):
    with pytest.raises(EmissionError, match="upstream does not"):
        build_entry(approved("com.example.app", uad_list=value))


def test_every_declared_list_category_is_accepted():
    for value in UadList:
        assert build_entry(approved("com.example.app", uad_list=str(value)))["list"] == value


@pytest.mark.parametrize("value", ["Unlisted", "recommended", "", "Safe"])
def test_a_removal_tier_the_ladder_does_not_carry_is_refused(value: str):
    with pytest.raises(EmissionError, match="not one of"):
        build_entry(approved("com.example.app", removal=value, floor=None))


def test_every_ladder_tier_is_accepted():
    for tier in Removal:
        assert build_entry(approved("com.example.app", removal=str(tier), floor=None))


def test_a_blank_description_is_refused():
    with pytest.raises(EmissionError, match="blank description"):
        build_entry(approved("com.example.app", description="   "))


def test_a_blank_package_name_is_refused():
    with pytest.raises(EmissionError, match="not a package name"):
        build_entry(approved("  "))


def test_a_control_character_in_a_dependency_is_refused():
    with pytest.raises(EmissionError, match="control character"):
        build_entry(approved("com.example.app", dependencies=("com.example\x00.other",)))


def test_a_blank_label_is_refused():
    with pytest.raises(EmissionError, match="label"):
        build_entry(approved("com.example.app", labels=("",)))


def test_an_entry_with_no_bundle_hash_is_refused():
    """The hash IS the evidence disclosure; an entry ships with it or not at all."""
    with pytest.raises(EmissionError, match="bundle_sha256"):
        build_entry(approved("com.example.app", bundle_sha256=""))


def test_an_entry_naming_no_model_is_refused():
    with pytest.raises(EmissionError, match="names no model"):
        build_entry(approved("com.example.app", model=" "))


# --- the floor still binds at the last gate ------------------------------------------------


def test_a_rating_below_its_own_computed_floor_is_refused():
    with pytest.raises(EmissionError, match="against a computed floor"):
        build_entry(approved("com.example.app", removal="Recommended", floor="Unsafe"))


def test_a_rating_above_its_floor_is_accepted():
    assert build_entry(approved("com.example.app", removal="Unsafe", floor="Advanced"))


def test_the_floor_comparison_is_not_the_enums_own_string_order():
    """`"Expert" < "Recommended"` lexicographically, so a string comparison would let an
    `Expert`-floored package ship as `Recommended` and refuse the safe direction."""
    with pytest.raises(EmissionError, match="against a computed floor"):
        build_entry(approved("com.example.app", removal="Recommended", floor="Expert"))
    assert build_entry(approved("com.example.app", removal="Expert", floor="Recommended"))


def test_a_floor_that_is_not_a_tier_is_refused():
    with pytest.raises(EmissionError, match="floor"):
        build_entry(approved("com.example.app", floor="Safe"))


# --- upstream's literal formatting rule ----------------------------------------------------


@pytest.mark.parametrize(
    "description",
    [
        "A vendor component.\n Removing it is safe.",
        "A vendor component. \nRemoving it is safe.",
        "A vendor component.\n\tRemoving it is safe.",
        "A vendor component.\t\nRemoving it is safe.",
        "A vendor component.\n\xa0Removing it is safe.",
    ],
)
def test_whitespace_next_to_a_newline_refuses_the_batch_naming_the_substring(description: str):
    """PR #1180, a literal maintainer request: it indents the next sentence. Rewriting it here
    would change what a human approved in triage, so the whole batch is refused."""
    with pytest.raises(EmissionError) as caught:
        insert_entries(FIXTURE, [approved("com.example.app", description=description)])
    message = str(caught.value)
    assert "com.example.app" in message
    assert repr(re.search(r"[^\S\n]\n|\n[^\S\n]", description).group()) in message


def test_a_newline_with_no_whitespace_beside_it_is_fine():
    out = insert_entries(
        FIXTURE, [approved("com.example.app", description="First sentence.\nSecond sentence.")]
    )
    assert json.loads(out)["com.example.app"]["description"] == "First sentence.\nSecond sentence."


# --- the byte property ---------------------------------------------------------------------


def test_every_byte_up_to_the_last_existing_entry_survives():
    out = insert_entries(FIXTURE, [approved("com.example.app")])
    prefix = prefix_through_last_entry(FIXTURE)
    assert out.startswith(prefix)
    assert out[len(prefix) : len(prefix) + 1] == b","


def test_the_document_parses_to_the_old_mapping_plus_exactly_the_new_entries():
    out = insert_entries(FIXTURE, [approved("com.example.app"), approved("com.example.other")])
    before = json.loads(FIXTURE)
    after = json.loads(out)
    assert set(after) - set(before) == {"com.example.app", "com.example.other"}
    for name, entry in before.items():
        assert after[name] == entry


def test_new_entries_append_in_the_order_given():
    out = insert_entries(FIXTURE, [approved("com.zzz.last"), approved("com.aaa.first")]).decode()
    assert out.index('"com.zzz.last"') < out.index('"com.aaa.first"')


def test_the_appended_block_matches_the_files_prevailing_style():
    out = insert_entries(FIXTURE, [approved("com.example.app")]).decode()
    appended = out[out.index('  "com.example.app"') :]
    assert appended == (
        '  "com.example.app": {\n'
        '    "list": "Oem",\n'
        '    "description": "A vendor component that does a thing, safe to remove.",\n'
        '    "dependencies": [],\n'
        '    "neededBy": [],\n'
        '    "labels": [],\n'
        '    "removal": "Recommended"\n'
        "  }\n"
        "}\n"
    )


def test_an_entry_lands_in_the_dominant_order_whatever_its_neighbours_use():
    """The fixture's last existing entry uses the dominant order and its middle one does not;
    an appended entry follows the constant rather than the neighbourhood."""
    out = insert_entries(FIXTURE, [approved("com.example.app", uad_list="Misc")]).decode()
    block = out[out.index('"com.example.app"') :]
    positions = [block.index(f'"{key}"') for key in DOMINANT_KEY_ORDER]
    assert positions == sorted(positions)


def test_non_ascii_is_written_raw_rather_than_escaped():
    """The live file carries 118 non-ASCII characters and zero `\\uXXXX` escapes, measured
    2026-08-13; an escaped entry would be the odd one out in its own file."""
    out = insert_entries(FIXTURE, [approved("com.example.app", description="Nothing’s ™ app.")])
    assert "Nothing’s ™ app.".encode() in out
    assert b"\\u2122" not in out.split(b'"com.example.app"')[1]


def test_a_closing_brace_inside_a_value_does_not_move_the_splice_point():
    """A hand scan backwards for a `}` is what this guards, and the fixture has to be the
    document that separates one: a member whose value ENDS in a brace inside a string, where a
    scan lands inside the string and splices there. A brace merely somewhere in a description
    does not discriminate — the last brace in that document is still the right one. The live
    file carries neither shape today, but that is a property of one download rather than of
    the format.
    """
    raw = (
        b"{\n"
        b'  "com.a": {\n    "description": "carries a brace } mid-sentence"\n  },\n'
        b'  "com.b": "ends with a brace }"\n'
        b"}\n"
    )
    out = insert_entries(raw, [approved("com.example.app")])
    parsed = json.loads(out)
    assert parsed["com.a"]["description"] == "carries a brace } mid-sentence"
    assert parsed["com.b"] == "ends with a brace }"
    assert out.startswith(prefix_through_last_entry(raw))


def test_a_file_with_no_trailing_newline_gets_none_added():
    raw = b'{\n  "com.a": {\n    "list": "Oem"\n  }\n}'
    out = insert_entries(raw, [approved("com.example.app")])
    assert out.endswith(b"}\n}")
    assert json.loads(out)


def test_an_unusual_trailer_is_copied_through_rather_than_normalised():
    raw = b'{\n  "com.a": {\n    "list": "Oem"\n  }\n}\n\n\n'
    assert insert_entries(raw, [approved("com.example.app")]).endswith(b"}\n\n\n")


def test_a_bom_and_leading_whitespace_survive_the_splice():
    raw = b"\xef\xbb\xbf\n" + b'{\n  "com.a": {\n    "list": "Oem"\n  }\n}\n'
    out = insert_entries(raw, [approved("com.example.app")])
    assert out.startswith(b"\xef\xbb\xbf\n{")
    assert json.loads(out.decode("utf-8-sig"))["com.example.app"]["list"] == "Oem"


def test_a_package_already_in_the_file_is_refused():
    with pytest.raises(EmissionError, match="already carried"):
        insert_entries(FIXTURE, [approved("org.lineageos.jelly")])


def test_the_same_package_twice_in_one_batch_is_refused():
    with pytest.raises(EmissionError, match="appears twice"):
        insert_entries(FIXTURE, [approved("com.example.app"), approved("com.example.app")])


def test_an_empty_batch_is_refused():
    with pytest.raises(EmissionError, match="no packages"):
        insert_entries(FIXTURE, [])


def test_a_file_that_does_not_parse_is_refused():
    with pytest.raises(EmissionError, match="not valid JSON"):
        insert_entries(b'{"com.a": ', [approved("com.example.app")])


def test_a_file_that_is_not_utf8_is_refused():
    with pytest.raises(EmissionError, match="not valid UTF-8"):
        insert_entries(b'{"com.a\xff": {}}', [approved("com.example.app")])


@pytest.mark.parametrize("raw", [b"[]", b'"a string"', b"42"])
def test_a_document_that_is_not_an_object_is_refused(raw: bytes):
    with pytest.raises(EmissionError, match="not an object"):
        insert_entries(raw, [approved("com.example.app")])


def test_an_empty_object_is_refused():
    """An empty list is not a fresh one, it is the wrong file: every existing package would
    look like an addition."""
    with pytest.raises(EmissionError, match="zero entries"):
        insert_entries(b"{}\n", [approved("com.example.app")])


def test_trailing_garbage_after_the_object_is_refused():
    with pytest.raises(EmissionError, match="after its top-level object"):
        insert_entries(FIXTURE + b'{"com.b": {}}\n', [approved("com.example.app")])


def test_the_real_upstream_file_keeps_every_byte_up_to_its_last_entry():
    """The fixture carries the same quirks, but only the real 5372-entry file carries them at
    the scale that makes a reformat a rejection reason. Gitignored, so it skips when absent."""
    path = upstream_list_path()
    if path is None:
        pytest.skip("set UADCLAW_HEAVY_UPSTREAM_LIST to a copy of the upstream uad_lists.json")
    raw = path.read_bytes()
    before = json.loads(raw)
    out = insert_entries(
        raw,
        [approved("com.example.newpkg"), approved("com.example.newpkg.two", uad_list="Misc")],
    )
    prefix = prefix_through_last_entry(raw)
    assert out.startswith(prefix)
    assert len(prefix) > len(raw) - 20
    after = json.loads(out)
    assert set(after) - set(before) == {"com.example.newpkg", "com.example.newpkg.two"}
    assert all(after[name] == entry for name, entry in before.items())
    assert out.endswith(b"\n  }\n}\n")


# --- the disclosure ------------------------------------------------------------------------


def body(**overrides: object) -> str:
    kwargs: dict[str, object] = {
        "vendor": "pixel",
        "packages": [approved("com.example.app"), approved("com.example.other")],
        "pipeline_version": "0.1.0",
        "commit_sha": "1234abcd" * 5,
        "base_commit": "fedc9876" * 5,
        "branch": "uadclaw/pixel-2026-08-13",
    }
    kwargs.update(overrides)
    return render_pr_body(**kwargs)  # type: ignore[arg-type]


def test_the_body_discloses_the_pipeline_version_the_commit_and_the_model():
    rendered = body()
    assert "0.1.0" in rendered
    assert "1234abcd" * 5 in rendered
    assert "deepseek-v4-flash" in rendered


def test_the_body_names_every_distinct_model_in_the_batch():
    rendered = body(
        packages=[approved("com.a", model="deepseek-v4-flash"), approved("com.b", model="other")]
    )
    assert "deepseek-v4-flash" in rendered
    assert "`other`" in rendered


def test_the_body_states_the_entries_are_model_written_and_human_reviewed():
    rendered = body().lower()
    assert "language model" in rendered
    assert "reviewed by a human" in rendered


def test_the_body_states_that_corroboration_was_attempted():
    assert "corroborat" in body().lower()


def test_the_body_carries_one_table_row_per_package():
    rendered = body(packages=[approved("com.a"), approved("com.b"), approved("com.c")])
    rows = [line for line in rendered.splitlines() if line.startswith("| `com.")]
    assert len(rows) == 3
    assert all(row.count("|") == 5 for row in rows)


def test_each_row_carries_the_list_the_removal_and_the_bundle_hash():
    rendered = body(packages=[approved("com.a", uad_list="Aosp", removal="Expert")])
    row = next(line for line in rendered.splitlines() if line.startswith("| `com.a`"))
    assert "Aosp" in row
    assert "Expert" in row
    assert SHA_A in row


def test_an_unhosted_bundle_renders_the_hash_and_no_url():
    """Nothing persists a bundle body, so a link would 404 on the maintainer."""
    rendered = body()
    assert SHA_A in rendered
    assert "](" not in rendered
    assert "http" not in rendered
    assert "regenerated deterministically" in rendered


def test_a_bundle_base_url_links_each_hash():
    rendered = body(
        packages=[approved("com.a", bundle_sha256=SHA_A), approved("com.b", bundle_sha256=SHA_B)],
        bundle_base_url="https://bundles.example/x/",
    )
    assert f"[`{SHA_A}`](https://bundles.example/x/{SHA_A})" in rendered
    assert f"[`{SHA_B}`](https://bundles.example/x/{SHA_B})" in rendered
    assert "regenerated deterministically" not in rendered


@pytest.mark.parametrize("url", ["javascript:alert(1)", "ftp://x/y", "bundles.example"])
def test_a_bundle_base_url_that_is_not_http_is_refused(url: str):
    with pytest.raises(EmissionError, match="http"):
        body(bundle_base_url=url)


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_commit_sha_is_refused(blank: str):
    with pytest.raises(EmissionError, match="no commit sha"):
        body(commit_sha=blank)


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_pipeline_version_is_refused(blank: str):
    with pytest.raises(EmissionError, match="no pipeline version"):
        body(pipeline_version=blank)


def test_a_blank_base_commit_is_recorded_as_unrecorded_rather_than_refused():
    """Navigation aid for the maintainer rather than part of the mandatory disclosure."""
    assert "| base commit | unrecorded |" in body(base_commit="")


def test_render_pr_body_refuses_an_empty_batch():
    with pytest.raises(EmissionError, match="no packages"):
        body(packages=[])


def test_render_pr_body_applies_the_same_gate_as_the_file_writer():
    with pytest.raises(EmissionError, match="against a computed floor"):
        body(packages=[approved("com.a", removal="Recommended", floor="Unsafe")])


def test_a_pipe_in_a_package_name_cannot_end_the_table_row_early():
    rendered = body(packages=[approved("com.a|evil")])
    row = next(line for line in rendered.splitlines() if line.startswith("| `com.a"))
    assert row.count("|") - row.count("\\|") == 5


def test_the_body_is_the_same_string_twice():
    """No clock and no run-scoped id anywhere in emission, the same property `bundle.py` rests
    on: two runs over one approved batch produce one document."""
    assert body() == body()
