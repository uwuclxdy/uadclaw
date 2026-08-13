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

from uadclaw import emission as emission_module
from uadclaw.classify import UadList
from uadclaw.emission import (
    DOMINANT_KEY_ORDER,
    SHARED_VENDOR,
    ApprovedPackage,
    EmissionError,
    already_carried,
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
    """The real list, or None when this box does not have it.

    An override that points at nothing FAILS rather than skipping: asking for the heavy input
    by name and silently getting the empty path back is how an operator reads a skip line as
    normal and believes a gate ran that never did.
    """
    override = os.environ.get("UADCLAW_HEAVY_UPSTREAM_LIST")
    if override:
        path = Path(override)
        assert path.is_file(), f"UADCLAW_HEAVY_UPSTREAM_LIST is set but not a file: {path}"
        return path
    path = REPO_ROOT / "data" / "uad_lists.json"
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
        build_entry(approved("com.example.app", removal=value, floor="Recommended"))


def test_every_ladder_tier_is_accepted():
    for tier in Removal:
        assert build_entry(approved("com.example.app", removal=str(tier), floor="Recommended"))


def test_a_blank_description_is_refused():
    with pytest.raises(EmissionError, match="not shippable text"):
        build_entry(approved("com.example.app", description="   "))


def test_a_blank_package_name_is_refused():
    with pytest.raises(EmissionError, match="not shippable text"):
        build_entry(approved("  "))


def test_a_control_character_in_a_dependency_is_refused():
    with pytest.raises(EmissionError, match="does not show it for what it is"):
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
    """Matched on the clause only this error carries: both floor errors say "floor", so the
    input would decide which one fired and the assertion would not notice."""
    with pytest.raises(EmissionError, match="A floor that cannot be read"):
        build_entry(approved("com.example.app", floor="Safe"))


# --- upstream's literal formatting rule ----------------------------------------------------


@pytest.mark.parametrize(
    "description",
    [
        "A vendor component.\n Removing it is safe.",
        "A vendor component. \nRemoving it is safe.",
    ],
)
def test_whitespace_next_to_a_newline_refuses_the_batch_naming_the_substring(description: str):
    """PR #1180, a literal maintainer request: it indents the next sentence. Rewriting it here
    would change what a human approved in triage, so the whole batch is refused.

    Only the plain-space spellings are here now. A tab or a NBSP next to a newline is refused
    earlier and harder by the invisible-character check, which names the codepoint — see
    `test_an_invisible_character_never_reaches_the_file`.
    """
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


def test_the_body_is_the_same_string_twice():
    """No clock and no run-scoped id anywhere in emission, the same property `bundle.py` rests
    on: two runs over one approved batch produce one document."""
    assert body() == body()


# --- the batch is walked more than once ----------------------------------------------------


def test_a_one_shot_iterable_emits_every_entry_rather_than_none():
    """`Sequence` is an annotation with no gate behind it and this repo runs no type checker,
    so a generator is what the first real caller writes. Walked more than once it used to go
    empty after the first pass and splice a trailing comma and no entries — invalid JSON, no
    exception, discovered whenever somebody next loaded the file."""
    out = insert_entries(FIXTURE, (item for item in [approved("com.p"), approved("com.q")]))
    parsed = json.loads(out)
    assert {"com.p", "com.q"} <= set(parsed)


def test_render_pr_body_also_accepts_a_one_shot_iterable():
    rendered = body(packages=(item for item in [approved("com.p"), approved("com.q")]))
    assert "`com.p`" in rendered
    assert "`com.q`" in rendered


def test_output_that_does_not_parse_is_refused_rather_than_returned(monkeypatch):
    """The self-check exists because "correct by construction" is an argument. Driven by
    breaking the renderer, which is the bug class it is a backstop for."""
    monkeypatch.setattr(emission_module, "_render_entry", lambda item, newline: '  "com.p": {')
    with pytest.raises(EmissionError, match="does not parse"):
        insert_entries(FIXTURE, [approved("com.p")])


def test_output_carrying_the_wrong_entries_is_refused_rather_than_returned(monkeypatch):
    monkeypatch.setattr(emission_module, "_render_entry", lambda item, newline: '  "com.other": {}')
    with pytest.raises(EmissionError, match="exactly the entries"):
        insert_entries(FIXTURE, [approved("com.p")])


# --- text that must never reach somebody else's repository ---------------------------------


# Every codepoint that renders as nothing, renders as a plain space, or reorders what follows
# it. Named by codepoint rather than embedded literally, because a test file carrying a raw RLO
# is itself a document nobody can read.
INVISIBLE = [
    0x0085,  # NEL
    0x2028,  # LINE SEPARATOR
    0x2029,  # PARAGRAPH SEPARATOR
    0x200B,  # ZERO WIDTH SPACE
    0x202E,  # RIGHT-TO-LEFT OVERRIDE
    0x202D,  # LEFT-TO-RIGHT OVERRIDE
    0x00A0,  # NO-BREAK SPACE
    0xFEFF,  # ZERO WIDTH NO-BREAK SPACE
    0x00AD,  # SOFT HYPHEN
    0x3000,  # IDEOGRAPHIC SPACE
]


@pytest.mark.parametrize("codepoint", INVISIBLE, ids=[f"U+{cp:04X}" for cp in INVISIBLE])
@pytest.mark.parametrize("field", ["package", "description", "labels", "dependencies"])
def test_an_invisible_character_never_reaches_the_file(field: str, codepoint: int):
    """`com.a<ZWSP>b` is written raw by `ensure_ascii=False` and reads as `com.ab` in the diff
    a maintainer approves; `com.a<RLO>b` reverses the rest of the line. Strictly worse than the
    control-character case, which at least escapes to something visible.

    Refused in all four fields including `description` and `labels`, which take no package
    charset — the invisible classes are as invisible in free text as in an identifier.
    """
    char = chr(codepoint)
    package = (
        approved(f"com.a{char}b")
        if field == "package"
        else approved(
            "com.a",
            **{
                "description": {"description": f"A vendor{char}component that does a thing."},
                "labels": {"labels": (f"lab{char}",)},
                "dependencies": {"dependencies": (f"com.dep{char}",)},
            }[field],
        )
    )
    with pytest.raises(EmissionError) as caught:
        insert_entries(FIXTURE, [package])
    assert f"U+{codepoint:04X}" in str(caught.value)


@pytest.mark.parametrize(
    ("codepoint", "expected"),
    [
        (0x0009, "a control character"),
        (0x0000, "a control character"),
        (0x0085, "a control character"),
        (0x200B, "an invisible formatting character"),
        (0x202E, "an invisible formatting character"),
        (0x2028, "a line separator"),
        (0x2029, "a paragraph separator"),
        (0x00A0, "a space that is not the plain ASCII space"),
        (0x3000, "a space that is not the plain ASCII space"),
    ],
    ids=lambda value: f"U+{value:04X}" if isinstance(value, int) else "",
)
def test_the_refusal_says_what_the_character_actually_looks_like(codepoint: int, expected: str):
    """A TAB and a NBSP render as ordinary horizontal space. Calling either "invisible" sends
    a reviewer hunting for something they can already see, so the message names the class."""
    with pytest.raises(EmissionError) as caught:
        insert_entries(FIXTURE, [approved("com.a", labels=(f"lab{chr(codepoint)}",))])
    message = str(caught.value)
    assert expected in message
    assert f"U+{codepoint:04X}" in message
    if codepoint not in (0x200B, 0x202E):
        assert "an invisible formatting character" not in message


@pytest.mark.parametrize("codepoint", [0x0009, 0x0000, 0x0085])
def test_a_control_character_is_named_by_codepoint_and_not_as_unnamed(codepoint: int):
    """No control character has a `unicodedata.name`, so a placeholder would read like a lookup
    failure in this module rather than a property of the codepoint."""
    with pytest.raises(EmissionError) as caught:
        insert_entries(FIXTURE, [approved("com.a", labels=(f"lab{chr(codepoint)}",))])
    assert "unnamed" not in str(caught.value)


@pytest.mark.parametrize("codepoint", [0x0009, 0x0085, 0x2028, 0x2029, 0x00A0, 0x3000])
def test_a_strip_eaten_character_is_still_named_rather_than_reported_as_padding(codepoint: int):
    """The five-plus-one set `str.strip()` removes. Ordering regression guard: if the padding
    check ran first these would report as merely "padded" and never name the codepoint that
    made the name a lookalike. U+2029 belongs here and was missing from my own written list."""
    assert (f"x{chr(codepoint)}").strip() == "x"
    with pytest.raises(EmissionError) as caught:
        insert_entries(FIXTURE, [approved("com.a", labels=(f"lab{chr(codepoint)}",))])
    assert f"U+{codepoint:04X}" in str(caught.value)


def test_every_character_upstream_actually_uses_in_a_key_still_passes():
    """The other half, and the reason an allow-list is defensible here rather than over-fitted:
    the sample is the whole destination population. All 5372 live keys use exactly these 61
    characters, so a rule refusing any of them would refuse a real upstream package name."""
    observed = ".0123456789ABCDEFGHIKLMNOPQRSTUVWX_abcdefghijklmnopqrstuvwxyz"
    assert len(set(observed)) == 61
    for char in observed:
        assert build_entry(approved(f"com.a{char}b"))
    # Wider than what is observed: no live key uses a hyphen, and the charset admits one.
    assert build_entry(approved("com.a-b"))


def test_a_lookalike_key_cannot_be_spelled():
    """The finding-11 class arriving through a different character set. That error message
    reads "It reads identically to the unpadded name in a diff" — so does this one."""
    lookalike = "com.android.settings" + chr(0x200B)
    assert lookalike != "com.android.settings"
    with pytest.raises(EmissionError, match=r"U\+200B"):
        insert_entries(FIXTURE, [approved(lookalike)])


@pytest.mark.parametrize("bad", ["com.a/b", "com.a b", "com.a:b", "com.a" + chr(0xE9), "com.a|b"])
def test_a_package_name_outside_upstreams_charset_is_refused(bad: str):
    with pytest.raises(EmissionError, match="no package name upstream uses"):
        insert_entries(FIXTURE, [approved(bad)])


@pytest.mark.parametrize("field", ["package", "description", "labels", "dependencies"])
def test_an_unpaired_surrogate_is_refused_as_an_emission_error(field: str):
    """androguard decodes manifest strings out of AXML's UTF-16, so a lone surrogate is
    reachable. It used to reach `str.encode` and raise `UnicodeEncodeError`, which is not an
    `EmissionError` and escapes a caller that wraps this stage in one."""
    package = (
        approved("com.a\ud800b")
        if field == "package"
        else approved(
            "com.a",
            **{
                "description": {"description": "A vendor component \ud800 does a thing."},
                "labels": {"labels": ("lab\ud800",)},
                "dependencies": {"dependencies": ("com.dep\ud800",)},
            }[field],
        )
    )
    with pytest.raises(EmissionError, match="unpaired surrogate"):
        insert_entries(FIXTURE, [package])


@pytest.mark.parametrize("bad", ["\x00", "\x07", "\x1b", "\r"])
def test_a_control_character_in_a_description_is_refused(bad: str):
    """`_CONTROL_CHARACTERS` states the reason itself — it stays valid JSON and invisible in a
    diff — and that argument is strictest for the description, the entry's whole content."""
    with pytest.raises(EmissionError, match="does not show it for what it is"):
        insert_entries(FIXTURE, [approved("com.a", description=f"A vendor{bad}component here.")])


def test_a_newline_in_a_label_is_refused():
    """A label is an identifier, so it gets the identifier's check rather than the prose one."""
    with pytest.raises(EmissionError, match="does not show it for what it is"):
        insert_entries(FIXTURE, [approved("com.a", labels=("one\ntwo",))])


@pytest.mark.parametrize(
    "overrides",
    [
        {"dependencies": (" com.dep",)},
        {"needed_by": ("com.dep ",)},
        {"labels": (" lab ",)},
    ],
)
def test_a_whitespace_padded_identifier_is_refused(overrides: dict[str, object]):
    """0 of the 5372 live keys and 0 of the 84 live dependency names carry padding, measured
    2026-08-13. It reads identically in a diff and is a different key to every exact-string
    check here, which is how it walks past the already-carried guard."""
    with pytest.raises(EmissionError, match="padded with whitespace"):
        insert_entries(FIXTURE, [approved("com.a", **overrides)])


def test_a_whitespace_padded_package_name_is_refused():
    with pytest.raises(EmissionError, match="padded with whitespace"):
        insert_entries(FIXTURE, [approved(" com.a ")])


def test_a_padded_name_cannot_slip_past_the_already_carried_guard():
    """The sharp end of the padding rule: the guard compares by exact string, so `"com.x "` is
    not `"com.x"` and upstream would take a second key that renders identically."""
    raw = b'{\n  "com.felica": {\n    "list": "Oem"\n  }\n}\n'
    with pytest.raises(EmissionError, match="padded with whitespace"):
        insert_entries(raw, [approved("com.felica ")])


@pytest.mark.parametrize(
    "key", [" pixel:oriole", "pixel :oriole", "pixel: oriole", "pixel:oriole "]
)
def test_a_padded_device_key_is_refused(key: str):
    """`" pixel"` and `"pixel"` are two vendors, so one OEM's batch would split across two
    branches and neither `insert_entries` call would ever see the other's packages."""
    with pytest.raises(EmissionError, match="device key"):
        vendor_for((key,))


def test_a_short_description_a_human_wrote_is_not_refused():
    """`classify.py`'s 20-600 window is the MODEL's generation budget, interpolated into the
    prompt, and it is addressed to a stage that runs before the triage edit this boundary runs
    after. 231 live entries sit under 20 characters and 88 over 600, so enforcing it here would
    refuse a reviewer's own approved text for a rule upstream's file does not keep."""
    out = insert_entries(FIXTURE, [approved("com.a", description="Font.")])
    assert json.loads(out)["com.a"]["description"] == "Font."


def test_a_long_description_a_human_expanded_is_not_refused():
    long_enough = "A genuinely complicated package. " * 25
    assert len(long_enough) > 600
    out = insert_entries(FIXTURE, [approved("com.a", description=long_enough)])
    assert json.loads(out)["com.a"]["description"] == long_enough


def test_a_description_past_the_sanity_ceiling_is_refused():
    """A ceiling on untrusted bytes rather than an editorial rule: upstream's longest is 1513,
    so anything past this is a manifest string that ran away."""
    with pytest.raises(EmissionError, match="over the 4096 ceiling"):
        insert_entries(FIXTURE, [approved("com.a", description="x" * 4097)])


def test_an_absurd_number_of_dependencies_is_refused():
    """The busiest live entry has 7."""
    with pytest.raises(EmissionError, match="over the"):
        insert_entries(
            FIXTURE, [approved("com.a", dependencies=tuple(f"com.d{n}" for n in range(65)))]
        )


def test_a_name_longer_than_the_ceiling_is_refused():
    with pytest.raises(EmissionError, match="over the 255 ceiling"):
        insert_entries(FIXTURE, [approved("com." + "a" * 256)])


# --- the floor is not optional -------------------------------------------------------------


def test_a_package_with_no_floor_cannot_ship_unchecked():
    """`floor` is `str` rather than `str | None` on purpose. The guard that used to wrap this
    comparison skipped it entirely for exactly the packages whose `rule_ladder` never ran, so
    their rating was emitted with nothing checking it."""
    with pytest.raises(EmissionError, match="A floor that cannot be read"):
        build_entry(approved("com.a", floor=None))


def test_the_floor_check_has_no_reachable_bypass():
    """Enumerated from the dataclass rather than from the call I happened to think of: every
    tier below `Unsafe` must refuse against an `Unsafe` floor, through every public writer."""
    for tier in Removal:
        package = approved("com.a", removal=str(tier), floor="Unsafe")
        if tier is Removal.UNSAFE:
            assert build_entry(package)
            continue
        for writer in (
            lambda p: build_entry(p),
            lambda p: insert_entries(FIXTURE, [p]),
            lambda p: body(packages=[p]),
        ):
            with pytest.raises(EmissionError, match="against a computed floor"):
                writer(package)


# --- markdown a stranger reads and clicks --------------------------------------------------


def row_cells(row: str) -> list[str]:
    """Split a GFM table row the way the spec does: on pipes that are not backslash-escaped."""
    cells: list[str] = []
    current = ""
    escaped = False
    for char in row:
        if escaped:
            current += "|" if char == "|" else "\\" + char
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append(current)
            current = ""
        else:
            current += char
    cells.append(current)
    return cells


def strip_code_spans(markdown: str) -> str:
    """Remove every code span, the way a parser resolves them: an opening run of N backticks
    closes on the next run of exactly N. What is left is the text markdown actually renders as
    prose, which is where a link can form."""
    runs = [(m.start(), m.end()) for m in re.finditer(r"`+", markdown)]
    out = []
    cursor = 0
    index = 0
    while index < len(runs):
        start, end = runs[index]
        width = end - start
        closing = next(
            (pair for pair in runs[index + 1 :] if pair[1] - pair[0] == width),
            None,
        )
        if closing is None:
            index += 1
            continue
        out.append(markdown[cursor:start])
        cursor = closing[1]
        index = runs.index(closing) + 1
    out.append(markdown[cursor:])
    return "".join(out)


def link_targets(markdown: str) -> list[str]:
    """Every URL the rendered body would actually link to.

    Deliberately NOT a regex over the raw text: a `](...)` inside a code span is inert, so a
    raw scan reports the safely-fenced values as links and goes red on correct output. And
    deliberately not row-shaped — a row splitter cannot see an injection in a sentence, which
    is exactly how one survived a review round.
    """
    return re.findall(r"\]\(([^)]*)\)", strip_code_spans(markdown))


def provenance_row(rendered: str, label: str) -> str:
    """The provenance table's row for one field."""
    rows = [line for line in rendered.splitlines() if line.startswith(f"| {label} | ")]
    assert len(rows) == 1, f"expected one {label} row, got {len(rows)}"
    return rows[0]


def package_row(rendered: str) -> str:
    """The one table row for the package under test, never the provenance table's header."""
    rows = [line for line in rendered.splitlines() if line.startswith("| ") and "com." in line]
    assert len(rows) == 1, f"expected one package row, got {len(rows)}"
    return rows[0]


def code_span_content(cell: str) -> str:
    """The text inside a code span, asserting the span actually closes where it should."""
    text = cell.strip()
    fence = text[: len(text) - len(text.lstrip("`"))]
    assert fence, f"not a code span: {cell!r}"
    assert text.endswith(fence), f"span does not close with its own fence: {cell!r}"
    inner = text[len(fence) : -len(fence)]
    assert fence not in inner, (
        f"the fence run appears inside the span, so it closes early: {cell!r}"
    )
    return inner[1:-1] if inner.startswith(" ") and inner.endswith(" ") else inner


@pytest.mark.parametrize("bad", ["|", "\\", "`", "``", "a`b``c", "[t](http://x)"])
def test_every_character_the_cell_layer_handles_is_pinned(bad: str):
    """One case per character `_cell`/`_code` claim to handle. Deleting any one of those
    replacements has to red something; an escaping function with one pinned character is an
    escaping function with two unpinned ones.

    Driven through `model` rather than through a package name: the package charset now refuses
    every one of these outright, so the live injection vector into the cell layer is the
    disclosure strings, which are free-form by necessity — a branch name carries slashes and a
    model id is whatever the API answered with.
    """
    value = f"m{bad}odel"
    rendered = body(packages=[approved("com.a", model=value)])
    row = provenance_row(rendered, "model")
    cells = row_cells(row)
    assert len(cells) == 4, f"row split into {len(cells) - 2} columns: {row!r}"
    assert code_span_content(cells[2]) == value


@pytest.mark.parametrize("bad", ["\n", "\r", "\r\n", "\x0b", "\x0c", "\x85", "\u2028", "\u2029"])
def test_the_cell_layer_collapses_every_line_ending_markdown_knows(bad: str):
    """Driven against `_code` directly, and that is the point rather than a shortcut.

    An earlier version of this docstring claimed the field gating had closed every public route
    to the cell layer. That was wrong, and measurably: `[\x00-\x1f\x7f]` does not match
    U+0085, U+2028 or U+2029, so all three sat in this collapse class and in no refusal, and
    reached `_cell` through `model`, `commit_sha` and `package`. The category-based check now
    does close them, so the guard is unreachable through the public API today — but it is
    reachable-or-not as a function of a check three functions away, which is exactly the kind
    of claim that goes stale without anything failing. So the primitive is pinned as a
    primitive: it holds for any string, which is what keeps it correct the day someone adds a
    seventh disclosure field, or relaxes a category, and does not think about this line.
    """
    assert emission_module._code(f"m{bad}odel") == "`m odel`"


def test_a_control_character_in_a_disclosure_string_is_refused():
    """The reachable half of the same concern: through the public API a line ending in a
    disclosure string never gets as far as the cell layer."""
    with pytest.raises(EmissionError, match="does not show it for what it is"):
        body(packages=[approved("com.a", model="m\rodel")])


PAYLOAD = "ab`[CLICK](http://phish.example)`cd"


def test_the_unhosted_body_links_to_nothing_at_all():
    """The whole-body property, not a per-row one. Every free string carries a backtick-plus-
    link payload at once, and the unhosted path is supposed to emit no link whatsoever — so
    the expected set is empty and any escape from any field shows up as a non-empty answer."""
    rendered = body(
        vendor="pixel",
        packages=[approved("com.a", model=f"m{PAYLOAD}")],
        pipeline_version=PAYLOAD,
        commit_sha=PAYLOAD,
        base_commit=PAYLOAD,
        branch=PAYLOAD,
    )
    assert link_targets(rendered) == []


def test_the_hosted_body_links_only_to_the_bundle_url():
    """The positive leg: the same payload everywhere, but now the body IS supposed to emit
    links, so the assertion is the exact set rather than emptiness. A test that only ever
    expects zero cannot tell a working fence from a renderer that stopped emitting links."""
    rendered = body(
        packages=[approved("com.a", bundle_sha256=SHA_A, model=f"m{PAYLOAD}")],
        pipeline_version=PAYLOAD,
        commit_sha=PAYLOAD,
        base_commit=PAYLOAD,
        branch=PAYLOAD,
        bundle_base_url="https://bundles.example/x",
    )
    assert link_targets(rendered) == [f"https://bundles.example/x/{SHA_A}"]


def test_the_commit_sha_is_fenced_in_the_prose_as_well_as_the_table():
    """Both sites, named. The prose one is in the `else` branch — the unhosted path, which is
    the normal case — so a body rendered with `bundle_base_url` set never reaches it."""
    rendered = body(commit_sha=PAYLOAD)
    assert link_targets(rendered) == []
    assert "pipeline at commit" in rendered
    assert rendered.count(PAYLOAD) == 2


def test_a_backtick_in_a_disclosure_string_cannot_open_a_link():
    """The whole reason `_code` sizes its fence: a value spelled with a backtick used to close
    the span wrapping it, and the rest rendered as a real link a maintainer clicks."""
    rendered = body(branch=PAYLOAD)
    assert code_span_content(row_cells(provenance_row(rendered, "branch"))[2]) == PAYLOAD
    assert link_targets(rendered) == []


def test_a_pipe_in_a_disclosure_string_cannot_end_the_table_row_early():
    rendered = body(branch="main|evil")
    assert len(row_cells(provenance_row(rendered, "branch"))) == 4


def test_a_backslash_in_a_disclosure_string_is_not_doubled():
    """A code span does not process backslash escapes, so doubling one renders a value the
    input does not have — a wrong claim in a document going to another repository."""
    rendered = body(branch="main\\evil")
    assert code_span_content(row_cells(provenance_row(rendered, "branch"))[2]) == "main\\evil"


@pytest.mark.parametrize(
    "bad", ["pixel|x\n## INJECTED HEADING", "  ", "", "pixel devices", "../etc", "pi`xel"]
)
def test_a_vendor_that_is_not_a_driver_name_is_refused(bad: str):
    """It reaches the body as a heading, where a newline in it writes a heading of its own
    inside the disclosure section."""
    with pytest.raises(EmissionError, match="not a driver name"):
        body(vendor=bad)


@pytest.mark.parametrize("good", ["pixel", "samsung", SHARED_VENDOR, "oppo", "nothing"])
def test_every_real_driver_name_is_accepted_as_a_vendor(good: str):
    assert body(vendor=good).startswith(f"## {good}: ")


@pytest.mark.parametrize(
    "field", ["model", "pipeline_version", "commit_sha", "base_commit", "branch"]
)
def test_a_surrogate_in_a_disclosure_string_is_refused_before_the_body_is_built(field: str):
    """`render_pr_body` returns a `str` its caller encodes, so a surrogate that survives to the
    body is a `UnicodeEncodeError` in the git lane where the contract promises an
    `EmissionError`. These five reach the body and never the file, which is why the file-side
    gating did not cover them."""
    overrides: dict[str, object] = (
        {"packages": [approved("com.a", model="deepseek\ud800v4")]}
        if field == "model"
        else {field: "x\ud800y"}
    )
    with pytest.raises(EmissionError, match="unpaired surrogate"):
        body(**overrides)


def test_a_surrogate_in_the_bundle_url_is_refused():
    with pytest.raises(EmissionError, match="unpaired surrogate"):
        body(bundle_base_url="https://bundles.example/\ud800")


def test_every_string_reaching_the_body_encodes_as_utf8():
    """The property behind the five cases above, stated once: whatever this function returns,
    a caller can write to disk. Enumerated from `render_pr_body`'s own signature rather than
    from the fields I happened to think of."""
    rendered = body(
        vendor="pixel",
        packages=[approved("com.a", model="deepseek-v4-flash")],
        pipeline_version="0.1.0",
        commit_sha="c" * 40,
        base_commit="d" * 40,
        branch="uadclaw/pixel",
        bundle_base_url="https://bundles.example/x",
    )
    assert rendered.encode("utf-8")


@pytest.mark.parametrize("bad", ["https://a.example/x)y", "https://a.example/a b", "https://a(x"])
def test_a_bundle_base_url_that_would_break_its_own_link_is_refused(bad: str):
    """A `)` closes the link target early and drops the rest of the URL into the page as
    text, so the link a maintainer clicks is not the one the operator configured."""
    with pytest.raises(EmissionError, match="bracket or whitespace"):
        body(bundle_base_url=bad)


# --- line endings --------------------------------------------------------------------------


def test_entries_appended_to_a_crlf_document_use_crlf():
    """The old bytes survived either way, but "indented like their neighbours" includes the
    line ending, and silently mixing them turns the next normalisation into a whole-file diff."""
    raw = b'{\r\n  "com.x": {\r\n    "list": "Oem"\r\n  }\r\n}\r\n'
    out = insert_entries(raw, [approved("com.a")])
    assert out.startswith(prefix_through_last_entry(raw))
    assert out.count(b"\n") == out.count(b"\r\n")
    assert json.loads(out)["com.a"]["list"] == "Oem"


def test_entries_appended_to_an_lf_document_stay_lf():
    """The live file is pure LF, measured 2026-08-13: 0 CRLF against 43199 LF."""
    out = insert_entries(FIXTURE, [approved("com.a")])
    assert b"\r" not in out


def test_a_document_already_mixing_line_endings_is_refused():
    raw = b'{\r\n  "com.x": {\n    "list": "Oem"\r\n  }\n}\r\n'
    with pytest.raises(EmissionError, match="mixes CRLF"):
        insert_entries(raw, [approved("com.a")])


# --- already_carried: the pre-filter in front of the refusal ---------------------------------


def test_already_carried_names_what_the_destination_has():
    """Nothing retires an approved package once it has shipped, so without this a vendor's
    second batch still carries its first and `insert_entries` refuses the whole thing —
    including the new approvals. Reported name-sorted so a log line is stable."""
    packages = [approved("com.felicanetworks.mfc"), approved("com.brand.new"), approved("a.b.c")]

    assert already_carried(FIXTURE, packages) == ("com.felicanetworks.mfc",)


def test_already_carried_is_empty_when_the_batch_is_all_new():
    assert already_carried(FIXTURE, [approved("com.brand.new")]) == ()


def test_already_carried_reports_every_collision_not_just_the_first():
    packages = [
        approved("com.felicanetworks.mfc"),
        approved("org.lineageos.jelly"),
        approved("com.brand.new"),
    ]

    assert already_carried(FIXTURE, packages) == (
        "com.felicanetworks.mfc",
        "org.lineageos.jelly",
    )


def test_already_carried_reads_a_document_carrying_a_byte_order_mark():
    """The live file is spliced through `raw_decode` after a BOM skip, so the pre-filter has to
    agree with it about what the document is — otherwise it reports no collisions and hands
    `insert_entries` the batch it was written to keep out of there."""
    assert already_carried("\ufeff".encode() + FIXTURE, [approved("org.lineageos.jelly")]) == (
        "org.lineageos.jelly",
    )


@pytest.mark.parametrize(
    "raw", [b"", b"not json at all", b"[1, 2, 3]", b"{", b"\xff\xfe not utf-8"]
)
def test_a_document_this_cannot_read_yields_no_drops_rather_than_an_error(raw):
    """The one place here that fails open, and it is bounded: the caller hands these same bytes
    to `insert_entries` on the very next line, which parses them under its own rules and
    refuses with the message written for it. A competing "this file is broken" error from a
    pre-filter would put two spellings of one refusal in front of one document."""
    assert already_carried(raw, [approved("com.brand.new")]) == ()


def test_the_refusal_behind_the_pre_filter_is_unchanged():
    """`already_carried` exists so a well-formed caller never builds a batch carrying an
    existing key. It does NOT loosen the refusal, which is a safety property: two members under
    one key make a document whose meaning depends on the reader."""
    with pytest.raises(EmissionError, match="already carried"):
        insert_entries(FIXTURE, [approved("org.lineageos.jelly")])


def test_the_refusal_no_longer_names_a_step_that_cannot_help():
    """It used to say "re-run the filter stage against this copy of the list". `filter` writes
    `package_analysis.queued`, and the approved set is derived from the triage decision log
    rather than from that column, so following that advice changes nothing — verified by
    setting `queued=False` and watching `load_approved` return the package unchanged."""
    with pytest.raises(EmissionError) as excinfo:
        insert_entries(FIXTURE, [approved("org.lineageos.jelly")])

    assert "already_carried" in str(excinfo.value)
    assert "does NOT help" in str(excinfo.value)
