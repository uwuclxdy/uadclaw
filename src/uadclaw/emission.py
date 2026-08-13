"""Approved entries into `uad_lists.json` bytes and into the PR body that discloses them
(`docs/pipeline-design.md` §10). Pure: no DB, no filesystem, no subprocess, no network, no
clock — the same posture as `bundle.py` and for the same reason. A clock in here would make
the emitted bytes differ between two runs over the same approved batch, and "the question is
pinned even though the answer is not" is the only reproducibility claim this project makes.

**Insertion is a textual splice, never a re-serialization.** `json.dump` over the whole
document is not a slower way to do this, it is a different and wrong operation. Measured
2026-08-13 over the live 5372-entry file: two key orders (4267 entries in the dominant
`list, description, dependencies, neededBy, labels, removal`, 1051 in
`description, removal, list, ...`), 51 entries carrying a `suggestions` key the upstream
struct does not declare, three more carrying `leabel`/`labelid`/a leading `labels`, and even
the package-key indentation is not uniform (5362 keys at two spaces, 6 at four, 1 at one). A
round-trip normalises every one of those into a multi-thousand-line reformat diff bundled
into a content PR, which is a documented upstream rejection reason. So every pre-existing
byte survives verbatim and the new entries are spliced in ahead of the closing brace.

Three more things shape the module:

- **The vocabulary is imported, never restated.** `removal` tiers come from `ladder.Removal`
  and `list` categories from `classify.UadList`. A second copy of a value set that ships into
  somebody else's repository is exactly the drift that goes unnoticed until a PR is rejected,
  and `Removal` is additionally never ordered by its own comparison — `danger_rank` is the
  only ordering, because `"Expert" < "Recommended"` lexicographically.
- **Refusal over repair.** A description carrying whitespace next to a newline, a rating under
  its computed floor, a package with no device provenance: each refuses the whole emission
  naming the package. The description is what a human approved in triage, so the fix belongs
  to that human via a triage edit rather than to a silent rewrite here, and a rating below its
  floor came from misreading the same evidence the description was written from.
- **`_validate` is the one seam, and `insert_entries` re-reads its own output.** All three
  public writers pass through `_validate`, so no caller can be the one that forgets — the same
  shape `classifystore._upsert` uses to make the floor structural rather than remembered. The
  splice then re-parses what it is about to return and compares it against the mapping that
  went in plus the entries asked for, because "correct by construction" is an argument and
  this is the one function here that could hand back a corrupt `uad_lists.json`.
- **The PR body is markdown a stranger reads and clicks.** Package names come out of
  downloaded firmware, so every free string in it is fenced as a code span sized to its own
  content, and `vendor` is refused rather than escaped — see `_code` and `_VENDOR`. **A
  hand-written backtick pair around a value is the bug, not a shortcut**: `_cell` neutralises
  a cell's text and does not fence, so ``f"`{_cell(x)}`"`` reads as careful and lets a backtick
  in `x` close the span and open a link. One such pair survived in the prose of the
  unhosted-bundle branch — the DEFAULT path — for a whole review round, four lines from a
  correct `_code` call, because the test that covered it was shaped like a table row and that
  line is a sentence. `_code` is the only spelling; there are no exceptions to grep for.
"""

import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from uadclaw.classify import UadList
from uadclaw.ladder import Removal, danger_rank

# Upstream's dominant key order, and the order every entry this module emits uses regardless
# of what its neighbours in the file happen to use. Emitting the minority order to "match the
# neighbourhood" would make the diff harder to read for no gain: an appended entry has no
# neighbourhood, it lands at the end of the file.
DOMINANT_KEY_ORDER: tuple[str, ...] = (
    "list",
    "description",
    "dependencies",
    "neededBy",
    "labels",
    "removal",
)

# The vendor a package is filed under when its device keys name more than one driver.
SHARED_VENDOR = "shared"

# The live file's prevailing shape, measured 2026-08-13: 5362 of 5372 package keys sit at two
# spaces, and every one of the 32248 field lines but two sits at four.
_KEY_INDENT = "  "
_FIELD_INDENT = "    "

# What RFC 8259 lets stand between tokens, and deliberately not `str.isspace()`: that also
# matches NBSP and U+2028, so skipping them here would accept a leading byte the JSON parser
# itself rejects and hand back a document `json.loads` refuses.
_JSON_WHITESPACE = " \t\n\r"

# Upstream's literal formatting rule, from maintainer `@AnonymousWP` on PR #1180: no space
# adjacent to a `\n` inside a description, because it indents the next sentence. Measured over
# the live file 2026-08-13: 42 entries violate the plain-space spelling.
#
# The class is wider than a literal space, but that width is no longer what carries a tab or a
# NBSP — `_invisible_character` runs first and refuses both outright, naming the codepoint.
# What is left for this rule is the plain ASCII space, which is the only one of the three that
# is legal text everywhere else in a description.
_WHITESPACE_NEXT_TO_NEWLINE = re.compile(r"[^\S\n]\n|\n[^\S\n]")

# The Unicode categories nothing this module writes may contain — Cc control, Cf format, Zl
# line separator, Zp paragraph separator, and any Zs that is not the plain ASCII space —
# mapped to what each actually looks like to a human reading the diff.
#
# Stated as CATEGORIES rather than as a character list on purpose. A hand-listed set was wrong
# here twice: `[\x00-\x1f\x7f]` missed U+0085 NEL, U+2028 and U+2029 — all three sat in the
# markdown line-break class and in no refusal — and it never saw U+200B ZWSP, U+202E RLO,
# U+00AD SHY, U+FEFF, U+00A0 or U+3000 at all. A key spelled `com.a<ZWSP>b` reads as `com.ab`
# in the PR diff a maintainer approves, and `com.a<RLO>b` reverses the rest of the line.
# `ensure_ascii=False` makes it worse than a bare control character, which at least escapes to
# something visible: these land as raw bytes.
#
# The descriptions are not interchangeable and the message quotes the right one. A TAB and a
# NBSP both render as ordinary horizontal space, so telling a reviewer their pasted tab is
# "invisible" sends them hunting for something they can already see. Only Cf is invisible in
# the literal sense. One dict rather than a set plus a lookup table, so widening the refusal
# means writing the description in the same edit, and so the lookup is total — a category
# refused with no entry here would raise `KeyError` instead of `EmissionError`.
#
# Measured over the live file 2026-08-13, so this refuses almost nothing upstream carries:
# across all 5308 non-empty descriptions the only characters in these classes are `\n` (6723,
# exempt in prose) and `\t` (3, refused deliberately — see `prose` in `_check_text`).
_REFUSED_CATEGORIES: dict[str, str] = {
    "Cc": "a control character",
    "Cf": "an invisible formatting character",
    "Zl": "a line separator",
    "Zp": "a paragraph separator",
    "Zs": "a space that is not the plain ASCII space",
}

# A UTF-16 surrogate with no pair. `json.dumps` passes one straight through and `str.encode`
# then raises `UnicodeEncodeError`, which is not an `EmissionError` and would escape a caller
# that wraps this stage in one. Reachable rather than theoretical: manifest strings are
# decoded from AXML's UTF-16 by androguard, which is where a lone surrogate is produced.
_LONE_SURROGATE = re.compile(r"[\ud800-\udfff]")

# A bundle hash as `bundle.bundle_sha256` spells it.
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

# Sanity ceilings on untrusted bytes, and deliberately NOT editorial rules. Every one is set
# far above what upstream actually carries, measured 2026-08-13 over the live file: longest
# description 1513 characters, longest key 81, longest dependency name 40, at most 7
# dependencies and 2 `neededBy` on any one entry, at most 1 label. They exist to refuse a
# manifest string that ran away, not to judge an entry.
#
# `classify.DESCRIPTION_MIN_CHARS`/`DESCRIPTION_MAX_CHARS` (20/600) are deliberately NOT the
# source. Those are the MODEL's generation budget — `classify.py` interpolates them into the
# prompt as an instruction to DeepSeek — and they are addressed to a stage that runs BEFORE
# the human triage edit this boundary runs after. 319 of the 5308 live descriptions break
# them, so a reviewer who tightens a description to "Xiaomi cloud sync service." or expands
# one past 600 to explain a genuinely complicated package would have their approved text
# refuse the whole batch here, for a rule upstream's own file does not keep. There is no
# minimum at all for the same reason: blank is already refused above, and 231 live entries
# sit under 20 characters.
_MAX_DESCRIPTION_CHARS = 4096
_MAX_NAME_CHARS = 255
_MAX_NAMES_PER_FIELD = 64

# Everything markdown reads as a line ending, collapsed to a space inside a table cell. `\r`
# alone is one to CommonMark, so a bare carriage return splits the row exactly like `\n`. A
# RUN collapses to a single space rather than one space per character: a CRLF is ONE line
# ending, and rendering it as two spaces misstates the value it came from.
_MARKDOWN_LINE_BREAK = re.compile("[\\r\\n\\x0b\\x0c\\x85\\u2028\\u2029]+")

# What a package name may contain, applied to the `package` key and to every `dependencies`
# and `neededBy` element. Measured against the whole destination population rather than a
# sample, which is what makes an allow-list defensible here instead of over-fitted: all 5372
# live keys and all 84 live dependency names use exactly 61 distinct characters, every one
# ASCII, and 0 of either fall outside this class — which is itself wider than what is observed,
# since no live key uses a hyphen at all. A name outside it is refused loudly, naming the
# codepoint, so widening this is a one-line change somebody makes on purpose.
_PACKAGE_NAME = re.compile(r"\A[A-Za-z0-9._-]+\Z")

# A vendor is the driver half of a `device_key` or `SHARED_VENDOR`, so it is a registry name
# from `firmware.py` rather than free text. Refused rather than escaped when it is not one: a
# vendor that would need escaping did not come from the registry, and it reaches the PR body
# as a heading where an injected `\n##` writes a heading of its own.
_VENDOR = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._-]*\Z")

# The schemes a bundle base URL may carry. Operator configuration rather than third-party
# input, but it is rendered into a link in a document a stranger clicks.
_BUNDLE_URL_SCHEMES = ("http://", "https://")

# What cannot appear raw in a markdown link target: `)` closes it early and drops the rest of
# the URL into the page as text, and whitespace ends it the same way.
_UNSAFE_IN_LINK_TARGET = re.compile(r"[()\s]")


class EmissionError(RuntimeError):
    """A batch could not be emitted. Bad input to emission — an approved row that is not
    shippable, or a `uad_lists.json` that is not the file it claims to be — never a bug here,
    and always fatal to the whole batch rather than to one entry: these bytes and the PR body
    that discloses them ship as one unit."""


@dataclass(frozen=True, slots=True)
class ApprovedPackage:
    """One human-approved package, as it will appear upstream plus what discloses it.

    `uad_list` rather than `list` because the attribute cannot be named after the builtin;
    `build_entry` maps it back to upstream's spelling, along with `needed_by` -> `neededBy`.

    Nothing here has a default. Every field is either shipped into somebody else's repository
    or is part of the disclosure that ships with it, so a caller that has not decided one has
    not finished, and a default would let it ship the default.
    """

    package: str
    uad_list: str
    description: str
    dependencies: tuple[str, ...]
    needed_by: tuple[str, ...]
    labels: tuple[str, ...]
    removal: str
    # The ladder's computed lower bound. NOT optional, deliberately: a package whose
    # `rule_ladder` never ran has no bound for a rating to sit above, so there is nothing here
    # to check it against and an `ApprovedPackage` for it must not exist. Making it `str`
    # rather than guarding on None moves that refusal onto the lane that builds these from
    # `package_analysis`, whose `floor` column IS nullable — the same shape `ladder.py` uses to
    # make lowering unrepresentable instead of merely forbidden. `Recommended` here means "the
    # ladder ran and nothing fired", never "no ladder result": it is `danger_rank` 0, so a
    # caller writing `floor=row.floor or "Recommended"` gets a floor that is structurally
    # present, semantically absent, and reads exactly like the safe version.
    floor: str
    bundle_sha256: str
    model: str
    # `<driver>:<device>`, from `device_scans.device_key`. The vendor grouping reads the
    # driver half of these and nothing else.
    device_keys: tuple[str, ...]


def _invisible_character(value: str, *, prose: bool) -> str | None:
    """The first character a diff cannot show for what it is, or None.

    Not "invisible": the class spans characters that render as nothing (Cf), as an ordinary
    space (a TAB, a NBSP), and as a line break (Zl, Zp, U+0085). What they share is that the
    text a maintainer reads is not the text that ships, which is the whole refusal.

    `\\n` is exempt in prose and only there: it is legal in a description (6723 live uses) and
    is the one separator a reviewer can actually see in a diff.
    """
    for char in value:
        if prose and char == "\n":
            continue
        category = unicodedata.category(char)
        if category in _REFUSED_CATEGORIES and not (category == "Zs" and char == " "):
            return char
    return None


def _check_text(value: object, *, subject: str, field: str, prose: bool = False) -> str:
    """One string this module writes into `uad_lists.json`, whatever field it came from.

    One seam for every one of them rather than a check per field: a name, a label and a
    description are all UTF-8 that ends up in somebody else's repository, and the ways that
    goes wrong (blank, invisible, unencodable) do not vary by which key it lands under.

    `prose` is the one axis that does, and it is the identifier/prose split rather than a
    per-field exception. An identifier admits no newline and no surrounding whitespace; prose
    admits both. Measured 2026-08-13 over the live file, which decides it rather than taste:
    0 of the 5372 keys and 0 of the 84 dependency names carry surrounding whitespace, against
    1096 of the 5308 descriptions. Padding on an identifier is refused because it is invisible
    in a diff AND because it walks past the `already carried` check — `" com.x"` is not
    `"com.x"` by exact string, so upstream would take a second key rendering identically.
    """
    if not isinstance(value, str) or not value.strip():
        raise EmissionError(
            f"{subject}: {field} is {value!r}, which is not shippable text. Upstream reads "
            "these as `String`, and a blank one becomes an entry nobody can look up."
        )
    found = _invisible_character(value, prose=prose)
    if found is not None:
        # No `unicodedata.name` exists for ANY control character, so the parenthetical is
        # dropped rather than filled with a placeholder: "U+0009 (unnamed)" reads like a
        # lookup failure in this module rather than a property of the codepoint.
        name = unicodedata.name(found, "")
        spelled = f"U+{ord(found):04X}" + (f" {name}" if name else "")
        raise EmissionError(
            f"{subject}: {field} {value!r} carries {spelled}, "
            f"{_REFUSED_CATEGORIES[unicodedata.category(found)]}. The diff a maintainer "
            "approves does not show it for what it is, so the text they read is not the text "
            "that ships; fix it in triage."
        )
    # Runs AFTER the character check, and that order is load-bearing. `str.strip()` removes
    # any character `str.isspace()` accepts, which is five of the refused set — U+0009, U+0085,
    # U+2028, U+2029 and every non-ASCII Zs (U+00A0, U+3000) — so a name ending in one of those
    # would report as merely "padded" and never name the codepoint that made it a lookalike.
    if not prose and value != value.strip():
        raise EmissionError(
            f"{subject}: {field} {value!r} is padded with whitespace. It reads identically to "
            "the unpadded name in a diff and is a different key to every exact-string check "
            "here, so it would slip past the already-carried guard and add a second entry for "
            "one package; fix the triage row rather than trimming it on the way out."
        )
    surrogate = _LONE_SURROGATE.search(value)
    if surrogate is not None:
        raise EmissionError(
            f"{subject}: {field} {value!r} carries the unpaired surrogate "
            f"{surrogate.group()!r}, which is not encodable as UTF-8. It comes from a manifest "
            "string androguard decoded out of AXML's UTF-16; re-extract the package's facts, "
            "or drop it from the batch."
        )
    ceiling = _MAX_DESCRIPTION_CHARS if prose else _MAX_NAME_CHARS
    if len(value) > ceiling:
        raise EmissionError(
            f"{subject}: {field} is {len(value)} characters, over the {ceiling} ceiling. That "
            "is a sanity bound on untrusted bytes rather than an editorial one — the longest "
            "description upstream carries is 1513 — so a value past it is a manifest string "
            "that ran away rather than an entry somebody wrote."
        )
    return value


def _validate(package: ApprovedPackage, *, subject: str) -> None:
    """Everything that must hold before an entry and its disclosure may ship.

    The single seam both writers pass through. Checks are ordered so the message names the
    first thing wrong rather than a consequence of it: the tiers are parsed before the floor
    is compared against one.
    """
    _check_package_name(package.package, subject=subject, field="package")
    name = package.package

    if package.uad_list not in tuple(UadList):
        raise EmissionError(
            f"{subject}: {name} has list {package.uad_list!r}, which upstream does not "
            f"declare. It is one of {', '.join(UadList)}; `Pending` exists in the upstream "
            "enum, is used by zero entries and has no written definition, so this pipeline "
            "cannot propose it either."
        )
    if package.removal not in tuple(Removal):
        raise EmissionError(
            f"{subject}: {name} has removal {package.removal!r}, which is not one of "
            f"{', '.join(Removal)}. `removal` reaches phones through Canta, AppManager and "
            "android-debloat-list as well as uad-ng, so a tier nobody defined is refused."
        )

    _check_text(package.description, subject=subject, field=f"{name}'s description", prose=True)
    offender = _WHITESPACE_NEXT_TO_NEWLINE.search(package.description)
    if offender is not None:
        raise EmissionError(
            f"{subject}: {name}'s description carries {offender.group()!r} — whitespace next "
            "to a newline, which indents the next sentence when the list is rendered and is a "
            "literal upstream request (PR #1180). Rewriting it here would change what a human "
            "approved in triage, so the whole batch is refused; edit the description in "
            "triage and re-emit."
        )

    # Unguarded on purpose. An earlier revision skipped the whole comparison when `floor` was
    # None, which emitted an unchecked rating for exactly the packages whose ladder never ran.
    if package.floor not in tuple(Removal):
        raise EmissionError(
            f"{subject}: {name} carries floor {package.floor!r}, which is not one of "
            f"{', '.join(Removal)}. A floor that cannot be read cannot bound anything; "
            "re-run the rule_ladder stage over this corpus."
        )
    if danger_rank(Removal(package.removal)) < danger_rank(Removal(package.floor)):
        raise EmissionError(
            f"{subject}: {name} is rated {package.removal} against a computed floor of "
            f"{package.floor}. Raising it to the floor here would keep the misreading that "
            "produced it and hide it behind a corrected number, so the batch is refused; "
            "re-classify the package or record a human decision at or above the floor."
        )

    for field_name, values in (
        ("dependencies", package.dependencies),
        ("neededBy", package.needed_by),
        ("labels", package.labels),
    ):
        if len(values) > _MAX_NAMES_PER_FIELD:
            raise EmissionError(
                f"{subject}: {name} carries {len(values)} {field_name} entries, over the "
                f"{_MAX_NAMES_PER_FIELD} ceiling. The busiest live entry has 7; a package with "
                "hundreds is a corpus-graph bug rather than an entry to ship."
            )
        for value in values:
            if field_name == "labels":
                # Upstream free text rather than an identifier, so no package charset — but
                # `_check_text` still refuses the invisible classes, which are as invisible here.
                _check_text(value, subject=subject, field=f"{name}'s label")
            else:
                _check_package_name(value, subject=subject, field=f"{name}'s {field_name} entry")

    if not _SHA256.fullmatch(package.bundle_sha256):
        raise EmissionError(
            f"{subject}: {name} carries bundle_sha256 {package.bundle_sha256!r}, which is not "
            "a sha256 digest. That hash is the entire evidence disclosure the PR body carries, "
            "and an entry ships with its disclosure or not at all."
        )
    if not package.model.strip():
        raise EmissionError(
            f"{subject}: {name} names no model. Disclosing which model wrote a description is "
            "mandatory upstream, so an unrecorded one is a refusal rather than a line quietly "
            "left out of the PR body."
        )
    # Kept after the blank check, whose message is the useful one for this field. The point of
    # the second call is the rest of `_check_text`: `model` reaches the PR body, and a body is
    # a `str` a caller encodes, so an unpaired surrogate here raises `UnicodeEncodeError` in
    # the git lane rather than an `EmissionError` here.
    _check_text(package.model, subject=subject, field=f"{name}'s model")


def _check_package_name(value: object, *, subject: str, field: str) -> str:
    """A package name, which is an identifier rather than text.

    Everything `_check_text` refuses, plus a charset. The charset is what makes a lookalike
    key impossible rather than merely unlikely: `_check_text` catches the invisible classes,
    and this catches everything else that is not what a package name is made of.
    """
    _check_text(value, subject=subject, field=field)
    name = str(value)
    if not _PACKAGE_NAME.fullmatch(name):
        offender = next(char for char in name if not _PACKAGE_NAME.fullmatch(char))
        raise EmissionError(
            f"{subject}: {field} {name!r} carries U+{ord(offender):04X} "
            f"({unicodedata.name(offender, 'unnamed')}), which no package name upstream uses. "
            "All 5372 live keys and all 84 live dependency names are ASCII letters, digits, "
            "dot, underscore or hyphen; a key outside that would be a lookalike of a real one "
            "rather than a package. Re-extract the facts, or widen the charset on purpose."
        )
    return name


def _vendor(device_keys: Sequence[str], *, subject: str) -> str:
    """The driver every one of these device keys names, or `SHARED_VENDOR`."""
    if not device_keys:
        raise EmissionError(
            f"{subject}: no device keys. A package with no device provenance has nothing to "
            "file it under, and its name is not a substitute — a name-prefix rule swept 70 "
            "Xiaomi packages into an OPPO bucket in this repo once already. Re-run the scan "
            "for the devices that shipped it, or drop it from the batch."
        )
    drivers: set[str] = set()
    for key in device_keys:
        driver, separator, device = key.partition(":")
        # Padding is refused on both halves for the reason `_check_text` states, plus one this
        # function owns: `" pixel"` and `"pixel"` are two vendors, so one OEM's batch would
        # split across two branches and neither `insert_entries` call would see the other's.
        if (
            not separator
            or not driver
            or not device
            or driver != driver.strip()
            or device != device.strip()
        ):
            raise EmissionError(
                f"{subject}: {key!r} is not a `<driver>:<device>` device key. The vendor a "
                "branch is filed under is read off that prefix and nothing else, so a key "
                "with no driver half, or one padded so it reads as a second vendor, would "
                "file the package under a vendor nobody wrote; fix the device_scans row "
                "rather than guessing from the package name."
            )
        _check_text(driver, subject=subject, field=f"the driver half of {key!r}")
        _check_text(device, subject=subject, field=f"the device half of {key!r}")
        drivers.add(driver)
    return next(iter(drivers)) if len(drivers) == 1 else SHARED_VENDOR


def vendor_for(device_keys: Sequence[str]) -> str:
    """Which vendor branch a package belongs on, from its device provenance alone.

    One driver across every key is that driver; more than one is `SHARED_VENDOR`, because a
    package two OEMs ship is not either OEM's to review in isolation. Deliberately not derived
    from the package name: the grouping reads a field that exists rather than a heuristic over
    one that only looks like it does.
    """
    return _vendor(device_keys, subject="vendor_for")


def group_by_vendor(packages: Iterable[ApprovedPackage]) -> dict[str, tuple[ApprovedPackage, ...]]:
    """Split an approved batch into the vendor branches it will be pushed as.

    Vendors and the packages under each come back sorted. Upstream's file is unsorted and
    append-only, so nothing downstream needs an order — which is exactly why this one is fixed:
    two runs over the same approved set have to produce the same branch, or the bytes stop
    being comparable and the batch stops being reviewable twice.
    """
    grouped: dict[str, list[ApprovedPackage]] = {}
    seen: set[str] = set()
    for item in packages:
        if item.package in seen:
            raise EmissionError(
                f"group_by_vendor: {item.package} appears twice in one batch. The two copies "
                "can carry different removals and would land on two different branches, where "
                "`insert_entries` never sees them together and neither refuses the other."
            )
        seen.add(item.package)
        vendor = _vendor(item.device_keys, subject=f"group_by_vendor: {item.package}")
        grouped.setdefault(vendor, []).append(item)
    return {
        vendor: tuple(sorted(grouped[vendor], key=lambda item: item.package))
        for vendor in sorted(grouped)
    }


def _entry(package: ApprovedPackage) -> dict[str, Any]:
    values: dict[str, Any] = {
        "list": package.uad_list,
        "description": package.description,
        "dependencies": list(package.dependencies),
        "neededBy": list(package.needed_by),
        "labels": list(package.labels),
        "removal": package.removal,
    }
    # Built by walking the constant rather than by writing the keys out in order, so the
    # emitted order IS `DOMINANT_KEY_ORDER` instead of happening to agree with it.
    return {key: values[key] for key in DOMINANT_KEY_ORDER}


def build_entry(package: ApprovedPackage) -> dict[str, Any]:
    """One entry's value, in upstream's spelling and in the dominant key order."""
    _validate(package, subject="build_entry")
    return _entry(package)


def _render_entry(package: ApprovedPackage, newline: str) -> str:
    """One entry as the text that goes into the file, in the file's own style.

    `ensure_ascii=False` because the live file's own convention is raw UTF-8: measured
    2026-08-13 it carries 118 non-ASCII characters (curly quotes, a `™`, non-breaking hyphens)
    and zero `\\uXXXX` escapes, so escaping would make every new entry the odd one out.
    """
    body = f",{newline}".join(
        f"{_FIELD_INDENT}{json.dumps(key, ensure_ascii=False)}: "
        f"{json.dumps(value, ensure_ascii=False)}"
        for key, value in _entry(package).items()
    )
    key = json.dumps(package.package, ensure_ascii=False)
    return f"{_KEY_INDENT}{key}: {{{newline}{body}{newline}{_KEY_INDENT}}}"


def _document_newline(text: str) -> str:
    """The line ending the document already uses, so appended entries match their neighbours.

    Three cases and only three: no `\\r\\n` at all is LF (the live file, measured 2026-08-13:
    0 CRLF against 43199 LF), every `\\n` preceded by `\\r` is CRLF, and anything between is a
    document already mixed, which is refused rather than guessed at — picking either ending
    for a mixed file makes the next person's normalisation a whole-file diff.
    """
    carriage_returns = text.count("\r\n")
    if carriage_returns == 0:
        return "\n"
    if carriage_returns == text.count("\n"):
        return "\r\n"
    raise EmissionError(
        "insert_entries: the current uad_lists.json mixes CRLF and bare LF line endings "
        f"({carriage_returns} CRLF against {text.count(chr(10))} LF total). Appending in "
        "either ending would leave it mixed and turn the next normalisation into a whole-file "
        "diff; normalise the file in its own commit first, then re-emit."
    )


def insert_entries(raw: bytes, packages: Sequence[ApprovedPackage]) -> bytes:
    """Append approved entries to `uad_lists.json`, leaving every existing byte untouched.

    **How the splice point is found, and why not the obvious way.** The closing brace is
    located by the JSON parser itself: `raw_decode` returns the index one past the value it
    consumed, so `end - 1` is the top-level object's closing brace by the parser's own
    accounting, for any document it accepts. A backwards `rfind(b"}")` would be right on
    today's file and wrong as a rule — it has to guess about a brace inside a description
    string (the live file has none today, but "no description contains a `}`" is a property of
    one download rather than of the format), about trailing whitespace, and about a BOM. This
    way there is nothing to guess: whatever the parser says it consumed is what gets spliced,
    and whatever trailed it is copied through byte for byte, trailing newline or not.

    The insertion point is then the last non-whitespace character before that brace, so the new
    entries follow the last entry's own line rather than the file's closing indentation, and
    the whitespace that separated the last entry from the brace is preserved ahead of it.

    **The result is re-parsed before it is returned.** The splice being correct by construction
    is an argument, and this is the one function in this repo that can hand back a corrupt
    `uad_lists.json`, so it carries a structural guarantee instead: the bytes going out parse,
    and they parse to exactly the mapping that went in plus exactly the entries asked for.
    Measured 2026-08-13 against the real 1.6 MB file, the whole check costs 46 ms — nothing
    against a stage whose input arrived as a multi-GB firmware image.
    """
    # Materialized once because this function walks the batch four times. A generator handed in
    # would be empty by the second walk, and the failure is silent: the collision check and the
    # rendering both see nothing, and the splice writes a trailing comma and no entries.
    items = tuple(packages)
    if not items:
        raise EmissionError(
            "insert_entries: no packages to insert. An empty batch would rewrite the file's "
            "final bytes for no content change and commit an empty diff; whether a vendor "
            "group with nothing approved is a branch worth cutting is the caller's call."
        )

    seen: set[str] = set()
    for item in items:
        if item.package in seen:
            raise EmissionError(
                f"insert_entries: {item.package} appears twice in one batch. Two members with "
                "one key make a JSON object whose meaning depends on the reader — serde and "
                "this repo's own loader keep the last, a human reviewing the diff reads the "
                "first — so it is refused rather than written."
            )
        seen.add(item.package)
        _validate(item, subject="insert_entries")

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EmissionError(
            f"insert_entries: the current uad_lists.json is not valid UTF-8 ({exc}). It is "
            "spliced as text so the new entries can be indented like their neighbours; "
            "re-fetch the file rather than emitting against bytes nobody can read."
        ) from exc

    start = 1 if text.startswith("\ufeff") else 0
    while start < len(text) and text[start] in _JSON_WHITESPACE:
        start += 1
    try:
        parsed, end = json.JSONDecoder().raw_decode(text, start)
    except json.JSONDecodeError as exc:
        raise EmissionError(
            f"insert_entries: the current uad_lists.json is not valid JSON ({exc}). Appending "
            "to a document nobody can parse would produce a second, larger broken file; "
            "re-fetch it from upstream's main."
        ) from exc
    if not isinstance(parsed, dict):
        raise EmissionError(
            f"insert_entries: the current uad_lists.json parsed to {type(parsed).__name__}, "
            "not an object. uad_lists.json is a JSON object keyed by package name."
        )
    if not parsed:
        raise EmissionError(
            "insert_entries: the current uad_lists.json carries zero entries. An empty file is "
            "not a fresh one, it is the wrong file — every existing package would look like an "
            "addition and the batch would be proposing 5372 entries somebody already wrote."
        )
    trailing = text[end:]
    if trailing.strip(_JSON_WHITESPACE):
        raise EmissionError(
            f"insert_entries: the current uad_lists.json carries {trailing.strip()[:40]!r} "
            "after its top-level object. That is not a document this module can splice; the "
            "usual cause is two files concatenated."
        )

    already = sorted(item.package for item in items if item.package in parsed)
    if already:
        raise EmissionError(
            f"insert_entries: {', '.join(already)} already carried in uad_lists.json. An "
            "addition PR that re-adds an existing key is a duplicate-key document upstream's "
            "serde round-trip does not catch; re-run the filter stage against this copy of "
            "the list, which is what decides the additions queue."
        )

    newline = _document_newline(text)
    closing = end - 1
    last = closing - 1
    while last >= start and text[last] in _JSON_WHITESPACE:
        last -= 1
    gap = text[last + 1 : closing]
    rendered = f",{newline}".join(_render_entry(item, newline) for item in items)
    spliced = f"{text[: last + 1]},{newline}{rendered}{gap}}}{trailing}"

    expected = {**parsed, **{item.package: _entry(item) for item in items}}
    try:
        reparsed, _ = json.JSONDecoder().raw_decode(spliced, start)
    except json.JSONDecodeError as exc:
        raise EmissionError(
            f"insert_entries: the spliced document does not parse ({exc}). That is this "
            "module's own output and therefore a bug here rather than bad input; the file on "
            "disk has not been touched. Report it with the batch that produced it."
        ) from exc
    if reparsed != expected:
        raise EmissionError(
            "insert_entries: the spliced document does not carry exactly the entries that "
            "went in plus the ones asked for. That is this module's own output and therefore "
            "a bug here rather than bad input; the file on disk has not been touched."
        )
    return spliced.encode()


def _cell(value: str) -> str:
    """One markdown table cell's text. A package name comes out of downloaded firmware, so a
    `|` in one would end the row early and silently drop every column after it, and any line
    ending would end the row outright.

    Deliberately no backslash doubling. GFM resolves `\\|` while it splits the row, before any
    inline parsing, so the pipe escape survives into a code span — but a code span does not
    process backslash escapes, so doubling one would render `com.a\\b` as `com.a\\\\b`, a wrong
    claim about a package name in a document going to somebody else's repository.
    """
    return _MARKDOWN_LINE_BREAK.sub(" ", value).replace("|", "\\|")


def _code(value: str) -> str:
    """One table cell rendered as a code span, fenced so its content cannot break out.

    Every free string in this body is a code span, and a backtick in the value closes the one
    wrapping it: a package name spelled ``com.a`[t](http://x)`b`` renders the middle as a real
    link a maintainer clicks. CommonMark closes a span only on a backtick run of exactly the
    opening length, so the fence is one longer than the longest run in the content, and content
    that starts or ends with a backtick is padded with the space the renderer then strips.
    """
    text = _cell(value)
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def render_pr_body(
    *,
    vendor: str,
    packages: Sequence[ApprovedPackage],
    pipeline_version: str,
    commit_sha: str,
    base_commit: str,
    branch: str,
    bundle_base_url: str | None = None,
) -> str:
    """The PR body that discloses how these entries were produced.

    Upstream's CONTRIBUTING demands the disclosure and maintainer `@AnonymousWP` asked on
    PR #1180 that AI-written descriptions be checked against an external source, so the body
    states plainly that the entries are LLM-generated, human-reviewed, and put through a
    corroboration pass. `pipeline_version` and `commit_sha` are refused blank: the disclosure
    is mandatory, so an unrecorded one is a refusal rather than a line quietly left out.
    `base_commit` and `branch` are navigation aids for the maintainer rather than part of that
    disclosure, so a blank one is rendered as unrecorded instead of failing the emission.
    """
    items = tuple(packages)
    if not items:
        raise EmissionError(
            "render_pr_body: no packages. A PR body announcing zero additions describes a "
            "branch with no diff on it."
        )
    if not _VENDOR.fullmatch(vendor):
        raise EmissionError(
            f"render_pr_body: vendor {vendor!r} is not a driver name. A vendor is the driver "
            "half of a device_key or the shared marker, so anything else did not come from "
            "`vendor_for`; it reaches the body as a heading, where a newline in it writes a "
            "heading of its own inside the disclosure. Pass what `group_by_vendor` keyed on."
        )
    if not pipeline_version.strip():
        raise EmissionError(
            "render_pr_body: no pipeline version. Which version of this pipeline wrote an "
            "entry is half of what makes the entry auditable, and disclosing it is not "
            "optional upstream; record it rather than opening the PR without it."
        )
    if not commit_sha.strip():
        raise EmissionError(
            "render_pr_body: no commit sha. The evidence bundle is regenerated from a named "
            "commit and from nothing else, so a body without one links its hashes to nothing "
            "and the disclosure is decorative."
        )
    base_url = bundle_base_url
    if base_url is not None:
        if not base_url.startswith(_BUNDLE_URL_SCHEMES):
            raise EmissionError(
                f"render_pr_body: bundle_base_url {base_url!r} is not an http(s) URL. It is "
                "rendered as a link a maintainer clicks; pass None when the bundles are not "
                "hosted, which is the normal case."
            )
        if _UNSAFE_IN_LINK_TARGET.search(base_url):
            raise EmissionError(
                f"render_pr_body: bundle_base_url {base_url!r} carries a bracket or whitespace. "
                "It goes straight into a markdown link target, where a `)` closes the link "
                "early and drops the rest of the URL into the page as text; percent-encode it "
                "in the setting rather than shipping a link that goes somewhere else."
            )
        _check_text(base_url, subject="render_pr_body", field="bundle_base_url")
        base_url = base_url.rstrip("/")
    # Every remaining free string that reaches the body. `render_pr_body` returns a `str` its
    # caller encodes, so an unpaired surrogate in any of these is a `UnicodeEncodeError` in the
    # git lane where the contract promises an `EmissionError`. The blank cases above already
    # raised for the two mandatory ones; `base_commit` and `branch` are skipped when blank
    # because blank is their documented "unrecorded" rendering.
    for label, value in (
        ("pipeline_version", pipeline_version),
        ("commit_sha", commit_sha),
        ("base_commit", base_commit),
        ("branch", branch),
    ):
        if value.strip():
            _check_text(value, subject="render_pr_body", field=label)
    for item in items:
        _validate(item, subject="render_pr_body")

    models = sorted({item.model for item in items})
    # `vendor` is interpolated bare rather than through `_cell`, and that is the point of the
    # charset check above: a value that would need escaping is refused instead.
    lines = [
        f"## {vendor}: {len(items)} package addition(s)",
        "",
        f"{len(items)} new entries for devices this pipeline scanned under the `{vendor}` "
        "driver, appended to `uad_lists.json` in the dominant key order. No existing entry is "
        "touched and nothing anywhere in the file is reformatted.",
        "",
        "### How these entries were produced",
        "",
        "The `list` and `description` fields are **generated by a large language model**, and "
        "every one of them was **reviewed by a human** before it reached this branch — an "
        "entry a reviewer rejected is not here. Each description was additionally put through "
        "an automated corroboration pass that searches the open web and judges whether an "
        "independent source supports the specific claim the description makes. Corroboration "
        "is attempted for every package and succeeds for a minority of them, because most "
        "preinstalled system components have no public footprint to find; a package no source "
        "could support is included on the human review alone, never on the model's word.",
        "",
        "`removal` is bounded below by a deterministic rule ladder computed from the package's "
        "own manifest and from the rest of the firmware it shipped in. The model may rate a "
        "package more dangerous to remove than that bound and can never rate it less: an "
        "answer under the floor is rejected rather than corrected up to it.",
        "",
        "`dependencies` and `neededBy` are never model output. They come from a mechanical "
        "graph over the scanned firmware, or from a human.",
        "",
        "### Provenance",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| pipeline | uadclaw {_code(pipeline_version)} |",
        f"| pipeline commit | {_code(commit_sha)} |",
        f"| base commit | {_code(base_commit) if base_commit.strip() else 'unrecorded'} |",
        f"| branch | {_code(branch) if branch.strip() else 'unrecorded'} |",
        f"| model | {', '.join(_code(model) for model in models)} |",
        "",
        "### Packages",
        "",
        "| package | list | removal | evidence bundle |",
        "| --- | --- | --- | --- |",
    ]
    for item in items:
        digest = _cell(item.bundle_sha256)
        bundle = (
            f"[{_code(item.bundle_sha256)}]({base_url}/{digest})"
            if base_url
            else _code(item.bundle_sha256)
        )
        lines.append(
            f"| {_code(item.package)} | {_cell(item.uad_list)} | {_cell(item.removal)} | {bundle} |"
        )
    lines.append("")
    if base_url:
        lines.append(
            "Each hash above links to the evidence bundle that package was classified from: "
            "the exact facts, graph edges and rule floor the model was shown, and nothing else."
        )
    else:
        lines.append(
            "Each hash above is the sha256 of the evidence bundle that package was classified "
            "from — the exact facts, graph edges and rule floor the model was shown. The "
            "bundles are not hosted anywhere. They are regenerated deterministically from this "
            f"pipeline at commit {_code(commit_sha)}, which serializes them canonically and "
            "writes no timestamp and no run-scoped id into them, so the same corpus produces "
            "byte-identical bundles and the same hashes on any machine."
        )
    return "\n".join(lines) + "\n"
