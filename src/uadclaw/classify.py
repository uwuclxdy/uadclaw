"""The model's proposal and the validator that decides whether it may exist.
Pure: no DB, no network, no clock.

This is the safety-critical half of the classification stage. `removal` reaches real phones
through Canta, AppManager and android-debloat-list as well as uad-ng, so the question this
module answers is not "can this response be made usable" but "is this response allowed at
all". **A permissive response is rejected, never clamped.**

That distinction is the single easiest correctness mistake available here, and it is worth
stating why the tempting version is wrong. `ladder.raise_to_floor` exists and would silently
turn a `Recommended` answer on an `Unsafe` package into `Unsafe`, producing a row that looks
correct. But a model that answered below the floor did not make a rounding error — it read
evidence that says `coreApp="true"` and concluded the package is safe to remove, so its
`description` and its `list` came out of that same misreading and are now attached to a
rating that was quietly corrected. Rejecting surfaces the misread; clamping hides it and
keeps the rest. So the floor is checked with `ladder.is_below_floor` and `raise_to_floor` is
never called on a model response anywhere in this repo.

What the model does and does not own:

- **owns** `description`. That is the real job (§7).
- **owns `list` only where the deterministic rule is undecided.** A model `list` that
  contradicts a decided rule is a rejection, not a value to prefer.
- **may raise `removal`** above the floor and may never land below it.
- **never touches `dependencies` / `neededBy`.** A response carrying either key at all is
  rejected rather than having the key ignored: a model that thinks it is allowed to write
  edges has been prompted wrong, and quietly dropping the field would hide that forever.

`unknown` is a valid, expected outcome rather than a failure. A model that cannot tell what a
package does is supposed to say so, because the alternative — inventing a vendor or a feature
— is exactly what the upstream maintainer's corroboration bar exists to catch.
"""

import enum
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from uadclaw.bundle import EvidenceBundle
from uadclaw.ladder import Removal, RemovalFloor, is_below_floor

# Description length bounds, measured 2026-08-11 against the live 5372-entry
# `data/uad_lists.json`: 5308 entries carry a non-empty description, min 4, max 1513,
# median 80. The bounds are not the extremes — they are chosen to admit the overwhelming
# body of what upstream already accepts while refusing the two shapes a model actually
# produces when it is failing. 20 characters rejects 4.4% of live entries (the one-word
# ones: eight of them are literally "Font") and rejects a model that answered "Vendor app";
# 600 rejects 1.7% (the essays) and catches a model that started narrating. An entry the
# model marks `unknown` is exempt from the lower bound and only from the lower bound.
DESCRIPTION_MIN_CHARS = 20
DESCRIPTION_MAX_CHARS = 600

# Bound on the model's own note to the reviewer. Not shipped upstream; it exists so a
# rejection in triage can be read back.
REASONING_BRIEF_MAX_CHARS = 400

# The literal the model uses for a field it could not determine. Lower-case and exact, so
# "Unknown"/"UNKNOWN" are rejected rather than silently normalised: a normalisation here
# would also swallow "unknown vendor app", which is an invented description.
UNKNOWN = "unknown"

# A space on either side of a newline escape. A literal upstream maintainer request on
# PR #1180 — it indents the next sentence when the list is rendered. Measured: 42 of the
# 5372 live entries violate it, so the file is not self-consistent and new entries still
# have to comply.
_SPACE_NEXT_TO_NEWLINE = re.compile(r" \n|\n ")


class UadList(enum.StrEnum):
    """The upstream `list` categories, spelled as `uad_lists.json` carries them.

    Live distribution, measured 2026-08-11 over 5372 entries: Oem 4242, Misc 433, Aosp 272,
    Carrier 242, Google 183. `Pending` exists in the upstream enum, is used by zero entries
    and has no written definition anywhere in the wiki, so it is deliberately absent here:
    proposing a value nobody has defined is not something this pipeline should be able to do.
    """

    AOSP = "Aosp"
    CARRIER = "Carrier"
    GOOGLE = "Google"
    MISC = "Misc"
    OEM = "Oem"


class Confidence(enum.StrEnum):
    """How sure the model is. Three named levels rather than a 0-1 float on purpose: a
    number invites a threshold, and an LLM's self-reported probability is not calibrated
    enough to carry one. Triage ranks on it; nothing branches on it."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# The fields a proposal may declare unknown. `removal` is absent because it always has a
# value: the floor is one. `list` is absent because the upstream schema already carries the
# answer — `Misc` is the catch-all, 433 live entries use it, and the deterministic rule
# returns undecided precisely so the model can pick. A second spelling of "I don't know" for
# one field is a second thing to keep consistent, and it was accepted alongside a confident
# `list` for exactly as long as it existed.
UNKNOWABLE_FIELDS: frozenset[str] = frozenset({"description"})

# Keys whose mere presence in a response is a rejection. These come from the corpus graph or
# from a human and a wrong edge silently changes a removal rating.
FORBIDDEN_KEYS: tuple[str, ...] = ("dependencies", "neededBy", "needed_by", "labels")


class ClassificationJobParams(BaseModel):
    """A `classification` job's target, validated at creation like a firmware job's is.

    `extra="forbid"` for the same reason: a misspelt key is a 422 at creation rather than a
    job that claims a worker, calls a paid API and only then discovers it was asked for
    something else.

    The default — no `packages`, no `limit`, `reclassify` false — means "every package the
    filter queued that has no classification for its current evidence bundle". That is the
    ordinary run, and it is idempotent: a second run over an unchanged corpus selects
    nothing, because every candidate already has a row against that bundle hash.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Named packages, in place of the queue. Bounded because it arrives from the network and
    # each entry becomes a paid call.
    packages: tuple[str, ...] = Field(default=(), max_length=1000)
    # Cap on how many packages this job calls the API for. `None` means the ceiling in
    # settings, which this can lower and never raise.
    limit: int | None = Field(default=None, ge=1)
    # Re-classify packages that already have a row for the current bundle. Off by default:
    # the model is not reproducible, so a re-run costs money and produces a different answer
    # to the same question, and that has to be asked for rather than happen by accident.
    reclassify: bool = False
    # Which provider this job's model calls go to, as an id in the settings provider table.
    # `None` means the default (`llm_default_provider`). This module is pure, so the id is
    # only CHECKED at the creation seam (`jobs.validate_job_params`), which resolves it
    # against the table, refuses an unknown one before any spend, and stores the RESOLVED id
    # on the row — a job then names its provider even if the default changes later.
    provider: str | None = None


class ClassificationRejected(ValueError):
    """A model response was refused. Carries the field and the reason, because the retry
    loop logs it, the park reason quotes it, and a bare "invalid" cannot be acted on."""

    def __init__(self, field: str, reason: str) -> None:
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")


class ModelProposal(BaseModel):
    """The response shape, before any of the semantic checks.

    `extra="forbid"` because a response carrying a field nobody asked for is a response to a
    different prompt. The forbidden-key check runs first so the common case of that — a model
    helpfully filling in `dependencies` — is refused by name rather than as "extra input".
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str
    list: UadList
    removal: Removal
    confidence: Confidence
    unknown_fields: tuple[str, ...] = ()
    reasoning_brief: str = ""


@dataclass(frozen=True, slots=True)
class Classification:
    """A validated proposal, with the provenance of every field it carries."""

    package: str
    bundle_sha256: str
    description: str
    list: UadList
    removal: Removal
    confidence: Confidence
    unknown_fields: tuple[str, ...]
    reasoning_brief: str
    provenance: Mapping[str, str]

    def as_json(self) -> dict[str, Any]:
        return {
            "package": self.package,
            "bundle_sha256": self.bundle_sha256,
            "description": self.description,
            "list": str(self.list),
            "removal": str(self.removal),
            "confidence": str(self.confidence),
            "unknown_fields": list(self.unknown_fields),
            "reasoning_brief": self.reasoning_brief,
            "provenance": dict(self.provenance),
        }


# --- the deterministic `list` derivation -------------------------------------------------
#
# Design §7: `list` is deterministic from cert issuer plus partition plus prefix, with the
# model only breaking ties in the catch-all category. Measured against the live 5372-entry
# list on 2026-08-11, only ONE of those rules is precise enough to be allowed to overrule a
# model, and saying which is the point of this block.
#
# Every row is measured over the entries the OEM rule has not already claimed (it runs
# first, so that is the population each later rule actually sees). Stating the population
# matters: over ALL entries instead, the AOSP row reads n=620 / 35.8%, a different number for
# a rule that never gets those entries.
#
#   rule                                             n      agrees   predicate
#   OEM vendor token in the package name            3147   98.7%    OEM_NAME_TOKENS below
#   AOSP namespace (`com.android.`/`android.`)       576   38.4%    _AOSP_NAMESPACES below
#   `com.google.` prefix -> Google                   367   47.7%    startswith("com.google.")
#   chipset vendor segment -> Misc                   236   76.7%    EXPLORATORY, see below
#
# The chipset row is the one this module cannot reproduce: there is no chipset rule in
# `derive_list`, because the measurement is what talked us out of writing one. Its 76.7% was
# taken over the segment set {qualcomm, qti, mediatek, mtk, unisoc, spreadtrum, sprd}, which
# lives here in prose and nowhere in code. Recorded rather than dropped because "we looked at
# this and it did not clear the bar" is the thing a later reader most needs, and a row with
# no predicate behind it reads as stale comment unless it says so.
#
# The bottom three are not close. The AOSP namespace is mostly Oem upstream because every
# OEM ships packages in it, and `com.google.*` splits between Google's own apps and Google's
# builds of AOSP mainline components (`documentsui`, `providers.media.module`,
# `networkstack.tethering`) that upstream calls Aosp. A prefix cannot tell those apart; the
# signing organization can, which is why the issuer is the second decider and the only one
# that can decide `Aosp`.
#
# Everything else is UNDECIDED, and undecided means the model's answer stands. `Carrier` in
# particular is never derived: which of Verizon, Orange or Telstra a package belongs to is a
# business fact no manifest carries.

# Matched as whole dotted SEGMENTS of a lower-cased package name, never as substrings: `mi`
# and `sec` as substrings hit half the corpus, and this table decides a value that can reject
# a model answer.
OEM_NAME_TOKENS: frozenset[str] = frozenset(
    {
        "samsung", "sec", "huawei", "hihonor", "honor", "motorola", "moto", "lge",
        "oneplus", "miui", "xiaomi", "redmi", "poco", "vivo", "oplus", "oppo", "coloros",
        "realme", "asus", "nubia", "zte", "transsion", "tecno", "infinix", "itel",
        "sonymobile", "sonyericsson", "tct", "alcatel", "zui", "lenovo", "meizu", "nothing",
        "evenwell", "blackview", "wingtech", "longcheer", "tinno", "hmd", "nokia",
    }
)  # fmt: skip

# Lower-cased substrings of the certificate issuer's Organization field. Matched loosely
# because the field is free text out of a vendor's own CA (`Organization: Xiaomi Inc.`,
# `Organization: Motorola Mobility LLC`) and the organization is what survives APK signature
# v3 key rotation when the common name does not.
OEM_ISSUER_ORGS: tuple[str, ...] = (
    "samsung", "huawei", "honor", "motorola", "lg electronics", "oneplus", "xiaomi",
    "beijing xiaomi", "vivo", "oppo", "guangdong", "realme", "asus", "zte", "transsion",
    "sony", "tcl", "lenovo", "meizu", "nothing technology", "nokia", "hmd",
)  # fmt: skip

# The AOSP test key: `Email Address: android@android.com, Organization: Android`. An APK
# signed with it is an unmodified AOSP build by definition, which is the one thing that makes
# `Aosp` safe to decide rather than to guess.
_AOSP_TEST_KEY_MARKERS: tuple[str, ...] = ("android@android.com",)

_AOSP_NAMESPACES: tuple[str, ...] = ("com.android.", "android.")


@dataclass(frozen=True, slots=True)
class ListDerivation:
    """What the deterministic rule concluded, and why.

    `value is None` means undecided — the model's answer is accepted as-is. It is a separate
    state from `Misc`: `Misc` is a decided catch-all somebody could argue with, undecided is
    "this rule has nothing to say", and collapsing them would let an undecided package reject
    a perfectly good model answer.
    """

    value: UadList | None
    rule: str
    detail: str

    @property
    def decided(self) -> bool:
        return self.value is not None


def _issuer_organization(cert_issuer: str | None) -> str:
    """The issuer string, lower-cased. Kept whole rather than split on `Organization:`
    because `human_friendly` orders its fields differently per certificate and a parse that
    misses would silently return "no organization" rather than raising."""
    return (cert_issuer or "").lower()


def derive_list(
    package: str, *, cert_issuer: str | None = None, partitions: Sequence[str] = ()
) -> ListDerivation:
    """The category the deterministic rules decide, or undecided.

    Order matters: the OEM rules run before the AOSP one, because an OEM's overlay of an
    AOSP package (`com.android.settings.overlay.miui`) lives in the AOSP namespace and is
    upstream's `Oem`, and the reverse order labels 316 live entries wrong.
    """
    issuer = _issuer_organization(cert_issuer)
    segments = set(package.lower().split("."))

    matched = sorted(segments & OEM_NAME_TOKENS)
    if matched:
        return ListDerivation(
            value=UadList.OEM,
            rule="rule:oem_name_token",
            detail=f"package name carries the vendor segment(s) {', '.join(matched)}",
        )
    for org in OEM_ISSUER_ORGS:
        if org in issuer:
            return ListDerivation(
                value=UadList.OEM,
                rule="rule:oem_cert_issuer",
                detail=f"signing organization names the vendor ({org})",
            )
    if any(marker in issuer for marker in _AOSP_TEST_KEY_MARKERS) and package.startswith(
        _AOSP_NAMESPACES
    ):
        return ListDerivation(
            value=UadList.AOSP,
            rule="rule:aosp_platform_key",
            detail="signed with the AOSP test key and in the AOSP namespace",
        )
    # Undecided. The partition travels into the bundle as evidence for the model rather than
    # as a decider: the closest thing to a partition rule is "chipset-vendor software is
    # Misc", and that agrees with upstream on 76.7% of the entries it would claim (see the
    # table above). Three wrong answers in thirteen is a useful prior and nowhere near a rule
    # allowed to REJECT a model that disagrees, which is the only thing deciding a value
    # means here.
    return ListDerivation(
        value=None,
        rule="llm:list",
        detail=(
            "no deterministic rule decided this package "
            f"(partitions: {', '.join(sorted(partitions)) or 'none'})"
        ),
    )


# --- validation ---------------------------------------------------------------------------


def _check_forbidden_keys(payload: Mapping[str, Any]) -> None:
    for key in FORBIDDEN_KEYS:
        if key in payload:
            raise ClassificationRejected(
                key,
                f"the response carries {key!r}. `dependencies` and `neededBy` come from the "
                "corpus graph or from a human and are never model output — a wrong edge "
                "silently changes a removal rating — so a response that writes one is "
                "refused rather than having the field dropped.",
            )


def _check_description(description: str, *, unknown_fields: Sequence[str]) -> None:
    if "description" in unknown_fields:
        if description != UNKNOWN:
            raise ClassificationRejected(
                "description",
                f"declared unknown but reads {description[:80]!r}. A field listed in "
                f"unknown_fields must be exactly {UNKNOWN!r}, so 'unknown' cannot be a "
                "hedge attached to an invented description.",
            )
        return
    stripped = description.strip()
    if description != stripped:
        raise ClassificationRejected(
            "description",
            "has leading or trailing whitespace, which ships into the upstream file as-is",
        )
    if len(description) < DESCRIPTION_MIN_CHARS:
        raise ClassificationRejected(
            "description",
            f"is {len(description)} characters, under the {DESCRIPTION_MIN_CHARS}-character "
            "floor measured against the live upstream list. Say what the package does, or "
            f"declare it unknown in unknown_fields with the description {UNKNOWN!r}.",
        )
    if len(description) > DESCRIPTION_MAX_CHARS:
        raise ClassificationRejected(
            "description",
            f"is {len(description)} characters, over the {DESCRIPTION_MAX_CHARS}-character "
            "ceiling; 98.3% of live upstream entries are shorter",
        )
    if _SPACE_NEXT_TO_NEWLINE.search(description):
        raise ClassificationRejected(
            "description",
            "puts a space next to a newline escape, which indents the following sentence "
            "when the list is rendered. A literal upstream maintainer request on PR #1180.",
        )


def _check_unknown_fields(unknown_fields: Sequence[str]) -> None:
    unexpected = sorted(set(unknown_fields) - UNKNOWABLE_FIELDS)
    if unexpected:
        raise ClassificationRejected(
            "unknown_fields",
            f"names {', '.join(unexpected)}, which is not a field that may be unknown. "
            f"Only {', '.join(sorted(UNKNOWABLE_FIELDS))} can be, because every other "
            "emitted field is derived rather than judged.",
        )


def _check_removal(removal: Removal, floor: RemovalFloor) -> None:
    if is_below_floor(removal, floor):
        raise ClassificationRejected(
            "removal",
            f"answered {removal} for a package whose computed floor is {floor.floor} "
            f"(set by {floor.rule}: "
            f"{floor.fired[0].detail if floor.fired else 'no rule'}). The floor is a lower "
            "bound the LLM may raise and never lower, and this is REJECTED rather than "
            "raised to the floor: an answer below it came from misreading the same evidence "
            "the description was written from, so correcting the number would keep the "
            "misreading and hide it.",
        )


def _check_list(proposed: UadList, derivation: ListDerivation) -> None:
    if derivation.decided and proposed is not derivation.value:
        raise ClassificationRejected(
            "list",
            f"answered {proposed} where the deterministic rule decided "
            f"{derivation.value} ({derivation.rule}: {derivation.detail}). The LLM only "
            "breaks ties the rules leave open.",
        )


def validate_response(
    payload: Mapping[str, Any],
    *,
    bundle: EvidenceBundle,
    floor: RemovalFloor,
    derivation: ListDerivation,
    model: str,
) -> Classification:
    """Turn one raw model response into a `Classification`, or refuse it.

    Every refusal is a `ClassificationRejected` naming the field, because the caller retries
    a bounded number of times and then parks the package with the reason attached, and a
    park reason that does not say which field was wrong is a park nobody can clear.
    """
    _check_forbidden_keys(payload)
    try:
        proposal = ModelProposal.model_validate(payload)
    except ValidationError as exc:
        first = exc.errors()[0] if exc.errors() else {}
        field = ".".join(str(part) for part in first.get("loc", ())) or "response"
        raise ClassificationRejected(
            field,
            f"{first.get('msg', 'did not match the required shape')} ({exc.error_count()} "
            f"error(s) in total)",
        ) from exc

    _check_unknown_fields(proposal.unknown_fields)
    _check_description(proposal.description, unknown_fields=proposal.unknown_fields)
    _check_removal(proposal.removal, floor)
    _check_list(proposal.list, derivation)
    if len(proposal.reasoning_brief) > REASONING_BRIEF_MAX_CHARS:
        raise ClassificationRejected(
            "reasoning_brief",
            f"is {len(proposal.reasoning_brief)} characters, over the "
            f"{REASONING_BRIEF_MAX_CHARS}-character ceiling",
        )

    return Classification(
        package=bundle.package,
        bundle_sha256=bundle.sha256,
        description=proposal.description,
        list=proposal.list,
        removal=proposal.removal,
        confidence=proposal.confidence,
        unknown_fields=tuple(sorted(proposal.unknown_fields)),
        reasoning_brief=proposal.reasoning_brief,
        provenance=provenance_for(derivation, model=model),
    )


def provenance_for(derivation: ListDerivation, *, model: str) -> dict[str, str]:
    """Which authority owns each emitted field, per the repo rule that every emitted field
    carries `rule:`, `graph:`, `llm:` or `human:`.

    Read by the store on a re-run: a field tagged `human:` survives a re-classification
    untouched, which is what makes a bad model run re-runnable without walking over an
    edit somebody made in triage.
    """
    return {
        "description": f"llm:{model}",
        "list": derivation.rule if derivation.decided else f"llm:{model}",
        "removal": f"llm:{model}",
        "confidence": f"llm:{model}",
        "reasoning_brief": f"llm:{model}",
        # Stated rather than omitted: these ARE emitted fields upstream, and naming their
        # authority here is what makes "the model never wrote these" auditable per row.
        "dependencies": "graph:corpus",
        "neededBy": "graph:corpus",
        "floor": "rule:ladder",
    }


# --- the prompt ----------------------------------------------------------------------------
#
# The system message is byte-identical on every call and carries the whole rubric, and the
# per-package evidence goes in the user message. That split is not cosmetic: DeepSeek's cache
# persists a detected common prefix as its own unit, and a hit costs $0.0028/1M against
# $0.14/1M for a miss — 50x — so every token that can live in the shared half should.

_EXAMPLE_RESPONSE = json.dumps(
    {
        "description": (
            "Carrier configuration provider. Supplies network operator settings to the "
            "telephony stack; removing it can leave mobile data unconfigured."
        ),
        "list": "Aosp",
        "removal": "Expert",
        "confidence": "medium",
        "unknown_fields": [],
        "reasoning_brief": "Declares carrier config provider authorities and ships in priv-app.",
    },
    indent=2,
    sort_keys=True,
)

SYSTEM_PROMPT = f"""\
You classify preinstalled Android packages for the Universal Android Debloater list. You are \
given one package's evidence, extracted from real firmware, and you answer with exactly one \
json object and nothing else.

Answer with this shape:

{_EXAMPLE_RESPONSE}

Fields:

- description: what the package does and what removing it breaks, in plain English. \
{DESCRIPTION_MIN_CHARS}-{DESCRIPTION_MAX_CHARS} characters. Never put a space next to a \
newline escape. Do not name the package again; the reader already has it.
- list: one of Aosp, Carrier, Google, Misc, Oem. Aosp = an unmodified Android Open Source \
component. Google = Google's own app or service. Oem = the device manufacturer's. Carrier = \
a mobile network operator's. Misc = anything else, including chipset-vendor software.
- removal: one of Recommended, Advanced, Expert, Unsafe, and NEVER less dangerous than the \
floor the evidence gives you. Recommended = safe to uninstall. Advanced = breaks obscure or \
minor functionality, overlays, or apps with a better alternative. Expert = breaks widespread \
or important functionality but nothing vital to the OS. Unsafe = can break vital parts of \
the OS, or is illegal to remove in some countries.
- confidence: low, medium or high.
- unknown_fields: the names of fields you could not determine, from ["description"]. Do not \
put "list" here — answer "Misc" instead, which is what it is for.
- reasoning_brief: one or two sentences of evidence for your answer, under \
{REASONING_BRIEF_MAX_CHARS} characters.

Rules:

1. The evidence carries a computed removal floor. It is a lower bound derived from manifest \
facts. You may answer a MORE dangerous tier than the floor. An answer below the floor is \
rejected outright, not corrected.
2. Answer "{UNKNOWN}" rather than inventing anything. If you cannot tell what a package does \
from the evidence, put "description" in unknown_fields and set description to exactly \
"{UNKNOWN}". Unknown is a normal, expected, useful answer. Guessing a vendor, a feature or a \
product name that is not in the evidence is the worst thing you can do here, because a human \
reviewer has to check every claim you make against an independent source.
3. Never output "dependencies", "neededBy" or "labels". Those come from a mechanical graph, \
not from you, and a response containing them is discarded.
4. Use only the evidence given. It includes similar existing entries as style anchors: match \
their register and length, do not copy their content.
"""


def user_prompt(bundle: EvidenceBundle) -> str:
    """The per-package half.

    The evidence is the same PAYLOAD the hash is taken over — indented here for the model
    and compact there for the digest, so it is the same object and deliberately not the same
    bytes. Nothing is added, removed or reordered on the way in, which is what makes
    `bundle_sha256` an honest record of what was asked.
    """
    return (
        f"Classify this package. Answer with one json object.\n\nEVIDENCE:\n"
        f"{json.dumps(bundle.payload, indent=2, sort_keys=True, ensure_ascii=False)}\n"
    )
