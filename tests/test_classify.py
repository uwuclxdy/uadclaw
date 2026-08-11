"""The validator, which is the only thing standing between a model and the removal rating
that reaches real phones through Canta, AppManager and android-debloat-list.

Every rejection test asserts the SPECIFIC field and reason, never that "something raised".
This repo has already been bitten by the alternative: a mutation that inverted a guard shared
across several fields made the field under test pass while two unrelated fixture fields newly
failed, and a bare `pytest.raises(ValidationError)` still caught an exception and read green.

The one behaviour worth stating in prose because no single assertion carries it: a permissive
answer is REJECTED, not clamped. `ladder.raise_to_floor` exists, would turn a `Recommended`
answer on an `Unsafe` package into a correct-looking row, and is deliberately never called
here — a model that answered below the floor misread the same evidence its description came
from, and correcting only the number keeps the misreading and hides it.
"""

import json

import pytest

from uadclaw.bundle import EvidenceBundle, PackageIdentity, build_bundle
from uadclaw.classify import (
    DESCRIPTION_MAX_CHARS,
    DESCRIPTION_MIN_CHARS,
    SYSTEM_PROMPT,
    UNKNOWN,
    ClassificationJobParams,
    ClassificationRejected,
    Confidence,
    UadList,
    derive_list,
    provenance_for,
    user_prompt,
    validate_response,
)
from uadclaw.corpus import CorpusPackage
from uadclaw.ladder import Removal, compute_floors

CORPUS = [
    CorpusPackage(package="com.android.settings", core_app=True, partitions=("system",)),
    CorpusPackage(package="com.example.plain", partitions=("product",)),
    CorpusPackage(package="com.example.privileged", priv_app=True, partitions=("product",)),
]
FLOORS = compute_floors(CORPUS)

GOOD_DESCRIPTION = "Vendor notes application. Removing it loses locally stored notes."


def bundle_for(name: str, **identity) -> EvidenceBundle:
    item = next(entry for entry in CORPUS if entry.package == name)
    return build_bundle(item, floor=FLOORS[name], identity=PackageIdentity(**identity))


def response(**overrides) -> dict:
    payload = {
        "description": GOOD_DESCRIPTION,
        "list": "Misc",
        "removal": "Recommended",
        "confidence": "medium",
        "unknown_fields": [],
        "reasoning_brief": "Declares no privileged surface.",
    }
    payload.update(overrides)
    return payload


def validate(name: str = "com.example.plain", *, derivation=None, **overrides):
    bundle = bundle_for(name)
    return validate_response(
        response(**overrides),
        bundle=bundle,
        floor=FLOORS[name],
        derivation=derivation or derive_list(name),
        model="deepseek-v4-flash",
    )


def rejection(name: str = "com.example.plain", *, derivation=None, **overrides):
    with pytest.raises(ClassificationRejected) as caught:
        validate(name, derivation=derivation, **overrides)
    return caught.value


# --- the floor -----------------------------------------------------------------------------


def test_a_response_below_the_floor_is_rejected_and_never_clamped():
    """`com.android.settings` is `coreApp="true"`, so its floor is Unsafe. A `Recommended`
    answer must not come back as a valid `Classification` carrying `Unsafe`."""
    error = rejection("com.android.settings", removal="Recommended")
    assert error.field == "removal"
    assert "Recommended" in error.reason
    assert "Unsafe" in error.reason
    assert "REJECTED rather than raised" in error.reason


@pytest.mark.parametrize("proposed", ["Recommended", "Advanced", "Expert"])
def test_every_tier_under_the_floor_is_rejected(proposed):
    assert rejection("com.android.settings", removal=proposed).field == "removal"


def test_a_response_at_the_floor_is_accepted():
    result = validate("com.android.settings", removal="Unsafe")
    assert result.removal is Removal.UNSAFE


def test_a_response_above_the_floor_is_accepted():
    """The floor is a lower bound. `com.example.privileged` sits in priv-app, floor Advanced;
    a model that says Expert is allowed to be more careful than the rules."""
    assert FLOORS["com.example.privileged"].floor is Removal.ADVANCED
    result = validate("com.example.privileged", removal="Expert")
    assert result.removal is Removal.EXPERT


def test_a_removal_outside_the_enum_is_rejected_by_name():
    assert rejection(removal="Safe").field == "removal"


# --- dependencies are never model output ---------------------------------------------------


@pytest.mark.parametrize("key", ["dependencies", "neededBy", "needed_by", "labels"])
def test_a_response_carrying_a_graph_owned_key_is_rejected_rather_than_ignored(key):
    error = rejection(**{key: ["com.other.package"]})
    assert error.field == key
    assert "corpus graph" in error.reason


def test_an_empty_graph_owned_key_is_still_a_rejection():
    """An empty list is the shape a model produces when it has been told the field exists and
    has nothing to put in it. Accepting it teaches the prompt nothing and leaves the door
    open for the non-empty case."""
    assert rejection(dependencies=[]).field == "dependencies"


# --- the list derivation -------------------------------------------------------------------


def test_the_oem_name_token_rule_decides_and_contradicting_it_is_a_rejection():
    derivation = derive_list("com.miui.securitycenter")
    assert derivation.value is UadList.OEM
    assert derivation.rule == "rule:oem_name_token"
    error = rejection(derivation=derivation, list="Google")
    assert error.field == "list"
    assert "rule:oem_name_token" in error.reason


def test_the_cert_issuer_decides_oem_when_the_name_does_not():
    derivation = derive_list(
        "com.qti.diagservices", cert_issuer="Common Name: xyz, Organization: Motorola Mobility LLC"
    )
    assert derivation.value is UadList.OEM
    assert derivation.rule == "rule:oem_cert_issuer"


def test_the_aosp_test_key_decides_aosp():
    derivation = derive_list(
        "com.android.settings",
        cert_issuer="Email Address: android@android.com, Organization: Android",
    )
    assert derivation.value is UadList.AOSP


def test_the_aosp_test_key_outside_the_aosp_namespace_is_undecided():
    """The emulator signs everything with the test key, including Google's own packages. The
    namespace is what makes the key mean "unmodified AOSP component"."""
    derivation = derive_list(
        "com.google.android.gms",
        cert_issuer="Email Address: android@android.com, Organization: Android",
    )
    assert derivation.value is None


def test_an_undecided_derivation_lets_the_model_answer():
    """Measured against the live list, a `com.google.` prefix agrees with upstream's `Google`
    only 47.7% of the time and `com.android.` with `Aosp` only 39.2%, so neither decides.
    Undecided means the model's answer stands."""
    derivation = derive_list("com.google.android.apps.messaging")
    assert derivation.value is None
    result = validate(derivation=derivation, list="Google")
    assert result.list is UadList.GOOGLE


def test_carrier_is_never_derived():
    """Which network operator a package belongs to is a business fact no manifest carries."""
    for name in ("com.vzw.hss.myverizon", "com.tmobile.pr.mytmobile", "com.orange.mytelekom"):
        assert derive_list(name).value is not UadList.CARRIER


def test_a_list_outside_the_enum_is_rejected_by_name():
    assert rejection(list="Vendor").field == "list"


# --- the description ------------------------------------------------------------------------


def test_a_space_next_to_a_newline_escape_is_rejected():
    """A literal upstream maintainer request on PR #1180: it indents the next sentence."""
    error = rejection(description="Notes app for the vendor. \nRemoving it loses the notes.")
    assert error.field == "description"
    assert "PR #1180" in error.reason
    assert rejection(description="Notes app for the vendor.\n Removing it loses notes.").field == (
        "description"
    )


def test_a_newline_without_an_adjacent_space_is_accepted():
    """3368 of the 5372 live entries carry a newline. The rule is about the space."""
    text = "Notes app for the vendor.\nRemoving it loses locally stored notes."
    assert validate(description=text).description == text


def test_a_description_under_the_measured_floor_is_rejected():
    error = rejection(description="Vendor app")
    assert error.field == "description"
    assert str(DESCRIPTION_MIN_CHARS) in error.reason


def test_a_description_over_the_measured_ceiling_is_rejected():
    error = rejection(description="a" * (DESCRIPTION_MAX_CHARS + 1))
    assert error.field == "description"
    assert str(DESCRIPTION_MAX_CHARS) in error.reason


def test_a_description_at_each_bound_is_accepted():
    assert len(validate(description="a" * DESCRIPTION_MIN_CHARS).description) == (
        DESCRIPTION_MIN_CHARS
    )
    assert len(validate(description="a" * DESCRIPTION_MAX_CHARS).description) == (
        DESCRIPTION_MAX_CHARS
    )


def test_surrounding_whitespace_is_rejected_rather_than_stripped():
    """It ships into the upstream file as written."""
    assert rejection(description=f"  {GOOD_DESCRIPTION}  ").field == "description"


# --- unknown is a valid outcome --------------------------------------------------------------


def test_unknown_is_accepted_rather_than_treated_as_a_failure():
    result = validate(description=UNKNOWN, unknown_fields=["description"])
    assert result.description == UNKNOWN
    assert result.unknown_fields == ("description",)


def test_unknown_cannot_be_a_hedge_attached_to_an_invented_description():
    error = rejection(
        description="Probably a Samsung telephony service of some kind.",
        unknown_fields=["description"],
    )
    assert error.field == "description"
    assert "must be exactly" in error.reason


def test_a_field_that_cannot_be_unknown_is_rejected():
    error = rejection(unknown_fields=["removal"])
    assert error.field == "unknown_fields"
    assert "removal" in error.reason


# --- shape ------------------------------------------------------------------------------------


def test_a_missing_required_field_names_that_field():
    payload = response()
    del payload["confidence"]
    with pytest.raises(ClassificationRejected) as caught:
        validate_response(
            payload,
            bundle=bundle_for("com.example.plain"),
            floor=FLOORS["com.example.plain"],
            derivation=derive_list("com.example.plain"),
            model="deepseek-v4-flash",
        )
    assert caught.value.field == "confidence"


def test_an_unasked_for_field_is_rejected():
    assert rejection(vendor="Samsung").field == "vendor"


def test_an_overlong_reasoning_brief_is_rejected():
    assert rejection(reasoning_brief="x" * 5000).field == "reasoning_brief"


def test_a_bad_confidence_is_rejected_by_name():
    assert rejection(confidence="very").field == "confidence"


def test_confidence_survives_onto_the_classification():
    assert validate(confidence="high").confidence is Confidence.HIGH


# --- provenance --------------------------------------------------------------------------------


def test_every_emitted_field_carries_a_provenance_tag():
    result = validate()
    for field in ("description", "list", "removal", "dependencies", "neededBy", "floor"):
        assert field in result.provenance
        assert result.provenance[field].split(":")[0] in {"rule", "graph", "llm", "human"}


def test_the_graph_owned_fields_are_never_tagged_llm():
    provenance = provenance_for(derive_list("com.example.plain"), model="deepseek-v4-flash")
    assert provenance["dependencies"] == "graph:corpus"
    assert provenance["neededBy"] == "graph:corpus"
    assert provenance["floor"] == "rule:ladder"


def test_a_rule_decided_list_is_tagged_rule_and_an_undecided_one_llm():
    decided = provenance_for(derive_list("com.miui.thing"), model="m")
    assert decided["list"] == "rule:oem_name_token"
    undecided = provenance_for(derive_list("com.example.plain"), model="m")
    assert undecided["list"] == "llm:m"


# --- the prompt --------------------------------------------------------------------------------


def test_the_system_prompt_honours_both_json_mode_clauses():
    """The API reference: without an explicit JSON instruction the model "may generate an
    unending stream of whitespace"; the guide separately requires an example."""
    assert "json" in SYSTEM_PROMPT.lower()
    assert "{" in SYSTEM_PROMPT and "}" in SYSTEM_PROMPT


def test_the_system_prompt_example_is_a_parseable_object_of_the_right_shape():
    start = SYSTEM_PROMPT.index("{")
    depth = 0
    for index in range(start, len(SYSTEM_PROMPT)):
        depth += SYSTEM_PROMPT[index] == "{"
        depth -= SYSTEM_PROMPT[index] == "}"
        if depth == 0:
            example = json.loads(SYSTEM_PROMPT[start : index + 1])
            break
    assert set(example) == {
        "description",
        "list",
        "removal",
        "confidence",
        "unknown_fields",
        "reasoning_brief",
    }


def test_the_prompt_tells_the_model_unknown_is_allowed_and_edges_are_not():
    assert UNKNOWN in SYSTEM_PROMPT
    assert "dependencies" in SYSTEM_PROMPT
    assert "below the floor is rejected" in SYSTEM_PROMPT


def test_the_user_prompt_carries_the_exact_bytes_the_hash_attests():
    bundle = bundle_for("com.example.plain")
    text = user_prompt(bundle)
    start = text.index("{")
    assert json.loads(text[start:]) == json.loads(bundle.canonical_json())


# --- job params ---------------------------------------------------------------------------------


def test_classification_params_default_to_the_whole_queue():
    params = ClassificationJobParams()
    assert params.packages == ()
    assert params.limit is None
    assert params.reclassify is False


def test_a_misspelt_param_is_refused_at_the_boundary():
    with pytest.raises(ValueError, match="reclassifiy"):
        ClassificationJobParams(reclassifiy=True)


def test_a_zero_limit_is_refused():
    with pytest.raises(ValueError, match="limit"):
        ClassificationJobParams(limit=0)
