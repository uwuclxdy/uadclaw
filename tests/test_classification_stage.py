"""The `llm` stage against a real database, with the API mocked at the transport.

The default suite must never touch the network and never need a key, so every response here
comes from an `httpx.MockTransport` handed to the client the stage builds. What only a
database can prove is the wiring, and the wiring has three shapes that are easy to get wrong
and silent when they are:

- a classification run must write `package_classification` and leave `package_analysis`
  byte-identical, because that table's contract is that nothing on it is ever model output;
- a package the model cannot answer must be PARKED as a row rather than raise, or one bad
  package costs the other 47 and nothing records why;
- a re-run must replace only `llm:`-provenance fields, so a bad model run is re-runnable
  without walking over a human's edit.
"""

import json
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import select

from uadclaw import jobs as jobs_module
from uadclaw import stages as stages_module
from uadclaw.classify import UNKNOWN
from uadclaw.classifystore import ClassificationStoreError, select_candidates
from uadclaw.corpusstore import load_config_inputs, require_corpus
from uadclaw.deepseek import DeepSeekBalanceError, DeepSeekClient
from uadclaw.facts import ApkFacts
from uadclaw.factstore import record_device_scan, store_device_facts
from uadclaw.ladder import Removal, compute_floors, danger_rank
from uadclaw.models import JobKind, PackageAnalysis, PackageClassification
from uadclaw.settings import get_settings
from uadclaw.stages import llm_stage
from uadclaw.worker import StageContext

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
PIXEL = "pixel:oriole"
BUILD = "cp2a.260705.006.a1"

UPSTREAM_JSON = json.dumps(
    {
        "com.android.settings": {
            "list": "Aosp",
            "removal": "Unsafe",
            "description": "System settings. Removing it makes the device unconfigurable.",
        },
        "com.example.notes.legacy": {
            "list": "Oem",
            "removal": "Recommended",
            "description": "Older notes app, superseded by the current one.",
        },
    }
)

# `Advanced` rather than `Recommended` so one fixture answer clears both candidates' floors:
# `com.example.launcher` sits in priv-app and is pinned at Advanced, `com.example.notes` has
# no rule at all and floors at Recommended. A shared fixture below one of the floors would
# park half the corpus in every test and make each of them measure the wrong thing.
GOOD = {
    "description": "Vendor notes application. Removing it loses locally stored notes.",
    "list": "Misc",
    "removal": "Advanced",
    "confidence": "medium",
    "unknown_fields": [],
    "reasoning_brief": "No privileged surface, no platform integration.",
}


def make_facts(package: str, **overrides) -> ApkFacts:
    values: dict[str, object] = {
        "package": package,
        "label": "Example",
        "label_unresolved": False,
        "version_code": 1,
        "partition": "product",
        "device_path": f"/product/app/{package}/{package}.apk",
        "priv_app": False,
        "sha256": "0" * 64,
        "cert_issuer": "Organization: Example Corp",
        "cert_subject": "Organization: Example Corp",
        "core_app": False,
        "shared_user_id": None,
        "persistent": False,
        "has_code": True,
        "overlay_target": None,
        "overlay_static": False,
        "overlay_priority": None,
    }
    values.update(overrides)
    return ApkFacts(**values)  # type: ignore[arg-type]


CORPUS = [
    make_facts("com.android.settings", core_app=True, partition="system"),
    make_facts("com.example.notes"),
    make_facts("com.example.launcher", priv_app=True),
]

# Four queued candidates rather than the default two, for the containment tests only:
# "one package's failure must not reach the others" is thin proof with a single other, and
# `com.example.alpha` sorts first, so a failure on it is the one the siblings are least
# likely to survive by having already finished.
WIDE_CORPUS = [*CORPUS, make_facts("com.example.alpha"), make_facts("com.example.zeta")]
WIDE_CANDIDATES = [
    "com.example.alpha",
    "com.example.launcher",
    "com.example.notes",
    "com.example.zeta",
]


def envelope(payload, *, finish_reason="stop"):
    content = payload if isinstance(payload, str) else json.dumps(payload)
    return {
        "id": "chat-1",
        "model": "deepseek-v4-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": 1500,
            "completion_tokens": 120,
            "total_tokens": 1620,
            "prompt_cache_hit_tokens": 1024,
            "prompt_cache_miss_tokens": 476,
            "completion_tokens_details": {"reasoning_tokens": 60},
        },
    }


@pytest.fixture
def fake_api(monkeypatch):
    """Replace the transport the stage's client is built with, never the client itself: the
    retry loop, the semaphore, the budget check and the request body all still run."""
    state = {"responses": [], "requests": []}

    def handler(request: httpx.Request) -> httpx.Response:
        state["requests"].append(json.loads(request.content))
        item = state["responses"]
        response = item.pop(0) if len(item) > 1 else item[0]
        return response(request) if callable(response) else response

    original = DeepSeekClient.from_settings

    def patched(cls_settings, *, transport=None):
        return original(cls_settings, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(DeepSeekClient, "from_settings", patched)
    return state


@pytest.fixture
def classification_env(monkeypatch, tmp_path):
    path = tmp_path / "uad_lists.json"
    path.write_text(UPSTREAM_JSON, encoding="utf-8")
    monkeypatch.setenv("UPSTREAM_LIST_PATH", str(path))
    monkeypatch.setenv("DEEPSEEK_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("DEEPSEEK_RETRY_BACKOFF_SECONDS", "0.001")
    return path


async def seed(session_factory, corpus: Sequence[ApkFacts] = CORPUS) -> uuid.UUID:
    """A corpus that has been through extract_facts, corpus_graph, filter and rule_ladder."""
    async with session_factory() as session, session.begin():
        firmware = await jobs_module.create_job(
            session,
            kind=JobKind.FIRMWARE_ANALYSIS.value,
            params={"driver": "pixel", "device": "oriole"},
        )
        await record_device_scan(
            session,
            job_id=firmware.id,
            device_key=PIXEL,
            build=BUILD,
            scanned_at=NOW,
            apk_total=len(corpus),
            parsed_ok=len(corpus),
            failures=[],
        )
        await store_device_facts(
            session, device_key=PIXEL, build=BUILD, facts=corpus, observed_at=NOW
        )
        for item in corpus:
            session.add(
                PackageAnalysis(
                    package=item.package,
                    updated_at=NOW,
                    upstream_present=item.package == "com.android.settings",
                    queued=item.package != "com.android.settings",
                    filter_verdict="queued",
                )
            )
    async with session_factory() as session, session.begin():
        job = await jobs_module.create_job(session, kind=JobKind.CLASSIFICATION.value)
        return job.id


def requested_package(request: httpx.Request) -> str:
    """The package a mocked request is asking about, parsed out of the prompt the way the
    model reads it. A substring match on the raw body would also hit a package NAMED in
    another one's evidence, so the branch would fire for the wrong request."""
    content = json.loads(request.content)["messages"][1]["content"]
    return json.loads(content.split("EVIDENCE:\n", 1)[1])["package"]


def context(job_id, session_factory) -> StageContext:
    return StageContext(job_id=job_id, attempt=1, scratch_dir=None, session_factory=session_factory)


async def rows(session_factory) -> dict[str, PackageClassification]:
    async with session_factory() as session:
        result = await session.execute(select(PackageClassification))
        return {row.package: row for row in result.scalars()}


# --- the happy path --------------------------------------------------------------------------


async def test_a_classification_run_writes_a_proposal_per_queued_package(
    db_env, classification_env, fake_api, db_session_factory
):
    fake_api["responses"] = [httpx.Response(200, json=envelope(GOOD))]
    job_id = await seed(db_session_factory)

    await llm_stage(context(job_id, db_session_factory))

    stored = await rows(db_session_factory)
    # `com.android.settings` is already upstream, so the filter never queued it.
    assert sorted(stored) == ["com.example.launcher", "com.example.notes"]
    row = stored["com.example.notes"]
    assert row.description == GOOD["description"]
    assert row.uad_list == "Misc"
    assert row.removal == "Advanced"
    assert row.parked is False
    assert row.attempts == 1
    assert len(row.bundle_sha256) == 64


async def test_the_raw_usage_envelope_is_persisted_per_row(
    db_env, classification_env, fake_api, db_session_factory
):
    """The cost and cache measurement reads these off real rows rather than a throwaway
    script, which is the only way the figure describes the prompt that actually shipped."""
    fake_api["responses"] = [httpx.Response(200, json=envelope(GOOD))]
    job_id = await seed(db_session_factory)

    await llm_stage(context(job_id, db_session_factory))

    usage = (await rows(db_session_factory))["com.example.notes"].usage
    assert usage["prompt_cache_hit_tokens"] == 1024
    assert usage["prompt_cache_miss_tokens"] == 476
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 60


async def test_the_prompt_carries_the_bundle_whose_hash_is_stored(
    db_env, classification_env, fake_api, db_session_factory
):
    fake_api["responses"] = [httpx.Response(200, json=envelope(GOOD))]
    job_id = await seed(db_session_factory)

    await llm_stage(context(job_id, db_session_factory))

    sent = {
        json.loads(request["messages"][1]["content"].split("EVIDENCE:\n", 1)[1])[
            "package"
        ]: request["messages"][1]["content"]
        for request in fake_api["requests"]
    }
    assert set(sent) == {"com.example.notes", "com.example.launcher"}
    assert "floor" in sent["com.example.launcher"]
    # The floor the model is shown is the one the ladder computed: priv-app is Advanced.
    evidence = json.loads(sent["com.example.launcher"].split("EVIDENCE:\n", 1)[1])
    assert evidence["floor"]["floor"] == "Advanced"


# --- nothing model-owned reaches package_analysis -----------------------------------------------


async def test_a_classification_run_leaves_package_analysis_untouched(
    db_env, classification_env, fake_api, db_session_factory
):
    """`package_analysis` carries the graph edges and the removal floor, and its contract is
    that nothing on it is ever model output. The two tables share no column, so this is a
    structural property — but it is the property, so it is asserted."""
    fake_api["responses"] = [httpx.Response(200, json=envelope(GOOD))]
    job_id = await seed(db_session_factory)

    async def snapshot():
        async with db_session_factory() as session:
            result = await session.execute(
                select(PackageAnalysis).order_by(PackageAnalysis.package)
            )
            return [
                (
                    row.package,
                    row.dependencies,
                    row.needed_by,
                    row.floor,
                    row.queued,
                    row.updated_at,
                )
                for row in result.scalars()
            ]

    before = await snapshot()
    await llm_stage(context(job_id, db_session_factory))
    assert await snapshot() == before


# --- rejection, retry, park ----


async def test_a_below_floor_response_is_retried_then_parked_never_clamped(
    db_env, classification_env, fake_api, db_session_factory
):
    """The end-to-end form of the repo's central safety rule. `com.example.launcher` sits in
    priv-app, floor Advanced; a model that keeps answering Recommended must produce a parked
    row and NOT a row silently raised to Advanced."""
    permissive = dict(GOOD, removal="Recommended")
    fake_api["responses"] = [
        lambda request: httpx.Response(
            200,
            json=envelope(permissive if "launcher" in request.content.decode() else GOOD),
        )
    ]
    job_id = await seed(db_session_factory)

    await llm_stage(context(job_id, db_session_factory))

    stored = await rows(db_session_factory)
    parked = stored["com.example.launcher"]
    assert parked.parked is True
    assert parked.removal is None, "a rejected proposal must not be stored, raised or otherwise"
    assert "removal" in parked.parked_reason
    assert "Advanced" in parked.parked_reason
    assert parked.attempts == get_settings().classification_max_calls_per_package
    # The other package is unaffected: one bad package must not cost the rest.
    assert stored["com.example.notes"].parked is False


async def test_a_malformed_response_retries_within_the_cap_then_succeeds(
    db_env, classification_env, fake_api, db_session_factory
):
    fake_api["responses"] = [
        httpx.Response(200, json=envelope("not json at all")),
        httpx.Response(200, json=envelope(GOOD)),
    ]
    job_id = await seed(db_session_factory)

    await llm_stage(context(job_id, db_session_factory))

    stored = await rows(db_session_factory)
    assert all(row.parked is False for row in stored.values())
    assert any(row.attempts == 2 for row in stored.values())


async def test_a_park_records_the_field_and_the_reason(
    db_env, classification_env, fake_api, db_session_factory
):
    fake_api["responses"] = [httpx.Response(200, json=envelope(dict(GOOD, dependencies=["com.x"])))]
    job_id = await seed(db_session_factory)

    await llm_stage(context(job_id, db_session_factory))

    for row in (await rows(db_session_factory)).values():
        assert row.parked is True
        assert "dependencies" in row.parked_reason
        assert "corpus graph" in row.parked_reason


async def test_unknown_is_stored_as_a_valid_outcome_rather_than_a_park(
    db_env, classification_env, fake_api, db_session_factory
):
    unknown = dict(GOOD, description=UNKNOWN, unknown_fields=["description"])
    fake_api["responses"] = [httpx.Response(200, json=envelope(unknown))]
    job_id = await seed(db_session_factory)

    await llm_stage(context(job_id, db_session_factory))

    row = (await rows(db_session_factory))["com.example.notes"]
    assert row.parked is False
    assert row.description == UNKNOWN
    assert row.unknown_fields == ["description"]


# --- idempotence and re-runs ----


async def test_a_second_run_over_an_unchanged_corpus_calls_nothing(
    db_env, classification_env, fake_api, db_session_factory
):
    """The model is not reproducible, so re-asking the same question costs money and returns
    a different answer. A plain re-run must select nothing."""
    fake_api["responses"] = [httpx.Response(200, json=envelope(GOOD))]
    job_id = await seed(db_session_factory)

    await llm_stage(context(job_id, db_session_factory))
    first_calls = len(fake_api["requests"])
    await llm_stage(context(job_id, db_session_factory))

    assert len(fake_api["requests"]) == first_calls


async def test_reclassify_asks_again_and_replaces_only_llm_owned_fields(
    db_env, classification_env, fake_api, db_session_factory
):
    """A human edit in triage carries `human:` provenance and must survive a re-run; the
    model's own fields are replaced."""
    fake_api["responses"] = [httpx.Response(200, json=envelope(GOOD))]
    job_id = await seed(db_session_factory)
    await llm_stage(context(job_id, db_session_factory))

    async with db_session_factory() as session, session.begin():
        row = await session.get(PackageClassification, "com.example.notes")
        row.description = "A human wrote this description during triage."
        row.provenance = {**row.provenance, "description": "human:uwuclxdy"}

    second = dict(GOOD, description="A different model answer entirely, at some length.")
    fake_api["responses"] = [httpx.Response(200, json=envelope(second))]
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session, kind=JobKind.CLASSIFICATION.value, params={"reclassify": True}
        )
        rerun_id = job.id
    await llm_stage(context(rerun_id, db_session_factory))

    stored = await rows(db_session_factory)
    kept = stored["com.example.notes"]
    assert kept.description == "A human wrote this description during triage."
    assert kept.provenance["description"] == "human:uwuclxdy"
    # The model still owns everything it was not overruled on.
    assert stored["com.example.launcher"].description == second["description"]
    assert kept.provenance["removal"].startswith("llm:")


async def test_a_park_against_new_evidence_never_leaves_a_below_floor_answer_on_the_row(
    db_env, classification_env, fake_api, db_session_factory
):
    """Reviewer finding 3, and it is the repo's central safety observable reached without
    `raise_to_floor` ever being called.

    A package is accepted at `Recommended`. A second device then observes it `coreApp=true`,
    so its floor becomes `Unsafe` and its bundle hash changes. The model keeps answering
    `Recommended`, is rejected, and parks. Re-pointing the old answer at the new hash would
    leave a stored `removal` of `Recommended` on a row naming a bundle whose floor is
    `Unsafe` — and permanently, because the matching hash then makes the package skip
    selection forever.
    """
    fake_api["responses"] = [httpx.Response(200, json=envelope(dict(GOOD, removal="Recommended")))]
    async with db_session_factory() as session, session.begin():
        firmware = await jobs_module.create_job(
            session,
            kind=JobKind.FIRMWARE_ANALYSIS.value,
            params={"driver": "pixel", "device": "oriole"},
        )
        await record_device_scan(
            session,
            job_id=firmware.id,
            device_key=PIXEL,
            build=BUILD,
            scanned_at=NOW,
            apk_total=1,
            parsed_ok=1,
            failures=[],
        )
        await store_device_facts(
            session,
            device_key=PIXEL,
            build=BUILD,
            facts=[make_facts("com.example.notes")],
            observed_at=NOW,
        )
        session.add(
            PackageAnalysis(
                package="com.example.notes", updated_at=NOW, queued=True, filter_verdict="queued"
            )
        )
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(session, kind=JobKind.CLASSIFICATION.value)
        job_id = job.id

    await llm_stage(context(job_id, db_session_factory))
    accepted = (await rows(db_session_factory))["com.example.notes"]
    assert accepted.parked is False and accepted.removal == "Recommended"

    # A second device raises the floor to Unsafe, so the evidence — and the hash — change.
    async with db_session_factory() as session, session.begin():
        await store_device_facts(
            session,
            device_key="google:emulator-a16",
            build="android-36.1",
            facts=[make_facts("com.example.notes", core_app=True)],
            observed_at=NOW,
        )
    await llm_stage(context(job_id, db_session_factory))

    row = (await rows(db_session_factory))["com.example.notes"]
    assert row.parked is True
    assert row.bundle_sha256 != accepted.bundle_sha256, "new evidence, new hash"
    assert row.removal is None, (
        "a park against evidence the proposal never answered must clear it: "
        f"{row.removal!r} would sit below this row's own bundle floor"
    )
    assert row.description is None and row.uad_list is None

    # And the row's own claim holds: whatever removal it carries is at or above the floor of
    # the bundle it names.
    async with db_session_factory() as session:
        corpus = await require_corpus(session)
        config = await load_config_inputs(session)
    floors = compute_floors(corpus, config=config)
    assert floors["com.example.notes"].floor is Removal.UNSAFE
    if row.removal is not None:
        assert danger_rank(Removal(row.removal)) >= danger_rank(floors["com.example.notes"].floor)


async def test_a_human_edit_survives_a_park_against_new_evidence(
    db_env, classification_env, fake_api, db_session_factory
):
    """The clearing above is about an unvalidated MODEL answer. A triage edit was never the
    model's to discard, so it outlives the park."""
    fake_api["responses"] = [httpx.Response(200, json=envelope(GOOD))]
    job_id = await seed(db_session_factory)
    await llm_stage(context(job_id, db_session_factory))

    async with db_session_factory() as session, session.begin():
        row = await session.get(PackageClassification, "com.example.notes")
        row.description = "A human wrote this during triage."
        row.provenance = {**row.provenance, "description": "human:uwuclxdy"}

    fake_api["responses"] = [httpx.Response(200, json=envelope(dict(GOOD, removal="Recommended")))]
    async with db_session_factory() as session, session.begin():
        await store_device_facts(
            session,
            device_key="google:emulator-a16",
            build="android-36.1",
            facts=[make_facts("com.example.notes", core_app=True)],
            observed_at=NOW,
        )
    await llm_stage(context(job_id, db_session_factory))

    row = (await rows(db_session_factory))["com.example.notes"]
    assert row.parked is True
    assert row.description == "A human wrote this during triage."
    assert row.provenance["description"] == "human:uwuclxdy"
    assert row.removal is None, "the model's own unvalidated answer still goes"


async def test_a_changed_corpus_makes_a_package_a_candidate_again(
    db_env, classification_env, fake_api, db_session_factory
):
    """The idempotence key is the bundle hash, not the package name: new evidence is a new
    question and has to be asked."""
    fake_api["responses"] = [httpx.Response(200, json=envelope(GOOD))]
    job_id = await seed(db_session_factory)
    await llm_stage(context(job_id, db_session_factory))
    before = len(fake_api["requests"])

    async with db_session_factory() as session, session.begin():
        await store_device_facts(
            session,
            device_key="google:emulator-a16",
            build="android-36.1",
            facts=[make_facts("com.example.notes", persistent=True)],
            observed_at=NOW,
        )
    await llm_stage(context(job_id, db_session_factory))

    assert len(fake_api["requests"]) > before


# --- selection ----


async def test_a_limit_bounds_the_spend_and_is_a_deterministic_prefix(
    db_env, classification_env, fake_api, db_session_factory
):
    fake_api["responses"] = [httpx.Response(200, json=envelope(GOOD))]
    async with db_session_factory() as session, session.begin():
        pass
    await seed(db_session_factory)
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session, kind=JobKind.CLASSIFICATION.value, params={"limit": 1}
        )
        limited_id = job.id

    await llm_stage(context(limited_id, db_session_factory))

    stored = await rows(db_session_factory)
    assert list(stored) == ["com.example.launcher"], "name-ordered prefix, not query order"


async def test_the_settings_ceiling_cannot_be_raised_by_a_jobs_own_params(
    db_env, classification_env, fake_api, monkeypatch, db_session_factory
):
    monkeypatch.setenv("CLASSIFICATION_MAX_PACKAGES", "1")
    fake_api["responses"] = [httpx.Response(200, json=envelope(GOOD))]
    await seed(db_session_factory)
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session, kind=JobKind.CLASSIFICATION.value, params={"limit": 500}
        )
        job_id = job.id

    await llm_stage(context(job_id, db_session_factory))

    assert len(await rows(db_session_factory)) == 1


async def test_named_packages_replace_the_queue(
    db_env, classification_env, fake_api, db_session_factory
):
    fake_api["responses"] = [httpx.Response(200, json=envelope(GOOD))]
    await seed(db_session_factory)
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session,
            kind=JobKind.CLASSIFICATION.value,
            params={"packages": ["com.example.notes"]},
        )
        job_id = job.id

    await llm_stage(context(job_id, db_session_factory))

    assert list(await rows(db_session_factory)) == ["com.example.notes"]


def test_select_candidates_is_a_sorted_prefix_regardless_of_input_order():
    class Fake:
        def __init__(self, digest):
            self.sha256 = digest

    bundles = {name: Fake(name) for name in ("com.c", "com.a", "com.b")}
    forward = select_candidates(
        queued=["com.c", "com.a", "com.b"], bundles=bundles, existing={}, limit=2
    )
    backward = select_candidates(
        queued=["com.b", "com.c", "com.a"], bundles=bundles, existing={}, limit=2
    )
    assert forward == backward == ["com.a", "com.b"]


# --- pipeline order ----


async def test_an_empty_queue_is_refused_rather_than_silently_succeeding(
    db_env, classification_env, fake_api, db_session_factory
):
    """Zero queued packages means the filter never ran, not "nothing worth proposing"."""
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(session, kind=JobKind.CLASSIFICATION.value)
        job_id = job.id
    async with db_session_factory() as session, session.begin():
        await store_device_facts(
            session, device_key=PIXEL, build=BUILD, facts=CORPUS, observed_at=NOW
        )

    with pytest.raises(ClassificationStoreError, match="queued = true"):
        await llm_stage(context(job_id, db_session_factory))

    assert fake_api["requests"] == []


async def test_an_exhausted_client_parks_the_package_instead_of_killing_the_job(
    db_env, classification_env, fake_api, db_session_factory
):
    """Reviewer finding 1. `complete_json` raising after its own wire retries used to escape
    `_classify_one` entirely, take the TaskGroup down, and leave NO row — not even a park —
    while cancelling every sibling. The empty-content shape is DeepSeek's own documented bug,
    so this is the failure most likely to happen in production."""
    fake_api["responses"] = [httpx.Response(200, json=envelope("", finish_reason="stop"))]
    job_id = await seed(db_session_factory)

    await llm_stage(context(job_id, db_session_factory))  # returns normally

    stored = await rows(db_session_factory)
    assert sorted(stored) == ["com.example.launcher", "com.example.notes"]
    budget = get_settings().classification_max_calls_per_package
    for row in stored.values():
        assert row.parked is True
        assert row.attempts == budget
        assert "empty content" in row.parked_reason
    assert len(fake_api["requests"]) == 2 * budget


async def test_a_persistent_503_parks_the_package_rather_than_the_job(
    db_env, classification_env, fake_api, db_session_factory
):
    """Same class as above through the other retryable error."""
    fake_api["responses"] = [httpx.Response(503, text="overloaded")]
    job_id = await seed(db_session_factory)

    await llm_stage(context(job_id, db_session_factory))

    stored = await rows(db_session_factory)
    budget = get_settings().classification_max_calls_per_package
    assert len(stored) == 2
    assert all(row.parked and row.attempts == budget for row in stored.values())
    assert len(fake_api["requests"]) == 2 * budget


async def test_the_two_retry_layers_do_not_multiply_and_the_row_reports_true_spend(
    db_env, classification_env, fake_api, db_session_factory
):
    """Reviewer finding 2. The mixed shape is the only one that separates a per-call
    decrement from a per-request one: a `503, 503, below-floor` cycle costs three requests
    inside ONE re-prompt, so charging the budget one per `complete_json` would let the
    package spend three times its ceiling while the row reported a third of it."""
    below_floor = dict(GOOD, removal="Recommended")
    cycle = [
        httpx.Response(503, text="busy"),
        httpx.Response(503, text="busy"),
        httpx.Response(200, json=envelope(below_floor)),
    ]
    state = {"i": 0}

    def handler(request):
        response = cycle[state["i"] % len(cycle)]
        state["i"] += 1
        return response

    fake_api["responses"] = [handler]
    await seed(db_session_factory)
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session,
            kind=JobKind.CLASSIFICATION.value,
            params={"packages": ["com.example.launcher"]},
        )
        job_id = job.id

    await llm_stage(context(job_id, db_session_factory))

    budget = get_settings().classification_max_calls_per_package
    row = (await rows(db_session_factory))["com.example.launcher"]
    assert row.parked is True
    # The row's own count IS the transport's count, and both are the single named budget.
    assert row.attempts == len(fake_api["requests"]) == budget


async def test_an_unexpected_http_status_parks_one_package_without_cancelling_its_siblings(
    db_env, classification_env, fake_api, db_session_factory
):
    """A 502 out of a proxy in front of the API is an ordinary event, and it raises a bare
    `DeepSeekError` — neither the malformed class nor the unavailable one, so `_classify_one`
    caught nothing, `complete_json` never retried it, and it escaped into the TaskGroup. That
    cancelled every sibling package, answers already paid for included, and left no row at
    all: the siblings looked like they had never been asked and the failing package was
    silently re-tried by every future run forever.
    """
    failing = "com.example.alpha"

    def handler(request):
        if requested_package(request) == failing:
            return httpx.Response(502, text="<html>502 Bad Gateway</html>")
        return httpx.Response(200, json=envelope(GOOD))

    fake_api["responses"] = [handler]
    job_id = await seed(db_session_factory, WIDE_CORPUS)

    await llm_stage(context(job_id, db_session_factory))  # returns normally

    stored = await rows(db_session_factory)
    assert sorted(stored) == WIDE_CANDIDATES, "every package reached a row"
    for package in WIDE_CANDIDATES:
        if package == failing:
            continue
        assert stored[package].parked is False, f"{package} was cancelled by a sibling"
        assert stored[package].description == GOOD["description"], package
    parked = stored[failing]
    assert parked.parked is True
    assert "502" in parked.parked_reason, "the reason names the status that caused the park"
    # One request: nothing documents this status as transient, so it is not retried, and the
    # row reports what it really spent rather than the whole budget.
    assert parked.attempts == 1


async def test_a_park_from_an_unexpected_status_is_charged_the_reprompts_before_it(
    db_env, classification_env, fake_api, db_session_factory
):
    """A park's `attempts` is the package's real spend on every path that can reach it. The
    exception only knows about the request it died in, so a below-floor re-prompt followed by
    a 404 is two requests and a row saying one would under-report the budget by half."""
    below_floor = dict(GOOD, removal="Recommended")
    fake_api["responses"] = [
        httpx.Response(200, json=envelope(below_floor)),
        httpx.Response(404, text="no such model"),
    ]
    await seed(db_session_factory)
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session,
            kind=JobKind.CLASSIFICATION.value,
            params={"packages": ["com.example.launcher"]},
        )
        job_id = job.id

    await llm_stage(context(job_id, db_session_factory))

    row = (await rows(db_session_factory))["com.example.launcher"]
    assert row.parked is True
    assert "404" in row.parked_reason
    assert row.attempts == len(fake_api["requests"]) == 2


async def test_a_database_failure_on_one_package_never_cancels_the_others(
    db_env, classification_env, fake_api, db_session_factory, monkeypatch
):
    """The same containment through the other half. Every package in the group has already
    been paid for by the time anything is written, so a write that fails for one of them
    costs that one row rather than all 48 — and the failure is recorded rather than lost."""
    real_store = stages_module.store_classification

    async def store(session, classification, **kwargs):
        if classification.package == "com.example.notes":
            raise RuntimeError("the write nobody predicted")
        await real_store(session, classification, **kwargs)

    monkeypatch.setattr(stages_module, "store_classification", store)
    fake_api["responses"] = [httpx.Response(200, json=envelope(GOOD))]
    job_id = await seed(db_session_factory)

    await llm_stage(context(job_id, db_session_factory))

    stored = await rows(db_session_factory)
    assert stored["com.example.launcher"].parked is False, "the sibling's answer survived"
    assert stored["com.example.notes"].parked is True
    assert "RuntimeError" in stored["com.example.notes"].parked_reason


async def test_a_failed_recovery_write_costs_its_own_package_and_not_its_siblings(
    db_env, classification_env, fake_api, db_session_factory, monkeypatch
):
    """The last hole in the boundary. The wrapper's fallback park is what turns any
    per-package failure into that package's own row, and it ran with nothing around it: a
    park that raised escaped `_classify_and_store` and cancelled every sibling — the exact
    shape this repo has now shipped three times, reached through the one call written to
    prevent it.

    It stays job-level, because a database nothing can be written to is. What changes is the
    blast radius: every sibling records its own answer first, and the stage raises once at the
    end naming what it could not record.
    """
    failing = "com.example.alpha"
    real_store = stages_module.store_classification
    real_park = stages_module.park_package

    async def store(session, classification, **kwargs):
        if classification.package == failing:
            raise RuntimeError("the write nobody predicted")
        await real_store(session, classification, **kwargs)

    async def park(session, package, **kwargs):
        if package == failing:
            raise RuntimeError("and the recovery write failed too")
        await real_park(session, package, **kwargs)

    monkeypatch.setattr(stages_module, "store_classification", store)
    monkeypatch.setattr(stages_module, "park_package", park)
    fake_api["responses"] = [httpx.Response(200, json=envelope(GOOD))]
    job_id = await seed(db_session_factory, WIDE_CORPUS)

    with pytest.raises(stages_module.StageRecordError, match=failing):
        await llm_stage(context(job_id, db_session_factory))

    stored = await rows(db_session_factory)
    assert failing not in stored, "nothing could be written for it, which is why the job fails"
    for package in WIDE_CANDIDATES:
        if package == failing:
            continue
        assert stored[package].parked is False, f"{package} was cancelled by a sibling"
        assert stored[package].description == GOOD["description"], package


async def test_a_floor_that_rose_past_a_triage_edit_costs_neither_the_package_nor_the_group(
    db_env, classification_env, fake_api, db_session_factory
):
    """The blocker end to end, through the two writers that produced it.

    A reviewer rates a package `Recommended`. A second device then observes it
    `coreApp="true"`, so the floor becomes `Unsafe` and the bundle changes — which is also
    what makes the package a candidate again, so the two preconditions arrive together rather
    than independently. The model answers at the new floor. That write used to be refused for
    carrying the stale human rating, the fallback park refused for the same reason, and the
    raise took every sibling in the group down with it.
    """
    fake_api["responses"] = [httpx.Response(200, json=envelope(GOOD))]
    job_id = await seed(db_session_factory, WIDE_CORPUS)
    await llm_stage(context(job_id, db_session_factory))

    edited = "com.example.notes"
    async with db_session_factory() as session, session.begin():
        row = await session.get(PackageClassification, edited)
        row.removal = "Recommended"
        row.provenance = {**row.provenance, "removal": "human:triage"}
    async with db_session_factory() as session, session.begin():
        analysis = await session.get(PackageAnalysis, edited)
        analysis.floor = "Unsafe"
        await store_device_facts(
            session,
            device_key="google:emulator-a16",
            build="android-36.1",
            facts=[make_facts(edited, core_app=True)],
            observed_at=NOW,
        )

    fake_api["responses"] = [httpx.Response(200, json=envelope(dict(GOOD, removal="Unsafe")))]
    async with db_session_factory() as session, session.begin():
        job = await jobs_module.create_job(
            session, kind=JobKind.CLASSIFICATION.value, params={"reclassify": True}
        )
        rerun_id = job.id

    await llm_stage(context(rerun_id, db_session_factory))  # returns normally

    stored = await rows(db_session_factory)
    assert sorted(stored) == WIDE_CANDIDATES, "every sibling still reached a row"
    row = stored[edited]
    assert row.parked is False
    assert row.removal == "Unsafe", "the answer at the new floor is the one that stands"
    assert row.provenance["removal"].startswith("llm:")
    assert row.provenance["removal_superseded"].startswith("rule:floor superseded human:triage")


async def test_an_account_level_failure_on_one_package_still_aborts_the_whole_job(
    db_env, classification_env, fake_api, db_session_factory
):
    """The control for the two above. A 402 reaching ONE package is still a fact about the
    account rather than about that package, so it is the failure that must never become a
    parked row: a boundary that contained everything would turn a dead account into 48 quiet
    parks carrying one real cause, each having burned its retry cap to get there."""
    failing = "com.example.alpha"

    def handler(request):
        if requested_package(request) == failing:
            return httpx.Response(402, text="Insufficient Balance")
        return httpx.Response(200, json=envelope(GOOD))

    fake_api["responses"] = [handler]
    job_id = await seed(db_session_factory, WIDE_CORPUS)

    with pytest.raises(BaseExceptionGroup) as caught:
        await llm_stage(context(job_id, db_session_factory))

    assert caught.group_contains(DeepSeekBalanceError)
    stored = await rows(db_session_factory)
    assert failing not in stored, "no park row for a failure that is not the package's"


async def test_an_empty_balance_aborts_the_job_rather_than_parking_every_package(
    db_env, classification_env, fake_api, db_session_factory
):
    """A 402 is not about this package, so parking on it would burn every remaining
    package's retry cap against a dead account and record 48 parks with one real cause. It
    aborts, and the TaskGroup cancels the siblings rather than leaving them writing rows
    through a session the stage has already closed."""
    fake_api["responses"] = [httpx.Response(402, text="Insufficient Balance")]
    job_id = await seed(db_session_factory)

    with pytest.raises(BaseExceptionGroup) as caught:
        await llm_stage(context(job_id, db_session_factory))

    assert any(isinstance(exc, DeepSeekBalanceError) for exc in caught.value.exceptions)
    assert await rows(db_session_factory) == {}, (
        "no park row for a failure that is not the package's"
    )


async def test_a_missing_key_fails_before_any_call_is_made(
    db_env, classification_env, fake_api, monkeypatch, db_session_factory
):
    monkeypatch.setenv("DEEPSEEK_KEY", "")
    job_id = await seed(db_session_factory)

    with pytest.raises(Exception, match="DEEPSEEK_KEY is empty"):
        await llm_stage(context(job_id, db_session_factory))

    assert fake_api["requests"] == []


async def test_the_llm_handler_is_registered_for_the_classification_kind_only():
    handlers = stages_module.pipeline_stage_handlers()
    assert handlers["llm"] is llm_stage
    assert "llm" not in jobs_module.stages_for("firmware_analysis")
    assert jobs_module.stages_for("classification") == ("llm", "corroborate")
