"""The removal floor, enforced where a row actually becomes observable.

`classify.validate_response` holds the floor for the MODEL, and held it alone for as long as
the model was the only writer. Triage adds a second one, so these assert the property against
the stored row rather than against a return value: a `package_classification.removal` may
never rank below the same package's `package_analysis.floor`, whichever writer put it there.

Every assertion here reads the row back out of Postgres after the call. A store that raised
and wrote anyway, or a store whose caller swallowed the raise, both pass a return-value test.
"""

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from uadclaw.classify import Classification, Confidence, UadList
from uadclaw.classifystore import BelowFloorError, park_package, store_classification
from uadclaw.ladder import Removal
from uadclaw.models import PackageAnalysis, PackageClassification

NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
PACKAGE = "com.example.notes"
BUNDLE = "b" * 64
MODEL = "deepseek-v4-flash"


def proposal(
    *,
    package: str = PACKAGE,
    removal: Removal = Removal.ADVANCED,
    owner: str = f"llm:{MODEL}",
    bundle_sha256: str = BUNDLE,
) -> Classification:
    return Classification(
        package=package,
        bundle_sha256=bundle_sha256,
        description="Vendor notes application. Removing it loses locally stored notes.",
        list=UadList.MISC,
        removal=removal,
        confidence=Confidence.MEDIUM,
        unknown_fields=(),
        reasoning_brief="",
        provenance={"description": owner, "removal": owner, "list": owner},
    )


async def seed_floor(session_factory, floor: str | None, *, package: str = PACKAGE) -> None:
    async with session_factory() as session, session.begin():
        session.add(
            PackageAnalysis(
                package=package,
                updated_at=NOW,
                queued=True,
                filter_verdict="queued",
                floor=floor,
                floor_rule="core_app" if floor else None,
                floor_reasons=[{"rule": "core_app", "floor": floor, "detail": "coreApp"}]
                if floor
                else [],
            )
        )


async def store(session_factory, classification: Classification) -> None:
    async with session_factory() as session, session.begin():
        await store_classification(
            session,
            classification,
            model=MODEL,
            thinking=True,
            usage={},
            attempts=1,
            at=NOW,
        )


async def stored_row(session_factory, package: str = PACKAGE) -> PackageClassification | None:
    async with session_factory() as session:
        result = await session.execute(
            select(PackageClassification).where(PackageClassification.package == package)
        )
        return result.scalars().one_or_none()


# --- the observable ------------------------------------------------------------------------


async def test_no_row_below_its_floor_survives_a_model_write(db_env, db_session_factory):
    """The property, from the direction `validate_response` already covers. Asserted here
    anyway: the validator is one door and this test is about the other."""
    await seed_floor(db_session_factory, "Unsafe")

    with pytest.raises(BelowFloorError, match="Recommended"):
        await store(db_session_factory, proposal(removal=Removal.RECOMMENDED))

    assert await stored_row(db_session_factory) is None


async def test_no_row_below_its_floor_survives_a_human_write(db_env, db_session_factory):
    """The direction the validator never sees. A triage edit reaches the same columns with
    no `validate_response` anywhere in its path, which is why the gate is at the store."""
    await seed_floor(db_session_factory, "Expert")

    with pytest.raises(BelowFloorError, match="Advanced"):
        await store(
            db_session_factory,
            proposal(removal=Removal.ADVANCED, owner="human:triage"),
        )

    assert await stored_row(db_session_factory) is None


async def test_a_below_floor_write_does_not_overwrite_the_row_already_there(
    db_env, db_session_factory
):
    """The refusal has to leave the good answer standing. A store that raised after its
    UPDATE landed would fail this and pass every "it raised" assertion."""
    await seed_floor(db_session_factory, "Expert")
    await store(db_session_factory, proposal(removal=Removal.UNSAFE))

    with pytest.raises(BelowFloorError):
        await store(db_session_factory, proposal(removal=Removal.RECOMMENDED))

    row = await stored_row(db_session_factory)
    assert row is not None
    assert row.removal == "Unsafe"


# --- ranking, never string order -----------------------------------------------------------


async def test_the_gate_ranks_by_danger_rather_than_by_string_order(db_env, db_session_factory):
    """`Removal` is a `StrEnum`, so `"Recommended" < "Expert"` is False and a `<` on the
    values would wave this straight through. Recommended is rank 0 against Expert's 2."""
    await seed_floor(db_session_factory, "Expert")

    with pytest.raises(BelowFloorError, match="Expert"):
        await store(db_session_factory, proposal(removal=Removal.RECOMMENDED))

    assert await stored_row(db_session_factory) is None


async def test_a_rating_above_its_floor_is_stored(db_env, db_session_factory):
    """The positive leg, and the other half of the string-order kill: lexicographically
    `"Advanced" < "Recommended"` is True, so a string comparison refuses this valid raise."""
    await seed_floor(db_session_factory, "Recommended")

    await store(db_session_factory, proposal(removal=Removal.ADVANCED))

    row = await stored_row(db_session_factory)
    assert row is not None
    assert row.removal == "Advanced"


async def test_a_rating_equal_to_its_floor_is_stored(db_env, db_session_factory):
    """The boundary. A gate written with `<=` refuses every package the model answered at
    exactly its floor, which is the common case rather than an edge one."""
    await seed_floor(db_session_factory, "Advanced")

    await store(db_session_factory, proposal(removal=Removal.ADVANCED))

    row = await stored_row(db_session_factory)
    assert row is not None
    assert row.removal == "Advanced"


# --- a floor that is not there ---------------------------------------------------------------


async def test_a_model_rating_with_no_stored_floor_is_stored_and_says_so(
    db_env, db_session_factory, caplog
):
    """NULL means the rule_ladder stage has not written this package, NOT a floor of the
    lowest rank. The model path validated against a floor `compute_floors` built from the
    live corpus — deliberately not from this column, which can lag it — so refusing here
    would reject a correct answer over a missing denormalised copy. It is logged instead,
    naming the package, because a gate that could not run is not a gate that passed.
    """
    with caplog.at_level("WARNING", logger="uadclaw.classifystore"):
        await store(db_session_factory, proposal(removal=Removal.RECOMMENDED))

    row = await stored_row(db_session_factory)
    assert row is not None
    assert row.removal == "Recommended"
    assert any(PACKAGE in record.getMessage() for record in caplog.records)


async def test_a_human_rating_with_no_stored_floor_is_refused(db_env, db_session_factory):
    """The asymmetry, and the reason it is not "NULL means no constraint". A human rating
    has no upstream validator at all, so a floor it cannot be checked against is the whole
    guarantee missing rather than a duplicate of one."""
    with pytest.raises(BelowFloorError, match="no computed floor"):
        await store(
            db_session_factory,
            proposal(removal=Removal.RECOMMENDED, owner="human:triage"),
        )

    assert await stored_row(db_session_factory) is None


async def test_a_floor_this_code_cannot_read_is_refused(db_env, db_session_factory):
    """A stored floor outside the enum is a corrupt row, and the safe reading of a floor
    nobody can rank is that nothing clears it."""
    await seed_floor(db_session_factory, "Extremely Unsafe")

    with pytest.raises(BelowFloorError, match="Extremely Unsafe"):
        await store(db_session_factory, proposal(removal=Removal.UNSAFE))

    assert await stored_row(db_session_factory) is None


# --- the park path ----------------------------------------------------------------------------


async def test_a_park_never_carries_a_human_rating_the_floor_has_overtaken(
    db_env, db_session_factory
):
    """`park_package` preserves a human-owned field across a park against new evidence, so
    it is a writer of `removal` too and passes the same seam.

    It used to REFUSE here, and that reading was the defect: the rating is unstorable, so
    refusing the park meant refusing every write of this row forever — including the fallback
    park `_classify_and_store` reaches for when the proposal write fails, whose raise then
    escaped its `TaskGroup` and cancelled every sibling package. The floor still holds. What
    changed is that the stale human value is dropped rather than carried into a write that
    cannot land.
    """
    await seed_floor(db_session_factory, "Recommended")
    await store(
        db_session_factory,
        proposal(removal=Removal.RECOMMENDED, owner="human:triage"),
    )

    # The corpus grew: a second device declares `coreApp`, so the floor is Unsafe now and the
    # bundle the model answers has changed with it.
    async with db_session_factory() as session, session.begin():
        analysis = await session.get(PackageAnalysis, PACKAGE)
        assert analysis is not None
        analysis.floor = "Unsafe"

    async with db_session_factory() as session, session.begin():
        await park_package(
            session,
            PACKAGE,
            bundle_sha256="c" * 64,
            model=MODEL,
            thinking=True,
            reason="removal: answered below the floor three times",
            usage={},
            attempts=3,
            at=NOW,
        )

    row = await stored_row(db_session_factory)
    assert row is not None
    assert row.parked is True
    assert row.bundle_sha256 == "c" * 64
    assert row.removal is None, "the rating the floor overtook is gone, never stored below it"
    assert row.provenance["removal_superseded"] == "rule:floor superseded human:triage Recommended"


async def test_a_floor_that_rises_past_a_human_rating_still_takes_a_valid_model_answer(
    db_env, db_session_factory
):
    """The blocker, at the seam that produced it.

    A `human:` removal is preserved into the values dict of every later write and carries its
    provenance with it, so the gate read a MODEL write as the human's and ranked the preserved
    value rather than the answer offered. Once the floor rose past the stored human rating,
    a correct answer AT the new floor was refused and the error named the old human value.
    """
    await seed_floor(db_session_factory, "Recommended")
    await store(
        db_session_factory,
        proposal(removal=Removal.RECOMMENDED, owner="human:triage", bundle_sha256="a" * 64),
    )

    async with db_session_factory() as session, session.begin():
        analysis = await session.get(PackageAnalysis, PACKAGE)
        assert analysis is not None
        analysis.floor = "Unsafe"

    await store(
        db_session_factory,
        proposal(removal=Removal.UNSAFE, bundle_sha256="c" * 64),
    )

    row = await stored_row(db_session_factory)
    assert row is not None
    assert row.removal == "Unsafe"
    assert row.provenance["removal"] == f"llm:{MODEL}", "the value written owns its provenance"


async def test_a_superseded_human_rating_is_recorded_rather_than_dropped_quietly(
    db_env, db_session_factory, caplog
):
    """Dropping it is the only reading that keeps the row writable, so what the reviewer is
    owed is being told. A warning is what an operator greps; the provenance entry is what the
    next reviewer reads off the card, and they are the person whose edit went."""
    await seed_floor(db_session_factory, "Recommended")
    await store(
        db_session_factory,
        proposal(removal=Removal.RECOMMENDED, owner="human:triage", bundle_sha256="a" * 64),
    )
    async with db_session_factory() as session, session.begin():
        analysis = await session.get(PackageAnalysis, PACKAGE)
        assert analysis is not None
        analysis.floor = "Expert"

    with caplog.at_level("WARNING", logger="uadclaw.classifystore"):
        await store(db_session_factory, proposal(removal=Removal.EXPERT, bundle_sha256="c" * 64))

    row = await stored_row(db_session_factory)
    assert row is not None
    assert row.provenance["removal_superseded"] == "rule:floor superseded human:triage Recommended"
    assert any(
        PACKAGE in record.getMessage() and "superseded" in record.getMessage()
        for record in caplog.records
    ), "the drop reaches the log as well as the row"


async def test_a_human_rating_the_floor_has_not_overtaken_is_still_preserved(
    db_env, db_session_factory
):
    """The positive leg, and the control for the three above: without it they are all
    satisfied by a `_preserved` that carries nothing at all, which is the other way to make
    a stale value stop refusing writes and it erases every triage edit ever made."""
    await seed_floor(db_session_factory, "Recommended")
    await store(
        db_session_factory,
        proposal(removal=Removal.EXPERT, owner="human:triage", bundle_sha256="a" * 64),
    )

    await store(
        db_session_factory,
        proposal(removal=Removal.ADVANCED, bundle_sha256="c" * 64),
    )

    row = await stored_row(db_session_factory)
    assert row is not None
    assert row.removal == "Expert", "the model's Advanced must not walk over the human's Expert"
    assert row.provenance["removal"] == "human:triage"
    assert "removal_superseded" not in row.provenance


async def test_a_park_with_nothing_to_preserve_still_lands(db_env, db_session_factory):
    """The park path's ordinary shape: no removal is written at all, so there is nothing to
    rank and the gate must not invent a reason to refuse."""
    await seed_floor(db_session_factory, "Unsafe")

    async with db_session_factory() as session, session.begin():
        await park_package(
            session,
            PACKAGE,
            bundle_sha256=BUNDLE,
            model=MODEL,
            thinking=True,
            reason="removal: answered below the floor three times",
            usage={},
            attempts=3,
            at=NOW,
        )

    row = await stored_row(db_session_factory)
    assert row is not None
    assert row.parked is True
    assert row.removal is None
