"""The removal floor: one case per rule in the design's table, plus the safety property.

`removal` is the one field of this pipeline that can break a stranger's phone, and it reaches
them through Canta, AppManager and android-debloat-list as well as uad-ng. So the tests here
are not "does the ladder work" — they are "can anything, through any call, end up below the
floor a package's own manifest justifies". Two of them are written as properties over the
whole API surface rather than as examples, because an example only covers the call the author
thought of and the next call is the one that will be added later.
"""

import inspect

import pytest

from uadclaw import ladder as ladder_module
from uadclaw.corpus import CorpusPackage
from uadclaw.etcconfig import ConfigInputs
from uadclaw.ladder import (
    DANGER_ORDER,
    SYSTEM_SHARED_USER_ID,
    FloorRule,
    Removal,
    RemovalFloor,
    compute_floors,
    danger_rank,
    deny_list_reason,
    is_below_floor,
    raise_to_floor,
    strictest,
)

PLAIN = CorpusPackage(package="com.example.plain", devices=("pixel:oriole",))


def floors_for(*corpus: CorpusPackage, config: ConfigInputs | None = None):
    return compute_floors(list(corpus), config=config)


# --- one case per rule in the design table -------------------------------------------------

# (id, corpus, config, package under test, expected floor, expected rule)
LADDER_CASES = [
    (
        "core_app",
        [CorpusPackage(package="com.android.systemui", core_app=True)],
        None,
        "com.android.systemui",
        Removal.UNSAFE,
        FloorRule.CORE_APP,
    ),
    (
        "sole_library_provider",
        [
            CorpusPackage(package="com.vendor.lib", libraries=("com.vendor.support",)),
            CorpusPackage(
                package="com.vendor.app", uses_libraries_required=("com.vendor.support",)
            ),
        ],
        None,
        "com.vendor.lib",
        Removal.UNSAFE,
        FloorRule.SOLE_LIBRARY_PROVIDER,
    ),
    (
        "sole_static_role_holder",
        [CorpusPackage(package="com.android.dialer")],
        ConfigInputs(static_role_holders={"android.app.role.DIALER": ("com.android.dialer",)}),
        "com.android.dialer",
        Removal.UNSAFE,
        FloorRule.SOLE_STATIC_ROLE_HOLDER,
    ),
    (
        "deny_list",
        [CorpusPackage(package="com.android.cellbroadcastreceiver")],
        None,
        "com.android.cellbroadcastreceiver",
        Removal.UNSAFE,
        FloorRule.DENY_LIST,
    ),
    (
        "system_shared_uid",
        [CorpusPackage(package="com.android.settings", shared_user_id=SYSTEM_SHARED_USER_ID)],
        None,
        "com.android.settings",
        Removal.EXPERT,
        FloorRule.SYSTEM_SHARED_UID,
    ),
    (
        "persistent",
        [CorpusPackage(package="com.android.phone", persistent=True)],
        None,
        "com.android.phone",
        Removal.EXPERT,
        FloorRule.PERSISTENT,
    ),
    (
        "sole_home_handler",
        [CorpusPackage(package="com.example.launcher", handles_home=True)],
        None,
        "com.example.launcher",
        Removal.EXPERT,
        FloorRule.SOLE_CORE_INTENT_HANDLER,
    ),
    (
        "sole_dialer_handler",
        [CorpusPackage(package="com.example.dialer", handles_dialer=True)],
        None,
        "com.example.dialer",
        Removal.EXPERT,
        FloorRule.SOLE_CORE_INTENT_HANDLER,
    ),
    (
        "sole_sms_deliver_handler",
        [CorpusPackage(package="com.example.messages", handles_sms_deliver=True)],
        None,
        "com.example.messages",
        Removal.EXPERT,
        FloorRule.SOLE_CORE_INTENT_HANDLER,
    ),
    (
        "sole_ime",
        [CorpusPackage(package="com.example.keyboard", is_input_method=True)],
        None,
        "com.example.keyboard",
        Removal.EXPERT,
        FloorRule.SOLE_CORE_INTENT_HANDLER,
    ),
    (
        "priv_app",
        [CorpusPackage(package="com.example.privileged", priv_app=True)],
        None,
        "com.example.privileged",
        Removal.ADVANCED,
        FloorRule.PRIVILEGED,
    ),
    (
        "privapp_allowlisted",
        [CorpusPackage(package="com.example.allowlisted")],
        ConfigInputs(
            privapp_permissions={"com.example.allowlisted": ("android.permission.REBOOT",)}
        ),
        "com.example.allowlisted",
        Removal.ADVANCED,
        FloorRule.PRIVILEGED,
    ),
    (
        "overlay_inherits_target_floor_capped_at_advanced",
        [
            CorpusPackage(package="com.android.systemui", core_app=True),
            CorpusPackage(
                package="com.android.systemui.overlay", overlay_target="com.android.systemui"
            ),
        ],
        None,
        "com.android.systemui.overlay",
        Removal.ADVANCED,
        FloorRule.OVERLAY_TARGET,
    ),
    (
        "default",
        [CorpusPackage(package="com.example.plain")],
        None,
        "com.example.plain",
        Removal.RECOMMENDED,
        FloorRule.DEFAULT,
    ),
]


@pytest.mark.parametrize(
    ("corpus", "config", "package", "expected_floor", "expected_rule"),
    [case[1:] for case in LADDER_CASES],
    ids=[case[0] for case in LADDER_CASES],
)
def test_each_ladder_rule_sets_its_floor(corpus, config, package, expected_floor, expected_rule):
    floor = compute_floors(corpus, config=config)[package]

    assert floor.floor == expected_floor
    assert floor.rule == expected_rule


def test_every_rule_in_the_table_has_a_case():
    """A rule added to `FloorRule` without a case here would ship unproven, and the ladder is
    the one place in this pipeline where unproven means a stranger's phone."""
    covered = {case[5] for case in LADDER_CASES}

    assert covered == set(FloorRule)


# --- the safety property --------------------------------------------------------------------

# Each entry builds a corpus where `com.example.subject` carries one boot-critical marker.
BOOT_CRITICAL = {
    "core_app": (
        [CorpusPackage(package="com.example.subject", core_app=True)],
        ConfigInputs(),
    ),
    "sole_library_provider": (
        [
            CorpusPackage(package="com.example.subject", static_libraries=("com.example.lib",)),
            CorpusPackage(package="com.example.user", uses_libraries_required=("com.example.lib",)),
        ],
        ConfigInputs(),
    ),
    "sole_static_role_holder": (
        [CorpusPackage(package="com.example.subject")],
        ConfigInputs(static_role_holders={"android.app.role.SMS": ("com.example.subject",)}),
    ),
    "deny_list": (
        [CorpusPackage(package="com.example.subject.cellbroadcast")],
        ConfigInputs(),
    ),
}


def boot_critical_floor(marker: str) -> RemovalFloor:
    corpus, config = BOOT_CRITICAL[marker]
    package = corpus[0].package
    return compute_floors(corpus, config=config)[package]


@pytest.mark.parametrize("marker", sorted(BOOT_CRITICAL))
@pytest.mark.parametrize("proposal", DANGER_ORDER, ids=[str(item) for item in DANGER_ORDER])
def test_a_boot_critical_package_cannot_be_rated_below_unsafe(marker, proposal):
    """The property the whole ladder exists for: whatever a later stage proposes — including
    the most permissive tier there is — combining it with the floor yields `Unsafe`."""
    floor = boot_critical_floor(marker)

    assert floor.floor == Removal.UNSAFE
    assert raise_to_floor(proposal, floor) == Removal.UNSAFE
    assert is_below_floor(proposal, floor) is (proposal != Removal.UNSAFE)


# Every public callable in `uadclaw.ladder` that can produce a rating, and how to invoke it
# with a boot-critical package and the most permissive proposal available. A function added to
# that module without an entry here fails the test below rather than shipping unexercised.
RATING_CALLABLES = {
    "strictest": lambda floor: strictest(Removal.RECOMMENDED, floor.floor),
    "raise_to_floor": lambda floor: raise_to_floor(Removal.RECOMMENDED, floor),
    "compute_floors": lambda floor: (
        compute_floors(BOOT_CRITICAL["core_app"][0], config=BOOT_CRITICAL["core_app"][1])[
            "com.example.subject"
        ].floor
    ),
}


def _produces_a_rating(function) -> bool:
    annotation = inspect.signature(function).return_annotation
    return "Removal" in str(annotation)


def test_no_public_ladder_call_can_return_below_the_floor():
    """The "through ANY code path" half, as a property of the module's surface rather than a
    spot check: every public function that can hand back a rating is enumerated from the
    module itself, must be registered here, and must answer `Unsafe` for a boot-critical
    package even when handed the most permissive proposal."""
    surface = {
        name
        for name, value in vars(ladder_module).items()
        if not name.startswith("_")
        and inspect.isfunction(value)
        and value.__module__ == ladder_module.__name__
        and _produces_a_rating(value)
    }

    assert surface == set(RATING_CALLABLES), (
        "a ladder function that can return a rating is not covered by the floor property; "
        "register it in RATING_CALLABLES with a boot-critical invocation"
    )
    floor = boot_critical_floor("core_app")
    for name, invoke in sorted(RATING_CALLABLES.items()):
        assert invoke(floor) == Removal.UNSAFE, f"{name} returned below the floor"


def test_a_floor_cannot_be_constructed_below_its_own_evidence():
    """Even hand-built, the type refuses to understate itself, so a caller assembling a
    `RemovalFloor` from stored columns cannot reintroduce a lower floor."""
    fired = boot_critical_floor("core_app").fired

    with pytest.raises(ValueError, match="does not match the strictest rule"):
        RemovalFloor(
            package="com.example.subject",
            floor=Removal.RECOMMENDED,
            rule=FloorRule.DEFAULT,
            fired=fired,
        )


def test_danger_order_is_not_the_enums_own_string_order():
    """`Removal` is a `StrEnum`, so `"Expert" < "Recommended"` compares true and sorting the
    tiers as strings inverts the ladder. Pinned because that mistake is invisible: every
    comparison still runs, and every rating simply lands one tier too low."""
    assert danger_rank(Removal.UNSAFE) > danger_rank(Removal.EXPERT)
    assert danger_rank(Removal.EXPERT) > danger_rank(Removal.ADVANCED)
    assert danger_rank(Removal.ADVANCED) > danger_rank(Removal.RECOMMENDED)
    assert sorted(DANGER_ORDER) != list(DANGER_ORDER)
    assert Removal.EXPERT < Removal.RECOMMENDED


# --- the clauses that are easy to get backwards ---------------------------------------------


def test_the_privapp_allowlist_is_an_integration_score_and_never_a_boot_risk_flag():
    """AOSP's "device won't boot" clause fires for a package that is still PRESENT and asks
    for a permission nobody allowlisted; removing the app removes the request. So a long
    allowlist entry is context, and it must never climb past `Advanced`."""
    config = ConfigInputs(
        privapp_permissions={
            "com.example.deeply.wired": tuple(f"android.permission.P{index}" for index in range(40))
        }
    )
    floor = compute_floors([CorpusPackage(package="com.example.deeply.wired")], config=config)[
        "com.example.deeply.wired"
    ]

    assert floor.floor == Removal.ADVANCED
    assert floor.privapp_allowlisted is True
    assert floor.privapp_permission_count == 40


def test_an_overlay_of_a_deny_listed_target_is_still_unsafe_on_its_own_name():
    """The overlay cap lowers nothing: it contributes `Advanced`, and the strictest rule that
    fired still wins. A real Pixel ships `com.android.cellbroadcastreceiver.overlay.pixel`,
    which is exactly this shape."""
    corpus = [
        CorpusPackage(package="com.android.cellbroadcastreceiver", core_app=True),
        CorpusPackage(
            package="com.android.cellbroadcastreceiver.overlay.pixel",
            overlay_target="com.android.cellbroadcastreceiver",
        ),
    ]

    floor = compute_floors(corpus)["com.android.cellbroadcastreceiver.overlay.pixel"]

    assert floor.floor == Removal.UNSAFE
    assert floor.rule == FloorRule.DENY_LIST
    assert {item.rule for item in floor.fired} >= {FloorRule.DENY_LIST, FloorRule.OVERLAY_TARGET}


def test_an_overlay_of_an_ordinary_target_inherits_only_that_targets_floor():
    corpus = [
        CorpusPackage(package="com.example.app", persistent=True),
        CorpusPackage(package="com.example.app.overlay", overlay_target="com.example.app"),
        CorpusPackage(package="com.example.plain"),
        CorpusPackage(package="com.example.plain.overlay", overlay_target="com.example.plain"),
    ]

    floors = compute_floors(corpus)

    assert floors["com.example.app.overlay"].floor == Removal.ADVANCED
    # The clause still fires and is still what the card reports; it just inherited nothing,
    # because the target is `Recommended` too.
    assert floors["com.example.plain.overlay"].floor == Removal.RECOMMENDED
    assert floors["com.example.plain.overlay"].rule == FloorRule.OVERLAY_TARGET


def test_an_overlay_whose_target_is_not_in_the_corpus_gets_no_inherited_floor():
    floor = compute_floors(
        [CorpusPackage(package="com.example.overlay", overlay_target="com.absent.target")]
    )["com.example.overlay"]

    assert floor.floor == Removal.RECOMMENDED
    assert FloorRule.OVERLAY_TARGET not in {item.rule for item in floor.fired}


def test_a_library_with_a_second_provider_is_not_a_sole_provider():
    corpus = [
        CorpusPackage(package="com.vendor.lib.a", libraries=("com.vendor.support",)),
        CorpusPackage(package="com.vendor.lib.b", libraries=("com.vendor.support",)),
        CorpusPackage(package="com.vendor.app", uses_libraries_required=("com.vendor.support",)),
    ]

    floors = compute_floors(corpus)

    assert floors["com.vendor.lib.a"].floor == Removal.RECOMMENDED
    assert floors["com.vendor.lib.b"].floor == Removal.RECOMMENDED


def test_a_library_nobody_requires_does_not_pin_its_provider():
    """`com.google.android.trichromelibrary` on both local corpora: it declares a static
    library and no package in either image consumes it, so there is nothing to break."""
    floors = compute_floors(
        [
            CorpusPackage(
                package="com.google.android.trichromelibrary",
                static_libraries=("com.google.android.trichromelibrary",),
            )
        ]
    )

    assert floors["com.google.android.trichromelibrary"].floor == Removal.RECOMMENDED


def test_a_static_role_with_another_corpus_holder_is_not_sole():
    config = ConfigInputs(
        static_role_holders={"android.app.role.SMS": ("com.example.a", "com.example.b")}
    )
    corpus = [CorpusPackage(package="com.example.a"), CorpusPackage(package="com.example.b")]

    floors = compute_floors(corpus, config=config)

    assert floors["com.example.a"].floor == Removal.RECOMMENDED
    assert floors["com.example.b"].floor == Removal.RECOMMENDED


def test_two_handlers_of_the_same_core_intent_are_neither_of_them_sole():
    corpus = [
        CorpusPackage(package="com.example.launcher.a", handles_home=True),
        CorpusPackage(package="com.example.launcher.b", handles_home=True),
    ]

    floors = compute_floors(corpus)

    assert floors["com.example.launcher.a"].floor == Removal.RECOMMENDED
    assert floors["com.example.launcher.b"].floor == Removal.RECOMMENDED


# --- the hardcoded jurisdiction deny-list ---------------------------------------------------

DENY_LISTED = [
    "com.android.cellbroadcastreceiver",
    "com.google.android.cellbroadcastservice",
    "com.android.emergency",
    "com.sec.android.emergencylauncher",
    "com.lge.cmas",
    "com.huaqin.sarcontroller",
    "com.tct.reducesar",
    "com.asus.sarprotection",
]

# Real entries from the live 5372-entry upstream list that a looser matcher swallows. Each one
# is a bare "wea"/"amber"/"sar" substring away from being rated Unsafe for no reason.
NOT_DENY_LISTED = [
    "telekom.hu.android.mobilvasarlas",
    "com.samsung.android.weather",
    "com.google.android.wearable.pixel.aspen",
    "com.motorola.livewallpaper3.prebuilt.mysterious_amber",
    "com.google.android.apps.weather",
]


@pytest.mark.parametrize("package", DENY_LISTED)
def test_the_deny_list_catches_the_statutory_families(package):
    assert deny_list_reason(package) is not None


@pytest.mark.parametrize("package", NOT_DENY_LISTED)
def test_the_deny_list_does_not_catch_lookalikes_from_the_live_upstream_list(package):
    assert deny_list_reason(package) is None


def test_the_strictest_rule_wins_and_every_rule_that_fired_is_recorded():
    """The triage card shows which rule set the floor, so a package that trips four rules has
    to report the most dangerous one and still carry the rest."""
    floor = compute_floors(
        [
            CorpusPackage(
                package="com.android.cellbroadcastreceiver",
                core_app=True,
                persistent=True,
                priv_app=True,
                shared_user_id=SYSTEM_SHARED_USER_ID,
            )
        ]
    )["com.android.cellbroadcastreceiver"]

    assert floor.floor == Removal.UNSAFE
    assert floor.rule == FloorRule.CORE_APP
    assert [item.rule for item in floor.fired] == [
        FloorRule.CORE_APP,
        FloorRule.DENY_LIST,
        FloorRule.SYSTEM_SHARED_UID,
        FloorRule.PERSISTENT,
        FloorRule.PRIVILEGED,
        FloorRule.DEFAULT,
    ]


def test_the_corpus_order_does_not_change_a_single_floor():
    corpus = [
        CorpusPackage(package="com.vendor.lib", libraries=("com.vendor.support",)),
        CorpusPackage(package="com.vendor.app", uses_libraries_required=("com.vendor.support",)),
        CorpusPackage(package="com.vendor.app.overlay", overlay_target="com.vendor.app"),
        PLAIN,
    ]

    forward = {name: floor.as_json() for name, floor in compute_floors(corpus).items()}
    backward = {
        name: floor.as_json() for name, floor in compute_floors(list(reversed(corpus))).items()
    }

    assert forward == backward
