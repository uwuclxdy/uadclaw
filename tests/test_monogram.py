"""The fallback chip has to be the same chip everywhere, so what is pinned here is agreement
rather than appearance.

Two screens derived it separately once and disagreed about a package's own letters, which is
why there is one function and why these tests exist at all.
"""

import subprocess
import sys

import pytest

from uadclaw.monogram import MONOGRAM_COLOURS, monogram_for
from uadclaw.web import templates


def test_the_letters_come_from_the_last_dotted_component():
    # Every Google package shares `com.google.android.`, so a chip built from the front of
    # the name would render the same two letters for most of the corpus.
    assert monogram_for("com.google.android.vending")[0] == "ve"
    assert monogram_for("com.android.settings")[0] == "se"


def test_a_component_too_short_to_fill_the_chip_falls_back_to_the_whole_name():
    assert monogram_for("com.x")[0] == "co"
    assert monogram_for("a.b")[0] == "ab"


def test_punctuation_and_digits_in_a_component_do_not_reach_the_chip_as_punctuation():
    letters, _ = monogram_for("com.vendor.a_b-c")
    assert letters == "ab"


def test_a_name_with_no_alphanumerics_still_renders_something():
    assert monogram_for("...")[0] == "??"


def test_the_palette_is_actually_spread_across_rather_than_nominally_available():
    """`0 <= digest[0] % MONOGRAM_COLOURS < MONOGRAM_COLOURS` is true of every implementation
    including one that returns a constant, so the bound is not worth asserting. What can fail
    is the spread: a chip palette that collapses onto one colour is the same screen as no
    palette at all."""
    packages = [f"com.google.android.{name}" for name in ("gms", "vending", "gsf", "tts")]
    packages += [f"com.android.{name}" for name in ("settings", "systemui", "phone", "vpndialogs")]
    packages += ["com.qti.qcc", "com.motorola.launcher", "com.samsung.knox", "com.oppo.market"]
    seen = {monogram_for(p)[1] for p in packages}
    assert seen <= set(range(MONOGRAM_COLOURS))
    assert len(seen) >= 6, f"12 names landed on {len(seen)} colours: {sorted(seen)}"


def test_the_colour_survives_a_restart():
    """`hash()` is salted per process, so a chip built on it changes colour on every restart
    and the same package looks like a different package after a deploy. This is the assertion
    that fails if anyone swaps the digest for `hash()`."""
    package = "com.google.android.vending"
    expected = monogram_for(package)[1]
    program = f"from uadclaw.monogram import monogram_for;print(monogram_for({package!r})[1])"
    seen = {
        subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            check=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert seen == {str(expected)}


@pytest.mark.parametrize("has_icon", [True, False])
def test_the_macro_renders_one_branch_and_never_both(has_icon):
    macro = templates.env.get_template("partials/pkg_icon.html").module.pkg_icon
    rendered = str(macro("com.google.android.vending", has_icon))
    assert ("<img" in rendered) is has_icon
    assert ("monogram" in rendered) is not has_icon


def test_the_icon_branch_encodes_a_package_name_that_would_otherwise_split_the_query():
    macro = templates.env.get_template("partials/pkg_icon.html").module.pkg_icon
    rendered = str(macro("com.evil&view=decided", True))
    assert "com.evil&view=decided" not in rendered
    assert "%26" in rendered


def test_a_slash_in_a_package_name_never_spells_a_path_segment():
    """A package name comes out of a downloaded APK's manifest, so it is not permitted to
    spell its own path. Jinja's `urlencode` keeps `/` safe, which is right for a query string
    and wrong here: the browser resolves `..` before the request is sent, so the link points
    at something other than the record it sits on."""
    macro = templates.env.get_template("partials/pkg_icon.html").module.pkg_icon
    rendered = str(macro("../../secret", True))
    assert "/icons/..%2F..%2Fsecret" in rendered
    assert "/icons/../.." not in rendered


def test_neither_branch_announces_the_name_a_second_time():
    """Every call site renders the package name as text beside the icon. Announcing it here
    too makes a screen reader read every queue row twice, on the one screen that exists to be
    driven without a pointer."""
    macro = templates.env.get_template("partials/pkg_icon.html").module.pkg_icon
    assert 'alt=""' in str(macro("com.a.b", True))
    assert 'aria-hidden="true"' in str(macro("com.a.b", False))
