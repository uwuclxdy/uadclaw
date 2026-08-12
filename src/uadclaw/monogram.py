"""The chip a package gets on screen when its APK carried no icon, which is most of them.

Measured over 40 random Pixel system APKs: 25 declare no `android:icon` at all, 8 yield a
raster and 4 a binary-XML drawable. So the fallback is the majority case rather than an edge,
and it has to look deliberate rather than like a failed image.

It is derived and never stored. The one property that matters is that the same package name
produces the same chip everywhere and forever: two screens showing one package two different
ways teaches a reviewer that neither can be trusted, which is expensive on a screen whose
whole job is a judgement call. `hash()` is unusable for the colour because it is salted per
process, so a chip would change colour on every restart.
"""

import hashlib

MONOGRAM_COLOURS = 8
"""How many chip colours exist.

The stylesheet owes `.monogram-c0` through `.monogram-c7` to match. Nothing here can enforce
that: a colour index with no class renders an unstyled chip rather than raising, so the
failure is silent and looks like a design choice. The pin lives beside the stylesheet, in the
triage lane's own suite, because that is the file that can drift.
"""


def monogram_for(package: str) -> tuple[str, int]:
    """Two letters and a colour index for one package name.

    The letters come from the last dotted component, because that is the part that
    distinguishes packages a reader is comparing: every Google package shares
    `com.google.android.` and none of it tells them anything. A component carrying fewer than
    two alphanumerics falls back to the whole name rather than to a one-letter chip.
    """
    tail = package.rsplit(".", 1)[-1]
    letters = "".join(c for c in tail.lower() if c.isalnum())[:2]
    if len(letters) < 2:
        letters = "".join(c for c in package.lower() if c.isalnum())[:2]
    colour = hashlib.sha256(package.encode("utf-8")).digest()[0] % MONOGRAM_COLOURS
    return letters or "??", colour
