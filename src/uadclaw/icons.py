"""One APK's launcher icon as bytes a browser can render. Pure: no database, no scratch, the
same posture as `facts.py`, and called from it while the APK is still open — `extract_facts`
deletes the APKs the moment their facts land, so there is no second pass.

Measured over the 312 APKs of `oriole-cp2a.260705.006.a1` at `max_dpi=320`: 223 declare no
resolvable icon at all, 48 yield a PNG, 5 a WEBP, and 36 a binary-XML drawable. So the absent
case is the majority case (the dashboard falls back to a monogram chip) and the XML case is a
third of everything that is present, which is why this module renders rather than only copies.

Two rules shape the whole file:

- **dispatch on the bytes.** A member named `.png` is whatever the archive says it is, so the
  magic decides the mime and a mismatch is a refusal. Same rule `unpack.py` follows.
- **nothing out of the APK is copied into the SVG.** Every emitted value is parsed into a
  float, a colour or a member of a fixed enum first and then re-serialized from that parsed
  value; `android:pathData` is re-emitted token by token from a fixed command set and
  reformatted numbers. An attribute that does not parse as the type expected refuses the whole
  icon rather than passing through, and an element or attribute outside the support set does
  the same: a half-rendered icon is worse than the monogram it would replace.

The support set is an allowlist rather than a blocklist, because every attribute this file has
not met yet is one it cannot render: `<clip-path>`, a gradient inlined through `aapt:attr`, a
raster nested inside an adaptive layer, `<shape>`, and anything new all land in the same
refusal.
"""

import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Protocol

from loguru import logger as _loguru_logger

# The `logger.disable("androguard")` facts.py's docstring explains, called again here because
# it binds at import time and either module can be the first to import androguard in a given
# process — the icon tests import this one alone. Same mechanism, not a second one.
_loguru_logger.disable("androguard")

from androguard.core.apk import APK  # noqa: E402
from androguard.core.axml import AXMLPrinter  # noqa: E402

logger = logging.getLogger(__name__)

ANDROID_NS = "{http://schemas.android.com/apk/res/android}"

MIME_PNG = "image/png"
MIME_WEBP = "image/webp"
MIME_JPEG = "image/jpeg"
MIME_SVG = "image/svg+xml"

# Everything `package_facts.icon_mime` may hold. The route serves nothing outside this set.
ICON_MIMES: frozenset[str] = frozenset({MIME_PNG, MIME_WEBP, MIME_JPEG, MIME_SVG})

# An icon is decoration on a 48px card. The measured rasters are 4-5 KB and the largest on the
# Pixel corpus is 67 KB, so this stores every ordinary one and skips the outlier rather than
# truncating it — half a PNG is a broken image, not a smaller one.
MAX_ICON_BYTES = 64 * 1024

# xhdpi. The card renders at 48px and the queue rail at 32px, so a denser variant is bytes
# nobody sees.
ICON_MAX_DPI = 320

# A drawable graph out of a downloaded firmware image is untrusted input: the bound is what
# stops a cycle, not the vendor's build tools.
#
# Two SEPARATE axes, and conflating them is what let a stack overflow through review once.
# `_MAX_DRAWABLE_DEPTH` bounds drawable REFERENCE resolution — how many `@ref` and inline-child
# hops one icon may take. `_MAX_ELEMENT_DEPTH` bounds ELEMENT NESTING inside a single decoded
# document, which the reference bound never touches: `<group>` inside `<group>` recurses through
# `_group`/`_vector_children` without resolving anything. Measured on this box, 497 nested
# `<group>` elements reached CPython's default recursion limit, out of roughly 24 KB of AXML.
_MAX_DRAWABLE_DEPTH = 8
_MAX_ELEMENT_DEPTH = 32
_MAX_REF_HOPS = 4

# The 64 KB cap above bounds what is STORED, which is not the same as bounding what is
# ALLOCATED to get there: the raster path checks its input, and the XML path used to decode,
# render and encode an unbounded document before meeting the output cap (1M path tokens peaked
# at 34 MB). The largest binary AXML drawable across both local corpora (540 APKs, top-level
# and referenced) is 22,664 bytes and the median is 960, so this is an order of magnitude of
# headroom over anything measured.
MAX_DRAWABLE_BYTES = 256 * 1024

_AXML_MAGIC = b"\x03\x00\x08\x00"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"

# AOSP's adaptive-icon canvas is 108dp with the outer 18dp on each side reserved for the
# launcher's mask, leaving 72dp visible. Rendering the whole canvas square shows bleed art no
# phone displays, so the document is cropped to the visible square and clipped to the circle a
# Pixel launcher masks with.
_ADAPTIVE_CANVAS = 108.0
_ADAPTIVE_BLEED = 18.0
_MASK_ID = "uadclaw-icon-mask"

_SVG_NS = "http://www.w3.org/2000/svg"

_PATH_COMMANDS = frozenset("MmZzLlHhVvCcSsQqTtAa")
_PATH_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")

# Dimension suffixes Android accepts where this module needs a plain number. dp and px are the
# same unit here: the canvas IS the drawable's own coordinate space.
_DIMENSION_SUFFIXES = ("dip", "dp", "px", "sp", "pt", "in", "mm")

_VECTOR_ATTRIBUTES = frozenset(
    {"name", "width", "height", "viewportWidth", "viewportHeight", "alpha", "autoMirrored"}
)
_GROUP_ATTRIBUTES = frozenset(
    {"name", "rotation", "pivotX", "pivotY", "scaleX", "scaleY", "translateX", "translateY"}
)
_PATH_ATTRIBUTES = frozenset(
    {
        "name",
        "pathData",
        "fillColor",
        "fillAlpha",
        "fillType",
        "strokeColor",
        "strokeAlpha",
        "strokeWidth",
        "strokeLineCap",
        "strokeLineJoin",
        "strokeMiterLimit",
    }
)
_INSET_ATTRIBUTES = frozenset(
    {"drawable", "inset", "insetLeft", "insetTop", "insetRight", "insetBottom", "visible"}
)
_ITEM_ATTRIBUTES = frozenset({"drawable", "id", "left", "top", "right", "bottom"})
_LAYER_ATTRIBUTES = frozenset({"drawable"})

# Both spellings of every enum, because AXML stores an enum as its integer and `AXMLPrinter`
# hands that integer over: `android:fillType="evenOdd"` in the source arrives as `"1"`. The
# integers are AOSP's own (`frameworks/base/core/res/res/values/attrs.xml`). Coding only the
# source spelling refused 2 of the 36 XML drawables on the Pixel corpus.
_FILL_RULES = {"nonZero": "nonzero", "0": "nonzero", "evenOdd": "evenodd", "1": "evenodd"}
_LINE_CAPS = {
    "butt": "butt",
    "0": "butt",
    "round": "round",
    "1": "round",
    "square": "square",
    "2": "square",
}
_LINE_JOINS = {
    "miter": "miter",
    "0": "miter",
    "round": "round",
    "1": "round",
    "bevel": "bevel",
    "2": "bevel",
}


class _Unsupported(Exception):
    """Some part of this drawable is outside the support set, so the whole icon is refused.

    Internal control flow: raised anywhere in the renderer, caught once in
    `svg_from_drawable`, and never seen by a caller. A refusal is a normal outcome — the
    package falls back to its monogram — so it is not an error type anyone handles.
    """


class DrawableSource(Protocol):
    """The APK's resource table, as the renderer needs it.

    Split out so the renderer stays a pure function of parsed XML: resolving `@7F110001`
    against an `resources.arsc` and decoding a nested binary AXML both live behind this,
    which is what lets every shape the renderer supports be tested without an APK.
    """

    def color(self, ref: str) -> str | None:
        """The `#AARRGGBB` literal a reference resolves to, or None when it is not a colour."""

    def element(self, ref: str) -> Any | None:
        """The parsed XML drawable a reference resolves to, or None when it is not one."""


@dataclass(frozen=True, slots=True)
class _Box:
    """Where a layer is painted, in the enclosing drawable's own units."""

    x: float
    y: float
    width: float
    height: float


# --- values ---------------------------------------------------------------------------------


def _num(value: float) -> str:
    """A float as a token of the emitted document. Every number in the output goes through
    here, so `nan`/`inf` cannot reach a viewer as a literal."""
    if not math.isfinite(value):
        raise _Unsupported(f"_num: {value!r} is not a finite number")
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-", "-0") else text


def _float(raw: str | None, default: float | None = None) -> float:
    if raw is None:
        if default is None:
            raise _Unsupported("_float: a required numeric attribute is absent")
        return default
    text = raw.strip()
    for suffix in _DIMENSION_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    try:
        value = float(text)
    except ValueError as exc:
        raise _Unsupported(f"_float: {raw!r} is not a number") from exc
    if not math.isfinite(value):
        raise _Unsupported(f"_float: {raw!r} is not finite")
    return value


def _colour_literal(text: str) -> tuple[str, float]:
    """`#RGB`/`#ARGB`/`#RRGGBB`/`#AARRGGBB` as `(#rrggbb, alpha)`."""
    if not text.startswith("#"):
        raise _Unsupported(f"_colour_literal: {text!r} is not a colour")
    digits = text[1:]
    if not digits or any(char not in _HEX_DIGITS for char in digits):
        raise _Unsupported(f"_colour_literal: {text!r} is not hexadecimal")
    if len(digits) in (3, 4):
        digits = "".join(char * 2 for char in digits)
    if len(digits) == 6:
        digits = "ff" + digits
    if len(digits) != 8:
        raise _Unsupported(f"_colour_literal: {text!r} is not 3, 4, 6 or 8 digits")
    return f"#{digits[2:].lower()}", int(digits[:2], 16) / 255


def _colour(raw: str | None, source: DrawableSource) -> tuple[str, float] | None:
    """`(#rrggbb, alpha)`, None when the attribute is absent, a refusal when it is present and
    is not a colour. A reference that resolves to nothing is a refusal rather than a default:
    painting the wrong colour is the half-rendered icon this module exists to avoid."""
    if raw is None:
        return None
    text = raw.strip()
    if text.startswith("@"):
        resolved = source.color(text)
        if resolved is None:
            raise _Unsupported(f"_colour: {text} does not resolve to a colour")
        text = resolved.strip()
    return _colour_literal(text)


def _path_data(raw: str) -> str:
    """`android:pathData` re-emitted from a fixed command set and reformatted numbers.

    Android's own `PathParser` reads arc flags as ordinary floats, so vendor-generated path
    data never uses SVG's compressed-flag spelling and a token-level re-emission means exactly
    what the device means. Separating every token with a space is what keeps it that way in a
    browser, which does implement the compressed form.
    """
    tokens: list[str] = []
    index = 0
    while index < len(raw):
        char = raw[index]
        if char in " \t\r\n,":
            index += 1
            continue
        if char in _PATH_COMMANDS:
            tokens.append(char)
            index += 1
            continue
        match = _PATH_NUMBER.match(raw, index)
        if match is None:
            raise _Unsupported(f"_path_data: {char!r} is not path syntax")
        tokens.append(_num(float(match.group())))
        index = match.end()
    if not tokens:
        raise _Unsupported("_path_data: empty path")
    return " ".join(tokens)


# --- elements -------------------------------------------------------------------------------


def _tag(element: Any) -> str:
    """The element's name, or `""` for anything namespaced or non-elemental.

    Only the android drawable vocabulary arrives unnamespaced out of `AXMLPrinter`, so `""`
    is how an `aapt:attr` gradient container, a vendor extension and a comment all reach the
    same refusal instead of being matched on their local name.
    """
    name = element.tag
    if not isinstance(name, str) or name.startswith("{"):
        return ""
    return name


def _attr(element: Any, name: str) -> str | None:
    value = element.get(f"{ANDROID_NS}{name}")
    return value if value else None


def _reject_unknown_attributes(element: Any, allowed: frozenset[str]) -> None:
    for key in element.attrib:
        if not key.startswith(ANDROID_NS) or key[len(ANDROID_NS) :] not in allowed:
            raise _Unsupported(f"_reject_unknown_attributes: <{_tag(element)}> carries {key}")


def _elements(parent: Any) -> list[Any]:
    return [child for child in parent if isinstance(child.tag, str)]


def _path(element: Any, source: DrawableSource) -> str:
    _reject_unknown_attributes(element, _PATH_ATTRIBUTES)
    if _elements(element):
        raise _Unsupported("_path: a path with children carries a gradient or an extension")
    data = _attr(element, "pathData")
    if data is None:
        raise _Unsupported("_path: no pathData")

    parts = [f'd="{_path_data(data)}"']
    fill = _colour(_attr(element, "fillColor"), source)
    if fill is None:
        # SVG defaults an unfilled path to black and a VectorDrawable defaults it to
        # transparent; leaving the attribute off paints a black square.
        parts.append('fill="none"')
    else:
        colour, alpha = fill
        alpha *= _float(_attr(element, "fillAlpha"), 1.0)
        parts.append(f'fill="{colour}"')
        if alpha < 1.0:
            parts.append(f'fill-opacity="{_num(alpha)}"')

    fill_type = _attr(element, "fillType")
    if fill_type is not None:
        rule = _FILL_RULES.get(fill_type)
        if rule is None:
            raise _Unsupported(f"_path: fillType {fill_type!r}")
        parts.append(f'fill-rule="{rule}"')

    stroke = _colour(_attr(element, "strokeColor"), source)
    width = _float(_attr(element, "strokeWidth"), 0.0)
    if stroke is not None and width > 0:
        colour, alpha = stroke
        alpha *= _float(_attr(element, "strokeAlpha"), 1.0)
        parts.append(f'stroke="{colour}"')
        parts.append(f'stroke-width="{_num(width)}"')
        if alpha < 1.0:
            parts.append(f'stroke-opacity="{_num(alpha)}"')
        for name, table, attribute in (
            ("strokeLineCap", _LINE_CAPS, "stroke-linecap"),
            ("strokeLineJoin", _LINE_JOINS, "stroke-linejoin"),
        ):
            raw = _attr(element, name)
            if raw is None:
                continue
            value = table.get(raw)
            if value is None:
                raise _Unsupported(f"_path: {name} {raw!r}")
            parts.append(f'{attribute}="{value}"')
        limit = _attr(element, "strokeMiterLimit")
        if limit is not None:
            parts.append(f'stroke-miterlimit="{_num(_float(limit))}"')
    return f"<path {' '.join(parts)}/>"


def _group(element: Any, source: DrawableSource, depth: int) -> str:
    """A `<group>` as an SVG `<g transform=…>`.

    Android composes the group as `T(pivot + translate) · R · S · T(-pivot)`, and an SVG
    transform list applies left to right in that same order, so the components are emitted in
    exactly that sequence. Any other order moves the artwork.

    `depth` is the element-nesting axis, not the reference axis: this function and
    `_vector_children` call each other, so without it a document nesting groups deeply
    overflows the interpreter stack rather than being refused.
    """
    if depth >= _MAX_ELEMENT_DEPTH:
        raise _Unsupported("_group: element nesting bound reached")
    _reject_unknown_attributes(element, _GROUP_ATTRIBUTES)
    pivot_x = _float(_attr(element, "pivotX"), 0.0)
    pivot_y = _float(_attr(element, "pivotY"), 0.0)
    translate_x = _float(_attr(element, "translateX"), 0.0)
    translate_y = _float(_attr(element, "translateY"), 0.0)
    rotation = _float(_attr(element, "rotation"), 0.0)
    scale_x = _float(_attr(element, "scaleX"), 1.0)
    scale_y = _float(_attr(element, "scaleY"), 1.0)

    parts: list[str] = []
    if pivot_x + translate_x or pivot_y + translate_y:
        parts.append(f"translate({_num(pivot_x + translate_x)} {_num(pivot_y + translate_y)})")
    if rotation:
        parts.append(f"rotate({_num(rotation)})")
    if (scale_x, scale_y) != (1.0, 1.0):
        parts.append(f"scale({_num(scale_x)} {_num(scale_y)})")
    if pivot_x or pivot_y:
        parts.append(f"translate({_num(-pivot_x)} {_num(-pivot_y)})")

    body = _vector_children(element, source, depth + 1)
    transform = f' transform="{" ".join(parts)}"' if parts else ""
    return f"<g{transform}>{body}</g>"


def _vector_children(parent: Any, source: DrawableSource, depth: int) -> str:
    body: list[str] = []
    for child in parent:
        tag = _tag(child)
        if tag == "path":
            body.append(_path(child, source))
        elif tag == "group":
            body.append(_group(child, source, depth))
        else:
            raise _Unsupported(f"_vector_children: <{tag or child.tag}> inside a vector")
    return "".join(body)


def _vector(element: Any, source: DrawableSource) -> tuple[str, float, float]:
    """`(body, viewport width, viewport height)`."""
    _reject_unknown_attributes(element, _VECTOR_ATTRIBUTES)
    width = _float(_attr(element, "viewportWidth"))
    height = _float(_attr(element, "viewportHeight"))
    if width <= 0 or height <= 0:
        raise _Unsupported("_vector: a viewport with no area")
    body = _vector_children(element, source, 0)
    alpha = _float(_attr(element, "alpha"), 1.0)
    if alpha < 1.0:
        body = f'<g opacity="{_num(alpha)}">{body}</g>'
    return body, width, height


def _rect(fill: tuple[str, float], box: _Box) -> str:
    colour, alpha = fill
    parts = [
        f'x="{_num(box.x)}"',
        f'y="{_num(box.y)}"',
        f'width="{_num(box.width)}"',
        f'height="{_num(box.height)}"',
        f'fill="{colour}"',
    ]
    if alpha < 1.0:
        parts.append(f'fill-opacity="{_num(alpha)}"')
    return f"<rect {' '.join(parts)}/>"


def _dimension(raw: str | None, extent: float) -> float:
    """An inset, either a percentage of the box it sits in or a plain dimension."""
    if raw is None:
        return 0.0
    text = raw.strip()
    if text.endswith("%"):
        return _float(text[:-1]) / 100.0 * extent
    return _float(text)


def _inset(element: Any, source: DrawableSource, box: _Box, depth: int) -> str:
    _reject_unknown_attributes(element, _INSET_ATTRIBUTES)
    default = _attr(element, "inset")
    left = _dimension(_attr(element, "insetLeft") or default, box.width)
    right = _dimension(_attr(element, "insetRight") or default, box.width)
    top = _dimension(_attr(element, "insetTop") or default, box.height)
    bottom = _dimension(_attr(element, "insetBottom") or default, box.height)
    return _child(element, source, _shrink(box, left, top, right, bottom), depth)


def _shrink(box: _Box, left: float, top: float, right: float, bottom: float) -> _Box:
    inner = _Box(box.x + left, box.y + top, box.width - left - right, box.height - top - bottom)
    if inner.width <= 0 or inner.height <= 0:
        raise _Unsupported("_shrink: an inset left nothing to paint")
    return inner


def _layer_list(element: Any, source: DrawableSource, box: _Box, depth: int) -> str:
    _reject_unknown_attributes(element, frozenset())
    fragments: list[str] = []
    for child in _elements(element):
        if _tag(child) != "item":
            raise _Unsupported(f"_layer_list: <{_tag(child)}> is not an item")
        _reject_unknown_attributes(child, _ITEM_ATTRIBUTES)
        inner = _shrink(
            box,
            _dimension(_attr(child, "left"), box.width),
            _dimension(_attr(child, "top"), box.height),
            _dimension(_attr(child, "right"), box.width),
            _dimension(_attr(child, "bottom"), box.height),
        )
        fragments.append(_child(child, source, inner, depth))
    if not fragments:
        raise _Unsupported("_layer_list: no items")
    return "".join(fragments)


def _fragment(element: Any, source: DrawableSource, box: _Box, depth: int) -> str:
    if depth > _MAX_DRAWABLE_DEPTH:
        raise _Unsupported("_fragment: drawable nesting bound reached")
    tag = _tag(element)
    if tag == "vector":
        body, width, height = _vector(element, source)
        return (
            f'<svg x="{_num(box.x)}" y="{_num(box.y)}" width="{_num(box.width)}" '
            f'height="{_num(box.height)}" viewBox="0 0 {_num(width)} {_num(height)}" '
            f'preserveAspectRatio="xMidYMid meet">{body}</svg>'
        )
    if tag == "inset":
        return _inset(element, source, box, depth + 1)
    if tag == "layer-list":
        return _layer_list(element, source, box, depth + 1)
    if tag == "color":
        _reject_unknown_attributes(element, frozenset({"color"}))
        fill = _colour(_attr(element, "color"), source)
        if fill is None:
            raise _Unsupported("_fragment: <color> with no colour")
        return _rect(fill, box)
    raise _Unsupported(f"_fragment: <{tag or element.tag}> is outside the support set")


def _child(element: Any, source: DrawableSource, box: _Box, depth: int) -> str:
    """The drawable an `android:drawable` reference or a single inline child names."""
    if depth > _MAX_DRAWABLE_DEPTH:
        raise _Unsupported("_child: drawable nesting bound reached")
    ref = _attr(element, "drawable")
    if ref is not None:
        colour = source.color(ref)
        if colour is not None:
            return _rect(_colour_literal(colour), box)
        resolved = source.element(ref)
        if resolved is None:
            raise _Unsupported(f"_child: {ref} is neither a colour nor an XML drawable")
        return _fragment(resolved, source, box, depth + 1)
    children = _elements(element)
    if len(children) != 1:
        raise _Unsupported(f"_child: <{_tag(element)}> holds {len(children)} drawables")
    return _fragment(children[0], source, box, depth + 1)


def _document(*, view_box: str, width: float, height: float, body: str) -> str:
    return (
        f'<svg xmlns="{_SVG_NS}" viewBox="{view_box}" '
        f'width="{_num(width)}" height="{_num(height)}">{body}</svg>'
    )


def _adaptive_icon(element: Any, source: DrawableSource) -> str:
    """Background then foreground, cropped to the visible square and masked to a circle.

    Both layers are required: a foreground floating on nothing because its background could
    not be resolved is exactly the half-rendered icon the monogram beats.
    """
    _reject_unknown_attributes(element, frozenset())
    canvas = _Box(0.0, 0.0, _ADAPTIVE_CANVAS, _ADAPTIVE_CANVAS)
    layers: list[str] = []
    for name in ("background", "foreground"):
        layer = element.find(name)
        if layer is None:
            raise _Unsupported(f"_adaptive_icon: no <{name}>")
        _reject_unknown_attributes(layer, _LAYER_ATTRIBUTES)
        layers.append(_child(layer, source, canvas, 1))

    visible = _ADAPTIVE_CANVAS - 2 * _ADAPTIVE_BLEED
    centre = _ADAPTIVE_CANVAS / 2
    return _document(
        view_box=(
            f"{_num(_ADAPTIVE_BLEED)} {_num(_ADAPTIVE_BLEED)} {_num(visible)} {_num(visible)}"
        ),
        width=visible,
        height=visible,
        body=(
            f'<defs><clipPath id="{_MASK_ID}"><circle cx="{_num(centre)}" '
            f'cy="{_num(centre)}" r="{_num(visible / 2)}"/></clipPath></defs>'
            f'<g clip-path="url(#{_MASK_ID})">{"".join(layers)}</g>'
        ),
    )


def svg_from_drawable(root: Any, source: DrawableSource) -> str | None:
    """A decoded `<vector>` or `<adaptive-icon>` as a complete SVG document, or None.

    Pure over `(root, source)`: hand it the same drawable twice and it returns the same bytes,
    which is what makes a re-scan's icon column idempotent.

    **Never raises.** `_Unsupported` is this module's own refusal and is expected; anything
    else is a defect or a shape nobody has met, and either way the answer a caller can act on
    is the same one — no icon. A bound closes the escape known today; the broad catch is what
    closes the ones nobody has thought of, and the traceback is logged in full first so an
    unexpected escape is visible rather than absorbed.
    """
    try:
        tag = _tag(root)
        if tag == "vector":
            body, width, height = _vector(root, source)
            return _document(
                view_box=f"0 0 {_num(width)} {_num(height)}",
                width=width,
                height=height,
                body=body,
            )
        if tag == "adaptive-icon":
            return _adaptive_icon(root, source)
        raise _Unsupported(f"svg_from_drawable: <{tag or root.tag}> is not an icon drawable")
    except _Unsupported as exc:
        logger.debug("icon refused: %s", exc)
        return None
    except Exception:
        logger.warning("icon renderer failed on a drawable it could not refuse", exc_info=True)
        return None


# --- the APK ---------------------------------------------------------------------------------


class ApkDrawables:
    """`DrawableSource` over one opened APK.

    androguard raises assorted types out of a resource table it cannot walk — the input is a
    file from a vendor firmware image, the same posture `facts._certificate_names` documents.
    Here every one of them is a refusal rather than a failure: an icon is decoration, and a
    package must not lose its twenty other facts because its launcher graphic is malformed.
    """

    def __init__(self, apk: APK, *, max_dpi: int = ICON_MAX_DPI):
        self._apk = apk
        self._max_dpi = max_dpi
        try:
            self._resources = apk.get_android_resources()
        except Exception:
            logger.debug("icon: resource table unreadable", exc_info=True)
            self._resources = None

    def color(self, ref: str) -> str | None:
        value = self._value(ref)
        return value if value is not None and value.startswith("#") else None

    def path(self, ref: str) -> str | None:
        """The zip member a reference resolves to, or None when it resolves to anything else
        (a colour, a theme attribute, nothing at all)."""
        value = self._value(ref)
        return value if value is not None and value.startswith("res/") else None

    def element(self, ref: str) -> Any | None:
        member = self.path(ref)
        if member is None or not member.endswith(".xml"):
            return None
        try:
            data = self._apk.get_file(member)
        except Exception:
            logger.debug("icon: %s is not in the archive", member, exc_info=True)
            return None
        if not data.startswith(_AXML_MAGIC) or len(data) > MAX_DRAWABLE_BYTES:
            return None
        try:
            return AXMLPrinter(data).get_xml_obj()
        except Exception:
            logger.debug("icon: %s did not decode as binary XML", member, exc_info=True)
            return None

    def _value(self, ref: str) -> str | None:
        """A reference resolved to its value, following a bounded chain of references."""
        if self._resources is None:
            return None
        seen: set[str] = set()
        current = ref
        for _ in range(_MAX_REF_HOPS):
            if not current.startswith("@") or current in seen:
                return None
            seen.add(current)
            resolved = self._resolve_once(current)
            if resolved is None:
                return None
            if not resolved.startswith("@"):
                return resolved
            current = resolved
        return None

    def _resolve_once(self, ref: str) -> str | None:
        try:
            resource_id = int(ref[1:].split(":")[-1], 16)
        except ValueError:
            return None
        try:
            candidates = self._resources.get_resolved_res_configs(resource_id)
        except Exception:
            logger.debug("icon: %s could not be resolved", ref, exc_info=True)
            return None
        pairs: list[tuple[int, str]] = []
        for config, value in candidates:
            try:
                pairs.append((int(config.get_density()), str(value)))
            except Exception:
                logger.debug("icon: %s has a candidate with no density", ref, exc_info=True)
        if not pairs:
            return None
        # The densest variant that is still worth its bytes; when every variant is denser than
        # the cap (an `anydpi` adaptive icon is 65534), the least dense of those.
        usable = [pair for pair in pairs if pair[0] <= self._max_dpi]
        return max(usable)[1] if usable else min(pairs)[1]


def _raster_mime(data: bytes) -> str | None:
    """The mime the bytes themselves declare. Never the member's extension: a `.png` member
    out of a downloaded image is whatever the archive put there."""
    if data.startswith(_PNG_MAGIC):
        return MIME_PNG
    if data.startswith(_JPEG_MAGIC):
        return MIME_JPEG
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return MIME_WEBP
    return None


def extract_icon(apk: APK, *, origin: str = "an unnamed apk") -> tuple[bytes, str] | None:
    """`(bytes, mime)` for one opened APK's launcher icon, or None when it has none this
    module can render.

    **Total: this never raises, whatever the APK contains.** That is a blast-radius decision,
    not tidiness. `facts.parse_apk` calls this while building twenty other signals, and
    `stages.parse_device_apks` catches `ApkParseError` alone — so an escape from here does not
    cost one APK its facts, it kills the whole `extract_facts` stage and discards every other
    package on the device. An icon is decoration; the two honest answers were "no icon plus a
    logged traceback" and "an `ApkParseError` the failure budget can see", and this is the
    first because the second still throws away a package's twenty good facts over its launcher
    graphic, and a vendor-wide drawable style would then fail whole devices through
    `max_apk_parse_failure_ratio` while every manifest parsed perfectly. Nothing is swallowed:
    the full traceback is logged before the refusal.

    `origin` names the APK in that log, since nothing else in the message identifies it.

    Blocking and CPU-bound, like the rest of `facts.parse_apk`, which is its only caller.
    """
    try:
        return _extract_icon(apk)
    except Exception:
        logger.warning("icon: %s yielded no icon, unexpected failure", origin, exc_info=True)
        return None


def _extract_icon(apk: APK) -> tuple[bytes, str] | None:
    member = apk.get_app_icon(max_dpi=ICON_MAX_DPI)
    if not member:
        return None

    data = apk.get_file(member)
    mime = _raster_mime(data)
    if mime is not None:
        return (data, mime) if len(data) <= MAX_ICON_BYTES else None
    if not data.startswith(_AXML_MAGIC) or len(data) > MAX_DRAWABLE_BYTES:
        return None

    svg = svg_from_drawable(AXMLPrinter(data).get_xml_obj(), ApkDrawables(apk))
    if svg is None:
        return None
    blob = svg.encode("utf-8")
    return (blob, MIME_SVG) if len(blob) <= MAX_ICON_BYTES else None
