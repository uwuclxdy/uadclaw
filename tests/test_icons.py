"""Icon extraction: the raster gate, the vector/adaptive renderer, and the route that serves
the result.

The renderer is exercised through `svg_from_drawable`, which takes a parsed element rather
than a binary AXML blob. That is the whole interesting surface — every refusal, every value
that reaches the output — and testing it this way keeps a hand-built AXML fixture (which would
test the fixture rather than androguard) out of the suite. The decode itself is one androguard
call, proven against the 312-APK Pixel tree by the backfill rather than here.

The security property these tests pin: **no string out of an APK is ever copied into the
emitted SVG**. Every value is parsed into a number, a colour or a fixed enum first, and
anything that does not parse refuses the whole icon.
"""

from datetime import UTC, datetime

import pytest
import sqlalchemy.exc
from lxml import etree
from sqlalchemy import event

from test_facts import make_facts, store
from uadclaw import icons as icons_module
from uadclaw.corpusstore import load_corpus
from uadclaw.icons import (
    ICON_MAX_DPI,
    ICON_MIMES,
    MAX_DRAWABLE_BYTES,
    MAX_ICON_BYTES,
    MIME_JPEG,
    MIME_PNG,
    MIME_SVG,
    MIME_WEBP,
    ApkDrawables,
    extract_icon,
    svg_from_drawable,
)
from uadclaw.models import PackageFact

ANDROID_NS = "http://schemas.android.com/apk/res/android"
AAPT_NS = "http://schemas.android.com/aapt"

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"payload"
WEBP_BYTES = b"RIFF\x10\x00\x00\x00WEBPVP8 "
JPEG_BYTES = b"\xff\xd8\xff" + b"payload"
PASSWORD = "test-only-admin-password"
NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


async def put_fact(session_factory, package: str, **icon) -> None:
    """A `package_facts` row written straight to the table.

    The route's guards are about row SHAPES, and the merge cannot produce all of them — an
    empty-but-not-null `icon_bytes` is filtered out by `merge_observations` before it could
    ever reach the merged row. Writing the row directly is what lets each guard be pinned on
    its own rather than as "at least one of two fired".
    """
    async with session_factory() as session, session.begin():
        session.add(
            PackageFact(
                package=package,
                device_count=1,
                devices=["pixel:oriole"],
                first_seen_at=NOW,
                last_seen_at=NOW,
                label="Example",
                partitions=["system"],
                **icon,
            )
        )


def drawable(markup: str):
    """A drawable in the shape androguard's `AXMLPrinter` hands over: unnamespaced tags,
    `android:`-namespaced attributes."""
    wrapper = etree.fromstring(
        f'<wrap xmlns:android="{ANDROID_NS}" xmlns:aapt="{AAPT_NS}">{markup.strip()}</wrap>'
    )
    return wrapper[0]


class FakeSource:
    """A `DrawableSource` whose whole resource table is two dicts."""

    def __init__(
        self, colors: dict[str, str] | None = None, elements: dict[str, str] | None = None
    ):
        self._colors = colors or {}
        self._elements = elements or {}

    def color(self, ref: str) -> str | None:
        return self._colors.get(ref)

    def element(self, ref: str):
        markup = self._elements.get(ref)
        return drawable(markup) if markup is not None else None


def render(markup: str, **kwargs) -> str | None:
    return svg_from_drawable(drawable(markup), FakeSource(**kwargs))


def test_the_stored_mime_strings_are_the_literals_a_browser_has_to_read():
    """Every other assertion here spells a mime through its constant, so a typo INSIDE the
    constant is invisible to all of them — measured: renaming `MIME_PNG` left the whole suite
    green. The browser reads the literal, and so does the dashboard's icon contract."""
    servable = {"image/png", "image/webp", "image/jpeg", "image/svg+xml"}

    assert (MIME_PNG, MIME_WEBP, MIME_JPEG, MIME_SVG) == (
        "image/png",
        "image/webp",
        "image/jpeg",
        "image/svg+xml",
    )
    assert servable == ICON_MIMES


def test_the_two_size_bounds_are_the_numbers_they_were_measured_against():
    """Both are built from a measurement, and both are invisible to any assertion that builds
    its fixture out of the constant. A retune belongs in review, not only in behaviour."""
    assert MAX_ICON_BYTES == 64 * 1024
    assert MAX_DRAWABLE_BYTES == 256 * 1024


# --- the vector renderer ----------------------------------------------------------------------


def test_a_vector_becomes_an_svg_with_the_androids_viewport_as_its_viewbox():
    svg = render("""
        <vector android:viewportWidth="24" android:viewportHeight="24">
          <path android:pathData="M0,0h24v24h-24z" android:fillColor="#FFFF0000"/>
        </vector>
    """)

    assert svg is not None
    assert 'viewBox="0 0 24 24"' in svg
    assert 'd="M 0 0 h 24 v 24 h -24 z"' in svg
    assert 'fill="#ff0000"' in svg


def test_path_data_that_is_not_path_syntax_refuses_the_whole_icon():
    """The one attribute that is itself a mini-language. It is re-emitted token by token from
    a fixed command set and parsed numbers, so anything else cannot reach the document."""
    svg = render("""
        <vector android:viewportWidth="24" android:viewportHeight="24">
          <path android:pathData="M0,0&quot;/&gt;&lt;script&gt;alert(1)&lt;/script&gt;"
                android:fillColor="#FFFF0000"/>
        </vector>
    """)

    assert svg is None


def test_a_fill_colour_resource_reference_is_resolved_through_the_apk():
    svg = render(
        """
        <vector android:viewportWidth="24" android:viewportHeight="24">
          <path android:pathData="M0,0h24v24h-24z" android:fillColor="@7F070000"/>
        </vector>
        """,
        colors={"@7F070000": "#ff0000ff"},
    )

    assert svg is not None
    assert 'fill="#0000ff"' in svg


def test_a_fill_colour_reference_that_resolves_to_nothing_refuses():
    """Not "render it black": a wrong colour is a half-rendered icon, and the monogram is the
    honest fallback."""
    svg = render("""
        <vector android:viewportWidth="24" android:viewportHeight="24">
          <path android:pathData="M0,0h24v24h-24z" android:fillColor="@7F070000"/>
        </vector>
    """)

    assert svg is None


def test_an_absent_fill_colour_renders_as_no_fill():
    """SVG defaults an unfilled path to black; a VectorDrawable defaults it to transparent.
    Leaving the attribute off would silently paint a black square."""
    svg = render("""
        <vector android:viewportWidth="24" android:viewportHeight="24">
          <path android:pathData="M0,0h24v24h-24z" android:strokeColor="#FF00FF00"
                android:strokeWidth="2"/>
        </vector>
    """)

    assert svg is not None
    assert 'fill="none"' in svg
    assert 'stroke="#00ff00"' in svg
    assert 'stroke-width="2"' in svg


def test_fill_alpha_becomes_fill_opacity():
    svg = render("""
        <vector android:viewportWidth="24" android:viewportHeight="24">
          <path android:pathData="M0,0h24v24h-24z" android:fillColor="#FFFF0000"
                android:fillAlpha="0.2"/>
        </vector>
    """)

    assert svg is not None
    assert 'fill-opacity="0.2"' in svg


def test_an_eight_digit_fill_colour_keeps_its_alpha():
    svg = render("""
        <vector android:viewportWidth="24" android:viewportHeight="24">
          <path android:pathData="M0,0h24v24h-24z" android:fillColor="#80FF0000"/>
        </vector>
    """)

    assert svg is not None
    assert 'fill="#ff0000"' in svg
    assert 'fill-opacity="0.502"' in svg


def test_a_group_transform_is_emitted_in_the_order_android_applies_it():
    """Android composes a group as translate(pivot+translate) rotate scale translate(-pivot).
    Emitting the components in any other order moves the artwork."""
    svg = render("""
        <vector android:viewportWidth="24" android:viewportHeight="24">
          <group android:translateX="2" android:translateY="3" android:pivotX="12"
                 android:pivotY="12" android:rotation="90" android:scaleX="2"
                 android:scaleY="0.5">
            <path android:pathData="M0,0h24v24h-24z" android:fillColor="#FFFF0000"/>
          </group>
        </vector>
    """)

    assert svg is not None
    assert 'transform="translate(14 15) rotate(90) scale(2 0.5) translate(-12 -12)"' in svg


def test_an_enum_attribute_arriving_as_its_integer_is_still_understood():
    """AXML stores an enum as its integer and `AXMLPrinter` hands that integer straight over,
    so `android:fillType="evenOdd"` in the source arrives here as `"1"`. Coding only the
    source spelling refuses every icon that declares a fill rule or a line cap — measured at 2
    of the 36 XML drawables on the Pixel corpus."""
    svg = render("""
        <vector android:viewportWidth="24" android:viewportHeight="24">
          <path android:pathData="M0,0h24v24h-24z" android:fillColor="#FFFF0000"
                android:fillType="1" android:strokeColor="#FF00FF00" android:strokeWidth="2"
                android:strokeLineCap="1" android:strokeLineJoin="2"/>
        </vector>
    """)

    assert svg is not None
    assert 'fill-rule="evenodd"' in svg
    assert 'stroke-linecap="round"' in svg
    assert 'stroke-linejoin="bevel"' in svg


def test_an_enum_integer_outside_its_range_refuses():
    svg = render("""
        <vector android:viewportWidth="24" android:viewportHeight="24">
          <path android:pathData="M0,0h24v24h-24z" android:fillColor="#FFFF0000"
                android:fillType="7"/>
        </vector>
    """)

    assert svg is None


def test_a_gradient_declared_through_aapt_attr_refuses():
    """`aapt:attr` is how a gradient is inlined into a vector. Dropping the element and
    keeping the path paints a flat shape the icon never had."""
    svg = render("""
        <vector android:viewportWidth="24" android:viewportHeight="24">
          <path android:pathData="M0,0h24v24h-24z">
            <aapt:attr name="android:fillColor"><gradient/></aapt:attr>
          </path>
        </vector>
    """)

    assert svg is None


def test_a_clip_path_refuses():
    """3 of the 36 XML drawables on the Pixel corpus. Rendering the paths and dropping the
    clip shows artwork the device never draws."""
    svg = render("""
        <vector android:viewportWidth="24" android:viewportHeight="24">
          <clip-path android:pathData="M0,0h12v24h-12z"/>
          <path android:pathData="M0,0h24v24h-24z" android:fillColor="#FFFF0000"/>
        </vector>
    """)

    assert svg is None


def test_an_android_attribute_the_renderer_does_not_implement_refuses():
    """The support set is stated as an allowlist rather than grown as a blocklist: a path
    that trims itself renders as a different shape, and every future attribute is unknown."""
    svg = render("""
        <vector android:viewportWidth="24" android:viewportHeight="24">
          <path android:pathData="M0,0h24v24h-24z" android:fillColor="#FFFF0000"
                android:trimPathEnd="0.5"/>
        </vector>
    """)

    assert svg is None


def test_a_vector_with_no_viewport_refuses():
    svg = render("""
        <vector>
          <path android:pathData="M0,0h24v24h-24z" android:fillColor="#FFFF0000"/>
        </vector>
    """)

    assert svg is None


def test_nothing_out_of_the_drawable_reaches_the_document_as_a_string():
    """`android:name` is the only free-text attribute on a supported element, and it is read
    and dropped rather than copied out."""
    svg = render("""
        <vector android:name="&quot;&gt;&lt;script&gt;" android:viewportWidth="24"
                android:viewportHeight="24">
          <group android:name="also&lt;evil&gt;">
            <path android:name="third&lt;evil&gt;" android:pathData="M0,0h24v24h-24z"
                  android:fillColor="#FFFF0000"/>
          </group>
        </vector>
    """)

    assert svg is not None
    assert "script" not in svg
    assert "evil" not in svg


# --- adaptive icons ---------------------------------------------------------------------------


ADAPTIVE = """
<adaptive-icon>
  <background android:drawable="@7F110001"/>
  <foreground>
    <vector android:viewportWidth="72" android:viewportHeight="72">
      <path android:pathData="M0,0h72v72h-72z" android:fillColor="#FF00FF00"/>
    </vector>
  </foreground>
</adaptive-icon>
"""


def test_an_adaptive_icon_paints_its_background_before_its_foreground():
    svg = render(ADAPTIVE, colors={"@7F110001": "#ffff0000"})

    assert svg is not None
    assert svg.index("#ff0000") < svg.index("#00ff00")


def test_an_adaptive_icon_is_cropped_and_masked_the_way_a_launcher_shows_it():
    """AOSP reserves the outer 18dp of the 108dp canvas for the mask, leaving 72dp visible.
    Rendering the whole canvas square shows bleed art no phone ever displays."""
    svg = render(ADAPTIVE, colors={"@7F110001": "#ffff0000"})

    assert svg is not None
    assert 'viewBox="18 18 72 72"' in svg
    assert '<circle cx="54" cy="54" r="36"/>' in svg


def test_an_adaptive_icon_whose_background_is_a_raster_refuses():
    """A layer that resolves to a PNG is the "raster nested somewhere unexpected" case: half
    the icon would render and half would not."""
    assert render(ADAPTIVE) is None


def test_an_adaptive_icon_missing_a_layer_refuses():
    svg = render("""
        <adaptive-icon>
          <foreground>
            <vector android:viewportWidth="72" android:viewportHeight="72">
              <path android:pathData="M0,0h72v72h-72z" android:fillColor="#FF00FF00"/>
            </vector>
          </foreground>
        </adaptive-icon>
    """)

    assert svg is None


def test_an_inset_foreground_is_scaled_into_the_canvas_it_declares():
    """`<inset>` is how a legacy square icon is fitted into an adaptive foreground; ignoring
    the inset would blow the artwork up to the full canvas."""
    svg = render(
        """
        <adaptive-icon>
          <background android:drawable="@bg"/>
          <foreground><inset android:drawable="@fg" android:inset="25%"/></foreground>
        </adaptive-icon>
        """,
        colors={"@bg": "#ffff0000"},
        elements={
            "@fg": """
            <vector android:viewportWidth="48" android:viewportHeight="48">
              <path android:pathData="M0,0h48v48h-48z" android:fillColor="#FF00FF00"/>
            </vector>
            """
        },
    )

    assert svg is not None
    assert 'x="27" y="27" width="54" height="54"' in svg


def test_a_layer_list_stacks_its_items_in_declaration_order():
    svg = render(
        """
        <adaptive-icon>
          <background android:drawable="@bg"/>
          <foreground>
            <layer-list>
              <item android:drawable="@one"/>
              <item android:drawable="@two"/>
            </layer-list>
          </foreground>
        </adaptive-icon>
        """,
        colors={"@bg": "#ffffffff", "@one": "#ff00ff00", "@two": "#ff0000ff"},
    )

    assert svg is not None
    assert svg.index("#00ff00") < svg.index("#0000ff")


def _nested_groups(count: int):
    """A `<vector>` whose `<group>` elements nest `count` deep, around one path."""
    root = etree.fromstring(
        f'<vector xmlns:android="{ANDROID_NS}" android:viewportWidth="24"'
        ' android:viewportHeight="24"/>'
    )
    parent = root
    for _ in range(count):
        parent = etree.SubElement(parent, "group")
    path = etree.SubElement(parent, "path")
    path.set(f"{{{ANDROID_NS}}}pathData", "M0,0h24v24h-24z")
    path.set(f"{{{ANDROID_NS}}}fillColor", "#FFFF0000")
    return root


def test_a_deeply_nested_drawable_refuses_instead_of_overflowing_the_stack():
    """Element nesting is a SEPARATE axis from reference resolution, and the reference bound
    never touched it: `_group` and `_vector_children` call each other. 497 nested `<group>`
    elements reached CPython's recursion limit, and the escape did not cost one APK its
    facts — it killed the whole `extract_facts` stage for the device."""
    assert svg_from_drawable(_nested_groups(600), FakeSource()) is None


def test_element_nesting_is_bounded_exactly_where_it_says_it_is():
    """The pair, so the bound's VALUE is pinned and not merely its existence: a bound loose
    enough to reach the interpreter's own limit passes the refusal half on its own."""
    assert svg_from_drawable(_nested_groups(icons_module._MAX_ELEMENT_DEPTH), FakeSource())
    assert (
        svg_from_drawable(_nested_groups(icons_module._MAX_ELEMENT_DEPTH + 1), FakeSource()) is None
    )


def test_a_resource_table_that_throws_refuses_the_icon_rather_than_the_caller():
    """`svg_from_drawable`'s contract is "refuse and fall back to the monogram", so a
    `DrawableSource` raising has to land as a refusal too. This is the reachable half of the
    widened catch: `ApkDrawables` guards its own androguard calls today, and a future
    androguard raising somewhere it does not is the shape that used to escape."""

    class Hostile:
        def color(self, ref: str):
            raise ValueError("androguard fell over walking the resource table")

        def element(self, ref: str):
            raise ValueError("androguard fell over walking the resource table")

    assert svg_from_drawable(drawable(ADAPTIVE), Hostile()) is None


def _inset_chain(length: int) -> dict[str, str]:
    """`length` `<inset>` drawables, each referencing the next, ending in a vector."""
    chain = {
        f"@d{index}": f'<inset android:drawable="@d{index + 1}" android:inset="1%"/>'
        for index in range(length)
    }
    chain[f"@d{length}"] = """
        <vector android:viewportWidth="48" android:viewportHeight="48">
          <path android:pathData="M0,0h48v48h-48z" android:fillColor="#FF00FF00"/>
        </vector>
    """
    return chain


ADAPTIVE_CHAIN = """
<adaptive-icon>
  <background android:drawable="@bg"/>
  <foreground android:drawable="@d0"/>
</adaptive-icon>
"""


def test_a_reference_chain_is_bounded_exactly_where_it_says_it_is():
    """Three hops render and four refuse at `_MAX_DRAWABLE_DEPTH == 8`, because each `<inset>`
    hop costs two levels. Pins the value: at 64 the four-hop case would render."""
    assert render(ADAPTIVE_CHAIN, colors={"@bg": "#ffffffff"}, elements=_inset_chain(3))
    assert render(ADAPTIVE_CHAIN, colors={"@bg": "#ffffffff"}, elements=_inset_chain(4)) is None


def test_a_drawable_that_references_itself_refuses_instead_of_recursing_forever():
    """A drawable graph out of a downloaded firmware image is untrusted input, so the depth
    bound is the structural answer rather than trusting the vendor's build tools."""
    svg = render(
        """
        <adaptive-icon>
          <background android:drawable="@bg"/>
          <foreground><inset android:drawable="@loop" android:inset="10%"/></foreground>
        </adaptive-icon>
        """,
        colors={"@bg": "#ffffffff"},
        elements={"@loop": '<inset android:drawable="@loop" android:inset="10%"/>'},
    )

    assert svg is None


# --- the APK-facing entry point ---------------------------------------------------------------


class FakeConfig:
    def __init__(self, density: int):
        self._density = density

    def get_density(self) -> int:
        return self._density


class FakeResources:
    """The slice of androguard's `ARSCParser` the resolver actually calls."""

    def __init__(self, table: dict[int, list[tuple[int, str]]]):
        self._table = table

    def get_resolved_res_configs(self, rid: int):
        if rid not in self._table:
            raise KeyError(f"no resource {rid:#x}")
        return [(FakeConfig(density), value) for density, value in self._table[rid]]


class FakeApk:
    def __init__(self, *, icon=None, files=None, resources=None):
        self._icon = icon
        self._files = files or {}
        self._resources = resources

    def get_app_icon(self, max_dpi: int = 65536):
        return self._icon

    def get_file(self, name: str) -> bytes:
        from androguard.core.apk import FileNotPresent

        if name not in self._files:
            raise FileNotPresent(name)
        return self._files[name]

    def get_android_resources(self):
        return self._resources


def test_a_png_icon_comes_back_with_the_mime_its_bytes_declare():
    apk = FakeApk(icon="res/mipmap/ic.png", files={"res/mipmap/ic.png": PNG_BYTES})

    assert extract_icon(apk) == (PNG_BYTES, MIME_PNG)


def test_a_webp_icon_comes_back_with_the_mime_its_bytes_declare():
    apk = FakeApk(icon="res/mipmap/ic.webp", files={"res/mipmap/ic.webp": WEBP_BYTES})

    assert extract_icon(apk) == (WEBP_BYTES, MIME_WEBP)


def test_a_resource_named_png_whose_bytes_are_not_a_png_is_refused():
    """This repo dispatches on the bytes and never on a name: an `.png` member out of a
    downloaded firmware image is whatever the archive says it is."""
    apk = FakeApk(icon="res/mipmap/ic.png", files={"res/mipmap/ic.png": b"<html>hi</html>"})

    assert extract_icon(apk) is None


def test_an_icon_over_the_cap_is_skipped_rather_than_truncated():
    """One APK on the Pixel corpus ships a 69 KB PNG. Half a PNG is a broken image, not a
    smaller one."""
    big = PNG_BYTES + b"\x00" * MAX_ICON_BYTES
    apk = FakeApk(icon="res/mipmap/ic.png", files={"res/mipmap/ic.png": big})

    assert extract_icon(apk) is None


def test_a_jpeg_icon_comes_back_with_the_mime_its_bytes_declare():
    apk = FakeApk(icon="res/mipmap/ic.jpg", files={"res/mipmap/ic.jpg": JPEG_BYTES})

    assert extract_icon(apk) == (JPEG_BYTES, MIME_JPEG)


def test_the_cap_stores_an_icon_of_exactly_its_size_and_refuses_one_byte_more():
    """The boundary from both sides, so the comparison operator is pinned as well as the
    number — and neither fixture is built out of the constant, so tightening the cap to a
    value that refuses every real icon cannot pass."""
    exact = PNG_BYTES + b"\x00" * (65536 - len(PNG_BYTES))
    over = exact + b"\x00"

    assert extract_icon(FakeApk(icon="i.png", files={"i.png": exact})) == (exact, MIME_PNG)
    assert extract_icon(FakeApk(icon="i.png", files={"i.png": over})) is None


def test_an_oversized_xml_drawable_is_refused_before_it_is_decoded(monkeypatch):
    """A bound on the STORED output is not a bound on the intermediate allocation: the decode,
    the render and the encode all happen before the 64 KB output cap is consulted. Asserting
    the decoder was never reached is the discriminating half — a refusal after decoding looks
    identical from the return value."""
    decoded: list[int] = []
    monkeypatch.setattr(icons_module, "AXMLPrinter", lambda data: decoded.append(len(data)))
    huge = b"\x03\x00\x08\x00" + b"\x00" * MAX_DRAWABLE_BYTES
    apk = FakeApk(icon="res/drawable/ic.xml", files={"res/drawable/ic.xml": huge})

    assert extract_icon(apk) is None
    assert decoded == []


def test_a_package_that_declares_no_icon_yields_nothing():
    assert extract_icon(FakeApk()) is None


def test_an_icon_whose_member_is_missing_from_the_zip_yields_nothing():
    assert extract_icon(FakeApk(icon="res/mipmap/ic.png")) is None


def test_extraction_never_raises_whatever_the_apk_does(monkeypatch):
    """`facts.parse_apk` calls this while building twenty other signals and
    `stages.parse_device_apks` catches `ApkParseError` alone, so an escape here does not cost
    one APK its facts — it kills the whole device's scan. A boring exception type on purpose:
    a guard spelled for the one failure that has been seen leaves every other one live."""

    class Hostile:
        def get_app_icon(self, max_dpi: int = 65536):
            raise ValueError("androguard fell over on a vendor resource table")

    assert extract_icon(Hostile(), origin="/system/app/Hostile/Hostile.apk") is None


# --- resource resolution ----------------------------------------------------------------------


def test_the_resolver_picks_the_densest_candidate_at_or_below_the_cap():
    """Above the cap is a bigger file for no visible gain; the densest below it is the best
    the 48px card can show. The 480 candidate is what pins `ICON_MAX_DPI` exactly rather than
    bracketing it: without it a retune anywhere in 320..639 changes which raster every APK on
    the corpus yields, and the byte size of everything stored, with nothing going red."""
    assert ICON_MAX_DPI == 320
    apk = FakeApk(
        resources=FakeResources(
            {
                0x7F110001: [
                    (160, "res/drawable-mdpi/a.png"),
                    (320, "res/drawable-xhdpi/a.png"),
                    (480, "res/drawable-xxhdpi/a.png"),
                    (640, "res/drawable-xxxhdpi/a.png"),
                ]
            }
        )
    )

    assert ApkDrawables(apk).path("@7F110001") == "res/drawable-xhdpi/a.png"


def test_the_resolver_answers_a_colour_resource_as_a_colour():
    apk = FakeApk(resources=FakeResources({0x7F110001: [(0, "#ffff0000")]}))
    source = ApkDrawables(apk)

    assert source.color("@7F110001") == "#ffff0000"
    assert source.element("@7F110001") is None


def test_the_resolver_answers_a_theme_attribute_as_nothing():
    """`?android:01010435` resolves to something no drawable renderer can read, and 1 of the
    32 adaptive icons on the Pixel corpus carries one."""
    apk = FakeApk(resources=FakeResources({0x7F110001: [(0, "?android:01010435")]}))
    source = ApkDrawables(apk)

    assert source.color("@7F110001") is None
    assert source.path("@7F110001") is None


def test_the_resolver_answers_an_unknown_resource_as_nothing():
    apk = FakeApk(resources=FakeResources({}))

    assert ApkDrawables(apk).color("@7F110001") is None


# --- the route --------------------------------------------------------------------------------


async def _login(client) -> None:
    resp = await client.post("/login", json={"password": PASSWORD})
    assert resp.status_code == 204


async def test_the_icon_route_requires_auth(db_env, client):
    resp = await client.get("/icons/com.example.app")

    assert resp.status_code == 401


async def test_the_icon_route_answers_the_stored_bytes_with_their_mime(
    db_env, db_session_factory, client
):
    await store(
        db_session_factory,
        "pixel:oriole",
        "A.1",
        make_facts("com.example.app", icon_bytes=PNG_BYTES, icon_mime=MIME_PNG),
    )
    await _login(client)

    resp = await client.get("/icons/com.example.app")

    assert resp.status_code == 200
    assert resp.content == PNG_BYTES
    # The literal, not the constant: what leaves this route has to be what a browser accepts.
    assert resp.headers["content-type"] == "image/png"


async def test_the_icon_route_blocks_sniffing_and_script_in_an_svg(
    db_env, db_session_factory, client
):
    """An SVG opened directly is a document, not just an `<img>` payload: the CSP is what
    keeps a stored icon from being a script host, and nosniff is what keeps a PNG from being
    re-read as HTML."""
    await store(
        db_session_factory,
        "pixel:oriole",
        "A.1",
        make_facts("com.example.app", icon_bytes=b"<svg/>", icon_mime=MIME_SVG),
    )
    await _login(client)

    resp = await client.get("/icons/com.example.app")

    assert resp.status_code == 200
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in resp.headers["content-security-policy"]
    assert "sandbox" in resp.headers["content-security-policy"]
    # `private` on a cookie-authenticated image is the property worth a header, not the age.
    assert resp.headers["cache-control"].startswith("private")


async def test_the_icon_route_404s_when_the_package_has_no_icon(db_env, db_session_factory, client):
    await store(db_session_factory, "pixel:oriole", "A.1", make_facts("com.example.app"))
    await _login(client)

    resp = await client.get("/icons/com.example.app")

    assert resp.status_code == 404


async def test_the_icon_route_404s_when_the_row_names_a_mime_it_will_not_serve(
    db_env, db_session_factory, client
):
    """The gate that makes echoing the column into `Content-Type` safe. The backfill writes
    that column through a plain UPDATE, so a value `extract_icon` never produced is a path
    that exists rather than a hypothetical."""
    await put_fact(
        db_session_factory, "com.example.app", icon_bytes=b"<html>", icon_mime="text/html"
    )
    await _login(client)

    resp = await client.get("/icons/com.example.app")

    assert resp.status_code == 404


async def test_the_icon_route_404s_when_the_row_names_a_mime_but_carries_no_bytes(
    db_env, db_session_factory, client
):
    """The other half, pinned separately: the previous test leaves both columns null, so
    either guard alone answers it and neither is proven. Empty-but-present bytes is the shape
    the pair constraint permits and the merge cannot produce."""
    await put_fact(db_session_factory, "com.example.app", icon_bytes=b"", icon_mime=MIME_PNG)
    await _login(client)

    resp = await client.get("/icons/com.example.app")

    assert resp.status_code == 404


async def test_the_icon_route_404s_rather_than_serving_a_row_over_the_cap(
    db_env, db_session_factory, client
):
    """The write path caps this, so a row over it means something else wrote the column. The
    bound belongs to the response as much as to the column."""
    await put_fact(
        db_session_factory,
        "com.example.app",
        icon_bytes=PNG_BYTES + b"\x00" * MAX_ICON_BYTES,
        icon_mime=MIME_PNG,
    )
    await _login(client)

    resp = await client.get("/icons/com.example.app")

    assert resp.status_code == 404


async def test_a_row_carrying_one_icon_column_without_the_other_cannot_be_stored(
    db_env, db_session_factory
):
    """Every screen decides whether to emit an `<img>` from `icon_mime` while this route
    serves `icon_bytes`, so a half-populated row renders a broken-image glyph in place of the
    monogram that is the designed answer. Rejected by the schema rather than promised in a
    docstring, which is what makes a partial backfill or a manual fix-up unable to create it."""
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        await put_fact(db_session_factory, "com.example.app", icon_mime=MIME_PNG)


async def test_the_corpus_query_does_not_carry_the_icon_bytes_it_never_reads(
    db_env, db_session_factory
):
    """`corpusstore.load_corpus` pulls the whole corpus for the corpus-graph stage, which
    reads twenty signals and no artwork. Measured against a throwaway database holding the
    real 312-package Pixel corpus, the column is 283,597 of 822,206 bytes of row payload —
    34.5%, transferred for nothing.

    The SQL `load_corpus` actually emitted, captured off the engine — not a statement this
    test builds for itself, which would assert the query against a copy of itself and stay
    green with the `defer` deleted. Pinning the observable also survives a refactor to
    `load_only` or a hand-written select, and still fails if the bytes come back.
    """
    await store(
        db_session_factory,
        "pixel:oriole",
        "A.1",
        make_facts("com.example.app", icon_bytes=PNG_BYTES, icon_mime=MIME_PNG),
    )
    emitted: list[str] = []

    async with db_session_factory() as session:
        engine = session.bind.sync_engine

        def capture(conn, cursor, statement, parameters, context, executemany):
            emitted.append(statement)

        event.listen(engine, "before_cursor_execute", capture)
        try:
            corpus = await load_corpus(session)
        finally:
            event.remove(engine, "before_cursor_execute", capture)

    reads = [statement for statement in emitted if "package_facts" in statement]

    assert [item.package for item in corpus] == ["com.example.app"]
    assert len(reads) == 1, f"load_corpus emitted {len(reads)} reads of package_facts"
    assert "icon_bytes" not in reads[0]
    assert "icon_mime" in reads[0], "only the bytes are deferred; has_icon reads the mime"


async def test_a_long_package_name_still_reaches_its_icon(db_env, db_session_factory, client):
    """The length bound exists to refuse what cannot match a row, not to shorten what can.
    Tightening it 404s real packages, and a 404 renders the monogram — visually identical to
    "this package has no icon", so nothing on the screen would ever say why.

    Not hypothetical: 3 of the 227 real names in `docs/research/emulator-a16-packages.tsv`
    are over 64 characters and the longest is 81, because `auto_generated_rro_product__` is a
    28-character SUFFIX on an already-long name."""
    package = "com.example." + "a" * 190
    await store(
        db_session_factory,
        "pixel:oriole",
        "A.1",
        make_facts(package, icon_bytes=PNG_BYTES, icon_mime=MIME_PNG),
    )
    await _login(client)

    resp = await client.get(f"/icons/{package}")

    assert resp.status_code == 200
    assert resp.content == PNG_BYTES


async def test_the_icon_route_404s_for_a_package_nobody_ever_scanned(
    db_env, db_session_factory, client
):
    await _login(client)

    resp = await client.get("/icons/com.example.nothing")

    assert resp.status_code == 404


async def test_a_package_name_longer_than_the_column_is_refused_before_the_query(client):
    """Bounded at the boundary rather than handed to the query — and pinned WITHOUT `db_env`,
    so `POSTGRES_HOST` is the unreachable default. A 404 here can only come from the length
    bound; letting the query run instead answers 503. With a database reachable the two are
    indistinguishable, because a name over 255 can never match a row in a `String(255)`
    column, so that arrangement pins nothing."""
    await _login(client)

    resp = await client.get("/icons/" + "a" * 256)

    assert resp.status_code == 404


async def test_a_database_that_is_not_answering_reads_as_unavailable_not_as_no_icon(client):
    """No `db_env`. A 404 would tell the browser this package HAS no icon, which is a
    different fact from "nobody could look"."""
    await _login(client)

    resp = await client.get("/icons/com.example.app")

    assert resp.status_code == 503


@pytest.mark.parametrize("mime", [MIME_PNG, MIME_WEBP, MIME_JPEG, MIME_SVG])
async def test_every_mime_the_extractor_can_store_is_servable(
    db_env, db_session_factory, client, mime
):
    await store(
        db_session_factory,
        "pixel:oriole",
        "A.1",
        make_facts("com.example.app", icon_bytes=PNG_BYTES, icon_mime=mime),
    )
    await _login(client)

    resp = await client.get("/icons/com.example.app")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == mime
