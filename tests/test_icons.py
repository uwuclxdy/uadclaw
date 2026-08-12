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

import pytest
from lxml import etree

from test_facts import make_facts, store
from uadclaw.icons import (
    ICON_MIMES,
    MAX_ICON_BYTES,
    MIME_PNG,
    MIME_SVG,
    MIME_WEBP,
    ApkDrawables,
    extract_icon,
    svg_from_drawable,
)

ANDROID_NS = "http://schemas.android.com/apk/res/android"
AAPT_NS = "http://schemas.android.com/aapt"

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"payload"
WEBP_BYTES = b"RIFF\x10\x00\x00\x00WEBPVP8 "
PASSWORD = "test-only-admin-password"


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

    assert (MIME_PNG, MIME_WEBP, MIME_SVG) == ("image/png", "image/webp", "image/svg+xml")
    assert servable == ICON_MIMES


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


def test_a_package_that_declares_no_icon_yields_nothing():
    assert extract_icon(FakeApk()) is None


def test_an_icon_whose_member_is_missing_from_the_zip_yields_nothing():
    assert extract_icon(FakeApk(icon="res/mipmap/ic.png")) is None


# --- resource resolution ----------------------------------------------------------------------


def test_the_resolver_picks_the_densest_candidate_at_or_below_the_cap():
    """Above the cap is a bigger file for no visible gain; the densest below it is the best
    the 48px card can show."""
    apk = FakeApk(
        resources=FakeResources(
            {
                0x7F110001: [
                    (160, "res/drawable-mdpi/a.png"),
                    (320, "res/drawable-xhdpi/a.png"),
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


async def test_the_icon_route_404s_when_the_package_has_no_icon(db_env, db_session_factory, client):
    await store(db_session_factory, "pixel:oriole", "A.1", make_facts("com.example.app"))
    await _login(client)

    resp = await client.get("/icons/com.example.app")

    assert resp.status_code == 404


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


@pytest.mark.parametrize("mime", [MIME_PNG, MIME_WEBP, MIME_SVG])
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
