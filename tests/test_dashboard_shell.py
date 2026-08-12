"""The dashboard shell: the static exemption, the login form, and how a rejection reaches
each kind of caller.

The shell is where task 16's three screen lanes plug in, so what is pinned here is the
contract they inherit rather than anything any one screen does. Two of these are security
assertions and are written as pairs on purpose: an exemption test that only proves the
allowed case passes is a test that would stay green if the rule allowed everything.
"""

import pytest

from uadclaw import web
from uadclaw.auth import PUBLIC_PATHS, PUBLIC_PREFIXES, is_public, login_url, safe_next

PASSWORD = "test-only-admin-password"


async def _login(client) -> None:
    resp = await client.post("/login", json={"password": PASSWORD})
    assert resp.status_code == 204


# --- the static exemption ---------------------------------------------------------------


async def test_static_assets_are_served_without_a_session(client):
    """The login page has to be styled before anyone can log in."""
    resp = await client.get("/static/ui.css")
    assert resp.status_code == 200
    assert "--accent" in resp.text


@pytest.mark.parametrize(
    "path",
    ["/statistics", "/staticky", "/static", "/staticfiles/secret", "/stats", "/jobs"],
)
def test_a_path_merely_starting_like_the_exempt_prefix_is_not_public(path):
    """The other half of the exemption, and the reason `PUBLIC_PREFIXES` keeps its trailing
    slash: `"/static"` as a prefix exempts every one of these.

    Asserted against `is_public` rather than through the client on purpose. Routed, most of
    these 404 whether or not they are exempt, so a status-code assertion would pass for a
    reason that has nothing to do with the rule under test — it stays green with the
    exemption wide open. This asks the function that actually makes the decision.
    """
    assert is_public(path) is False


@pytest.mark.parametrize("path", ["/static/ui.css", "/static/vendor/x.js", "/health", "/login"])
def test_the_public_surface_is_public(path):
    """The positive leg. Without it the test above is satisfied by a rule that exempts
    nothing at all, including the stylesheet the login page cannot render without."""
    assert is_public(path) is True


async def test_a_protected_route_that_shares_the_prefixs_opening_letters_still_401s(client):
    """`/stats` reads the database and sorts alphabetically next to the exempt prefix. The
    unit test above is the real pin; this is the end-to-end confirmation that the middleware
    is the thing consulting it."""
    resp = await client.get("/stats")
    assert resp.status_code == 401


def test_the_exemption_list_is_exactly_what_it_is_documented_to_be():
    """A widening of either set is a security decision. This fails loudly when one happens,
    so it is made deliberately rather than noticed later."""
    assert set(PUBLIC_PATHS) == {"/health", "/login"}
    assert PUBLIC_PREFIXES == ("/static/",)
    assert all(prefix.endswith("/") for prefix in PUBLIC_PREFIXES)


# --- the open-redirect guard ------------------------------------------------------------


@pytest.mark.parametrize(
    "candidate",
    [
        "//evil.example",
        "/\\evil.example",
        "https://evil.example",
        "http://evil.example",
        "evil.example",
        "/triage\\@evil.example",
        "/triage\nLocation: https://evil.example",
        "/triage\r\nSet-Cookie: x=1",
        "",
        None,
    ],
)
def test_safe_next_refuses_anything_that_could_leave_this_origin(candidate):
    assert safe_next(candidate) == "/"


@pytest.mark.parametrize(
    "candidate",
    ["/", "/triage", "/jobs?state=running", "/corpus?package=com.example.app"],
)
def test_safe_next_keeps_a_same_origin_path(candidate):
    assert safe_next(candidate) == candidate


def test_login_url_drops_a_pointless_next():
    assert login_url("/") == "/login"
    assert login_url(None) == "/login"
    assert login_url("//evil.example") == "/login"
    assert login_url("/triage") == "/login?next=/triage"


# --- how a rejection reaches each caller ------------------------------------------------


async def test_a_browser_navigation_is_redirected_to_the_login_form(client):
    resp = await client.get("/triage", headers={"accept": "text/html"})
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login?next=/triage"


async def test_a_browser_navigation_keeps_its_query_string_in_next(client):
    resp = await client.get("/jobs?state=running", headers={"accept": "text/html"})
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login?next=/jobs%3Fstate%3Drunning"


async def test_an_htmx_request_is_told_to_navigate_rather_than_swapping_the_login_page(client):
    """htmx swaps a response into an element. A 303 it followed itself would paste the whole
    login page inside whatever fragment made the request, so the answer is a header."""
    resp = await client.get("/triage", headers={"accept": "text/html", "HX-Request": "true"})
    assert resp.status_code == 401
    assert resp.headers["HX-Redirect"] == "/login?next=/triage"


async def test_a_json_caller_still_gets_the_401_it_always_got(client):
    resp = await client.get("/jobs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 401
    assert resp.json() == {"detail": "not authenticated"}


# --- the login form ---------------------------------------------------------------------


async def test_the_login_page_renders_a_form_without_a_session(client):
    resp = await client.get("/login")
    assert resp.status_code == 200
    assert 'name="password"' in resp.text
    assert "<form" in resp.text


async def test_the_login_page_carries_next_into_its_form(client):
    resp = await client.get("/login?next=/corpus")
    assert resp.status_code == 200
    assert 'value="/corpus"' in resp.text


async def test_the_login_page_refuses_to_carry_an_offsite_next(client):
    resp = await client.get("/login?next=//evil.example")
    assert resp.status_code == 200
    assert "evil.example" not in resp.text


async def test_a_form_login_redirects_to_where_the_browser_was_headed(client):
    resp = await client.post("/login", data={"password": PASSWORD, "next": "/corpus"})
    assert resp.status_code == 303
    assert resp.headers["location"] == "/corpus"


async def test_a_form_login_will_not_redirect_off_this_origin(client):
    resp = await client.post("/login", data={"password": PASSWORD, "next": "//evil.example"})
    assert resp.status_code == 303
    assert resp.headers["location"] == "/"


async def test_a_wrong_form_password_re_renders_the_form_and_grants_nothing(client):
    resp = await client.post("/login", data={"password": "wrong", "next": "/triage"})
    assert resp.status_code == 401
    assert "<form" in resp.text
    # The rejection must not have handed out a session on the way past.
    follow = await client.get("/triage", headers={"accept": "text/html"})
    assert follow.status_code == 303


async def test_a_wrong_json_password_still_raises_the_json_shape(client):
    resp = await client.post("/login", json={"password": "wrong"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "bad credentials"


async def test_the_login_page_sends_an_authenticated_visitor_onward(client):
    await _login(client)
    resp = await client.get("/login?next=/corpus")
    assert resp.status_code == 303
    assert resp.headers["location"] == "/corpus"


# --- the shell itself -------------------------------------------------------------------


async def test_root_sends_a_browser_to_triage_and_a_script_the_identity_payload(client):
    await _login(client)
    html = await client.get("/", headers={"accept": "text/html"})
    assert html.status_code == 303
    assert html.headers["location"] == "/triage"

    payload = await client.get("/")
    assert payload.status_code == 200
    assert payload.json() == {"app": "uadclaw"}


@pytest.mark.parametrize("item", web.NAV_ITEMS)
async def test_every_nav_destination_resolves(db_env, client, item):
    """A nav entry pointing at a route nobody registered is a dead link the moment a screen
    renames its path, and nothing else in the suite would notice.

    Takes `db_env` so it tests what its name says. Without it the database is unreachable and
    this silently became an assertion about how screens behave when Postgres is down — a real
    property, but not this one, and two screen authors independently had to work out why a
    test called "resolves" was failing them. That property now has its own test below.
    """
    await _login(client)
    resp = await client.get(item.href, headers={"accept": "text/html"})
    assert resp.status_code == 200, f"{item.key} -> {item.href}"


@pytest.mark.parametrize("item", web.NAV_ITEMS)
async def test_every_screen_degrades_when_the_database_is_unreachable(client, item):
    """No `db_env`, so `postgres_host` stays at its container-only default and every query
    raises. Each screen must still render, with a visible error.

    Asserting the error marker rather than only the status is the point: a screen that
    swallowed the failure and rendered its EMPTY state would answer 200 and be lying, and
    "no packages yet" is the most expensive lie this dashboard could tell a reviewer.

    Note what this does NOT cover, because it is worth knowing: a database host that is
    routable but dead makes asyncpg hang rather than raise, since nothing sets a connect
    timeout. Measured against a blackhole address, all four screens hang instead of
    degrading. This test reaches the DNS-failure case only.
    """
    await _login(client)
    resp = await client.get(item.href, headers={"accept": "text/html"})
    assert resp.status_code == 200, f"{item.key} 500ed instead of degrading"
    assert "callout-danger" in resp.text, f"{item.key} hid the failure instead of showing it"


async def test_the_shell_marks_the_screen_you_are_on(client):
    await _login(client)
    resp = await client.get("/corpus", headers={"accept": "text/html"})
    assert 'class="navbar-link active"' in resp.text


async def test_the_shell_loads_the_pinned_htmx_before_any_first_party_script(client):
    await _login(client)
    resp = await client.get("/triage", headers={"accept": "text/html"})
    assert f"/static/vendor/{web.HTMX_FILENAME}" in resp.text


def test_the_vendored_htmx_is_the_build_that_was_verified():
    """Pinned by digest, not by filename alone: a rename is not a re-verification, and this
    file was checked against the npm registry's published tarball integrity when it landed."""
    import hashlib

    body = (web.STATIC_DIR / "vendor" / web.HTMX_FILENAME).read_bytes()
    assert (
        hashlib.sha256(body).hexdigest()
        == "71ea67185bfa8c98c39d31717c6fce5d852370fcdfd129db4543774d3145c0de"
    )


def test_every_font_the_stylesheet_asks_for_is_actually_vendored():
    """A missing subset is a 404 per page load and a silently wrong typeface, which no
    functional test would ever catch."""
    import re

    css = (web.STATIC_DIR / "ui.css").read_text()
    declared = set(re.findall(r'url\("fonts/([^"]+)"\)', css))
    assert declared
    present = {path.name for path in (web.STATIC_DIR / "fonts").glob("*.woff2")}
    assert declared <= present, f"declared but missing: {sorted(declared - present)}"
    assert present <= declared, f"vendored but unused: {sorted(present - declared)}"


# --- escaping ---------------------------------------------------------------------------


def test_the_template_environment_escapes_by_default_and_not_only_for_html_files():
    """Package names, labels and fetched page titles all reach a template and none of them
    are this pipeline's bytes. Starlette's own default leaves strings and non-`.html`
    templates unescaped, so this asserts the stricter setting actually took."""
    rendered = web.templates.env.from_string("{{ value }}").render(
        value="<script>alert(1)</script>"
    )
    assert "<script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_an_undefined_template_variable_is_an_error_rather_than_a_blank(client):
    """A screen that silently renders empty because a context key was misspelled is the
    kind of bug a reviewer reads straight past."""
    from jinja2 import UndefinedError

    with pytest.raises(UndefinedError):
        web.templates.env.from_string("{{ nothing_supplied }}").render()
