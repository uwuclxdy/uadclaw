"""The emission screen: the list of branch emissions and one emission's detail, read-only.

Real Postgres throughout, seeded straight into `branch_emission` / `branch_emission_package`
through a real session — the three outcome badges depend on the `commit_oid IS NULL` /
`reconciled` pair, and that tri-state is the whole point of the row, so a mock could only
re-assert what the view already decided. `db_session_factory` truncates every table per test,
so a test's seeded rows are the entire log that test sees.
"""

import re
from datetime import UTC, datetime, timedelta

import pytest

from uadclaw.models import BranchEmission, BranchEmissionPackage

PASSWORD = "test-only-admin-password"


def _utcnow() -> datetime:
    return datetime.now(UTC)


async def _login(client) -> None:
    resp = await client.post("/login", json={"password": PASSWORD})
    assert resp.status_code == 204


def _emission(
    *,
    branch: str,
    package_count: int,
    pr_body: str,
    created_at: datetime,
    commit_oid: str | None,
    reconciled: bool,
) -> BranchEmission:
    return BranchEmission(
        vendor="pixel",
        branch=branch,
        repo_path="/srv/uadclaw/upstream",
        list_path="resources/assets/uad_lists.json",
        base_commit="a" * 40,
        list_sha256="b" * 64,
        pipeline_version="0.1.0",
        pipeline_commit_sha="c" * 40,
        package_count=package_count,
        pr_body=pr_body,
        created_at=created_at,
        commit_oid=commit_oid,
        committed_at=created_at if commit_oid is not None else None,
        reconciled=reconciled,
    )


def _pkg(
    emission_id: int,
    package: str,
    *,
    uad_list: str = "aosp",
    removal: str = "Recommended",
    floor: str = "Recommended",
) -> BranchEmissionPackage:
    return BranchEmissionPackage(
        emission_id=emission_id,
        package=package,
        bundle_sha256="d" * 64,
        uad_list=uad_list,
        removal=removal,
        floor=floor,
    )


async def _seed_emission(
    factory,
    *,
    branch: str = "uadclaw/pixel-000000000001",
    commit_oid: str | None = None,
    reconciled: bool = False,
    pr_body: str = "## pixel: 1 package addition(s)\n",
    created_at: datetime | None = None,
    packages: list[tuple[str, str, str, str]] | None = None,
) -> int:
    packages = packages or [("com.example.one", "aosp", "Recommended", "Recommended")]
    async with factory() as session, session.begin():
        row = _emission(
            branch=branch,
            package_count=len(packages),
            pr_body=pr_body,
            created_at=created_at or _utcnow(),
            commit_oid=commit_oid,
            reconciled=reconciled,
        )
        session.add(row)
        await session.flush()
        emission_id = row.id
        for package, uad_list, removal, floor in packages:
            session.add(_pkg(emission_id, package, uad_list=uad_list, removal=removal, floor=floor))
    return emission_id


def _status_badge(text: str, branch: str) -> tuple[str, str]:
    """The (tag class, label) of the status badge in `branch`'s list row, walked from the
    row's own `<tr>` so the assertion cannot be satisfied by any other tag on the page."""
    idx = text.index(branch)
    row_start = text.rindex("<tr>", 0, idx)
    row_end = text.index("</tr>", idx)
    row = text[row_start:row_end]
    match = re.search(r'class="tag ([^"]+)">([^<]*)</span>', row)
    assert match, f"no status badge in the row for {branch}: {row!r}"
    return match.group(1), match.group(2).strip()


# --- auth ---------------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/emission", "/emission/1"])
async def test_every_screen_route_is_behind_the_default_deny_middleware(client, path):
    resp = await client.get(path)
    assert resp.status_code == 401


# --- empty state ----------------------------------------------------------------------------


async def test_a_fresh_database_reads_as_empty_not_broken(db_env, db_session_factory, client):
    await _login(client)
    resp = await client.get("/emission", headers={"accept": "text/html"})
    assert resp.status_code == 200
    assert "nothing has shipped yet" in resp.text
    assert "could not load" not in resp.text


# --- list -----------------------------------------------------------------------------------


async def test_the_list_renders_vendor_branch_count_and_one_badge_per_outcome(
    db_env, db_session_factory, client
):
    """The three outcomes are the load-bearing part of the row: `commit_oid IS NULL` is
    "outcome unknown", a commit the pipeline watched is "emitted", and one read back after a
    crash is "reconciled". Each renders its own badge and the two "has a commit" states never
    collapse."""
    now = _utcnow()
    await _seed_emission(
        db_session_factory,
        branch="uadclaw/pixel-000000000001",
        commit_oid=None,
        created_at=now,
    )
    await _seed_emission(
        db_session_factory,
        branch="uadclaw/pixel-000000000002",
        commit_oid="e" * 40,
        reconciled=False,
        created_at=now,
        packages=[
            ("com.example.a", "aosp", "Recommended", "Recommended"),
            ("com.example.b", "aosp", "Recommended", "Recommended"),
        ],
    )
    await _seed_emission(
        db_session_factory,
        branch="uadclaw/pixel-000000000003",
        commit_oid="f" * 40,
        reconciled=True,
        created_at=now,
    )
    await _login(client)

    resp = await client.get("/emission", headers={"accept": "text/html"})

    assert resp.status_code == 200
    text = resp.text
    assert _status_badge(text, "uadclaw/pixel-000000000001") == ("tag-default", "outcome unknown")
    assert _status_badge(text, "uadclaw/pixel-000000000002") == ("tag-success", "emitted")
    assert _status_badge(text, "uadclaw/pixel-000000000003") == ("tag-warning", "reconciled")
    # vendor and package count render, and the two-package row is the one that says so.
    assert ">pixel<" in text
    assert ">2<" in text
    assert ">1<" in text


async def test_the_list_is_newest_first(db_env, db_session_factory, client):
    older = _utcnow() - timedelta(hours=1)
    newer = _utcnow()
    await _seed_emission(
        db_session_factory, branch="uadclaw/pixel-older", created_at=older, commit_oid="e" * 40
    )
    await _seed_emission(
        db_session_factory, branch="uadclaw/pixel-newer", created_at=newer, commit_oid="f" * 40
    )
    await _login(client)

    resp = await client.get("/emission", headers={"accept": "text/html"})

    assert resp.text.index("uadclaw/pixel-newer") < resp.text.index("uadclaw/pixel-older")


# --- detail ---------------------------------------------------------------------------------


async def test_the_detail_renders_batch_branch_commits_and_the_reconciled_flag(
    db_env, db_session_factory, client
):
    emission_id = await _seed_emission(
        db_session_factory,
        branch="uadclaw/pixel-000000000009",
        commit_oid="e" * 40,
        reconciled=True,
        pr_body="## pixel: 2 package addition(s)\n",
        packages=[
            ("com.example.one", "aosp", "Expert", "Recommended"),
            ("com.example.two", "carrier", "Recommended", "Recommended"),
        ],
    )
    await _login(client)

    resp = await client.get(f"/emission/{emission_id}", headers={"accept": "text/html"})

    assert resp.status_code == 200
    text = resp.text
    assert "com.example.one" in text
    assert "com.example.two" in text
    assert "uadclaw/pixel-000000000009" in text
    assert "a" * 40 in text  # base commit
    assert "e" * 40 in text  # emitted commit
    assert '<span class="tag tag-warning">reconciled</span>' in text
    assert ">yes<" in text  # the reconciled flag, spelled yes rather than a bare true
    assert ">Expert<" in text  # removal column
    assert ">carrier<" in text  # uad_list column
    assert ">Recommended<" in text  # floor column


# --- untrusted bytes: pr body and package names ---------------------------------------------


async def test_the_pr_body_is_escaped_inside_the_copyable_element(
    db_env, db_session_factory, client
):
    """The PR body is the one artifact a human copies into GitHub, and it was built from
    manifest strings. Inside a `<textarea readonly>` the autoescaped entities decode to the raw
    body in the box but are never markup in the response."""
    body = "<script>alert(1)</script> Tom & Jerry"
    emission_id = await _seed_emission(db_session_factory, commit_oid="e" * 40, pr_body=body)
    await _login(client)

    resp = await client.get(f"/emission/{emission_id}", headers={"accept": "text/html"})

    assert resp.status_code == 200
    assert "<textarea" in resp.text
    assert "readonly" in resp.text
    assert "&lt;script&gt;" in resp.text
    assert "<script>alert(1)</script>" not in resp.text
    assert "Tom &amp; Jerry" in resp.text


async def test_a_slash_in_a_package_name_is_encoded_in_the_detail_link(
    db_env, db_session_factory, client
):
    """`_package_href` goes through `web.url_segment`, never the `|urlencode` filter: jinja's
    `do_urlencode` calls `url_quote` with `safe=b"/"`, so a package name carrying a `/` comes
    back unescaped and the link points at a different path than the record."""
    emission_id = await _seed_emission(
        db_session_factory,
        commit_oid="e" * 40,
        packages=[("com.example/evil", "aosp", "Recommended", "Recommended")],
    )
    await _login(client)

    resp = await client.get(f"/emission/{emission_id}", headers={"accept": "text/html"})

    assert 'href="/corpus/com.example%2Fevil"' in resp.text
    assert 'href="/corpus/com.example/evil"' not in resp.text


# --- not found and error, distinct from each other and from empty ---------------------------


async def test_a_missing_emission_reads_as_not_found_not_as_an_error(
    db_env, db_session_factory, client
):
    await _login(client)
    resp = await client.get("/emission/999999", headers={"accept": "text/html"})
    # 200, not 404: htmx 2.0.10 does not swap a non-2xx response by default, so a boosted link
    # into a missing emission must still render visible content.
    assert resp.status_code == 200
    assert "no emission with id" in resp.text
    assert "could not load" not in resp.text


async def test_a_db_failure_reads_as_an_error_never_as_an_empty_log(client):
    """No `db_env`: `POSTGRES_HOST` keeps its unreachable default, so the query fails for real
    rather than being mocked."""
    await _login(client)
    resp = await client.get("/emission", headers={"accept": "text/html"})
    assert resp.status_code == 200
    assert "could not load" in resp.text
    assert "nothing has shipped yet" not in resp.text


async def test_a_db_failure_on_the_detail_page_also_reads_as_an_error(client):
    await _login(client)
    resp = await client.get("/emission/1", headers={"accept": "text/html"})
    assert resp.status_code == 200
    assert "could not load" in resp.text
    assert "no emission with id" not in resp.text
