"""The jobs screen: the list, one run's detail, the two launch controls, and the scratch
lease.

**No test here touches the network, and that is a property of the seam rather than of care.**
Every driver is installed through `firmware._driver_factories`, the same registry
`firmware.get_driver` resolves, so the view's real code path runs against a driver whose
`list_available` is a local list. `FakeDriver.fetch` raises outright, because nothing on this
screen may download firmware. Two tests deliberately use REAL drivers, and both take a path
that raises before a client is opened: `MotorolaDriver.list_available` refuses an empty
`MOTOROLA_DEVICES` first thing, and `PixelDriver.list_available` calls `_ack_headers()` before
`_open_client()`.
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from uadclaw import firmware as firmware_module
from uadclaw.firmware import FirmwareDriver, FirmwareRef, TermsPosture, TermsRisk
from uadclaw.models import Job, JobKind, JobState, ScratchLease
from uadclaw.settings import get_settings
from uadclaw.views import jobs as jobs_view

PASSWORD = "test-only-admin-password"


async def _login(client) -> None:
    resp = await client.post("/login", json={"password": PASSWORD})
    assert resp.status_code == 204


def _utcnow() -> datetime:
    return datetime.now(UTC)


# --- the driver seam ----------------------------------------------------------------------


class FakeDriver(FirmwareDriver):
    """A driver with no I/O in it at all. `calls` is what proves a page render never listed."""

    def __init__(
        self,
        name: str,
        *,
        refs: list[FirmwareRef] | None = None,
        error: Exception | None = None,
        risk: TermsRisk = TermsRisk.PUBLIC,
        acknowledged: bool = True,
        summary: str = "a source with nothing to accept.",
    ) -> None:
        self.name = name
        self._refs = refs if refs is not None else []
        self._error = error
        self._risk = risk
        self._acknowledged = acknowledged
        self._summary = summary
        self.calls = 0

    def terms(self) -> TermsPosture:
        return TermsPosture(
            risk=self._risk,
            source_url=f"https://example.invalid/{self.name}",
            summary=self._summary,
            acknowledged=self._acknowledged,
        )

    async def list_available(self) -> list[FirmwareRef]:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return list(self._refs)

    async def fetch(self, ref, dest_dir):  # pragma: no cover - never reached by this screen
        raise AssertionError("the jobs screen must never download firmware")


def _ref(device: str, build: str, **kwargs) -> FirmwareRef:
    return FirmwareRef(
        driver="pixel",
        device=device,
        build=build,
        url=f"https://example.invalid/{device}-{build}.zip",
        **kwargs,
    )


def set_config(monkeypatch, **values: str) -> None:
    """Set configuration and drop the settings cache.

    The `client` fixture builds the app during setup, which resolves the `lru_cache`d
    `get_settings()` before the test body runs, so a bare `setenv` here would be read by
    nothing. Clearing the cache is what makes the next `get_settings()` see it.
    """
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()


def install_drivers(monkeypatch, *drivers: FakeDriver) -> None:
    """Register these through the real registry, so `get_driver`'s enable/disable check and
    `enabled_driver_names` are both on the path under test rather than bypassed."""
    factories = {driver.name: (lambda settings, d=driver: d) for driver in drivers}
    monkeypatch.setattr(firmware_module, "_driver_factories", lambda: factories)


@pytest.fixture(autouse=True)
def _cold_device_cache():
    """The index cache is process-local, so one test's listing would otherwise answer the
    next test's fetch and hide a missing call."""
    jobs_view._device_index_cache.clear()
    yield
    jobs_view._device_index_cache.clear()


# --- seeding ------------------------------------------------------------------------------


async def _seed_job(
    factory,
    *,
    kind: JobKind = JobKind.FIRMWARE_ANALYSIS,
    state: JobState = JobState.QUEUED,
    stage: str | None = None,
    params: dict | None = None,
    **columns,
) -> uuid.UUID:
    job_id = uuid.uuid4()
    async with factory() as session, session.begin():
        session.add(
            Job(
                id=job_id,
                kind=kind.value,
                params=params if params is not None else {"driver": "pixel", "device": "comet"},
                state=state,
                stage=stage,
                attempt=columns.pop("attempt", 1),
                log_tail=columns.pop("log_tail", ""),
                created_at=columns.pop("created_at", _utcnow()),
                **columns,
            )
        )
    return job_id


# --- the page renders without reaching a vendor -------------------------------------------


async def test_the_jobs_page_never_asks_a_vendor_for_its_index(
    db_env, db_session_factory, client, monkeypatch
):
    """`list_available` is a 1.2 MB request against a vendor, so a page load must not make
    one. The driver raises if anything calls it."""
    driver = FakeDriver("pixel", error=AssertionError("a page render listed the index"))
    install_drivers(monkeypatch, driver)
    await _login(client)

    resp = await client.get("/jobs", headers={"accept": "text/html"})

    assert resp.status_code == 200
    assert driver.calls == 0
    assert "no device list yet" in resp.text


async def test_the_page_offers_only_the_drivers_configuration_leaves_enabled(
    db_env, db_session_factory, client, monkeypatch
):
    set_config(monkeypatch, DISABLED_FIRMWARE_DRIVERS="samsung")
    install_drivers(monkeypatch, FakeDriver("pixel"), FakeDriver("samsung"))
    await _login(client)

    resp = await client.get("/jobs", headers={"accept": "text/html"})

    assert '<option value="pixel"' in resp.text
    assert '<option value="samsung"' not in resp.text


async def test_each_driver_shows_the_terms_posture_it_declares(
    db_env, db_session_factory, client, monkeypatch
):
    """`terms()` exists in code specifically so this screen can show it."""
    install_drivers(
        monkeypatch,
        FakeDriver(
            "samsung",
            risk=TermsRisk.REVERSE_ENGINEERED,
            acknowledged=False,
            summary="a private endpoint with no published terms at all.",
        ),
    )
    await _login(client)

    resp = await client.get("/jobs", headers={"accept": "text/html"})

    assert "reverse engineered" in resp.text
    assert "a private endpoint with no published terms at all." in resp.text
    assert "not acknowledged" in resp.text


async def test_the_firmware_sources_table_collapses_when_nothing_needs_attention(
    db_env, db_session_factory, client, monkeypatch
):
    """The terms table is reference material collapsed behind `<details>` so a returning
    operator does not scroll past it every visit. A request/response test cannot observe an
    interactive open/close, but the rendered markup is production-visible and pinnable:
    `<details>` with no `open` attribute, and a trigger naming the driver count with no
    attention flag, when every driver is public and acknowledged."""
    install_drivers(monkeypatch, FakeDriver("pixel"))
    await _login(client)

    resp = await client.get("/jobs", headers={"accept": "text/html"})

    assert "<details>" in resp.text
    assert "<details open>" not in resp.text
    assert "show terms for 1 driver" in resp.text
    assert "not acknowledged" not in resp.text


async def test_the_firmware_sources_table_opens_and_flags_unacknowledged_drivers(
    db_env, db_session_factory, client, monkeypatch
):
    """An unacknowledged or reverse-engineered driver is risk state, not reference detail —
    progressive disclosure is for reference detail only, so this has to be visible without a
    click. The panel opens itself and the trigger names how many need a decision."""
    install_drivers(
        monkeypatch,
        FakeDriver("pixel", acknowledged=True, risk=TermsRisk.PUBLIC),
        FakeDriver("samsung", acknowledged=False, risk=TermsRisk.REVERSE_ENGINEERED),
    )
    await _login(client)

    resp = await client.get("/jobs", headers={"accept": "text/html"})

    assert "<details open>" in resp.text
    assert "show terms for 2 drivers · 1 not acknowledged" in resp.text


# --- loading a device list ----------------------------------------------------------------


async def test_the_load_devices_button_carries_hx_disabled_elt(
    db_env, db_session_factory, client, monkeypatch
):
    """docs/todo.md §16: `hx-disabled-elt="this"` disables the button while
    `POST /jobs/launch/devices` is in flight — a multi-second vendor request per the module
    docstring — and had never been observed to fire, because the only browser that could drive
    it failed instantly. Observing it actually fire still needs a browser running htmx's JS,
    which nothing in this suite does; what a request/response test can pin is that the wiring
    survives, so a regression dropping the attribute is caught rather than silently shipped."""
    install_drivers(monkeypatch, FakeDriver("pixel"))
    await _login(client)

    resp = await client.get("/jobs", headers={"accept": "text/html"})

    assert 'hx-disabled-elt="this"' in resp.text
    assert 'hx-post="/jobs/launch/devices"' in resp.text


async def test_a_realistic_sized_index_renders_grouped_by_device_not_by_ref(
    db_env, db_session_factory, client, monkeypatch
):
    """docs/todo.md §16: no screen had ever rendered a real vendor index. The Pixel index is
    2293 refs across 58 devices, and `_device_options` groups by device, so the rendered
    `<select>` should be bounded by device count rather than ref count — reasoning the todo
    item names as unmeasured. This renders a stubbed index of that exact shape through the
    real view and template and measures the result instead of reasoning about it."""
    device_count = 58
    total_refs = 2293
    base, extra = divmod(total_refs, device_count)
    refs = [
        _ref(f"device{i:02d}", f"BUILD.{i:02d}.{b:03d}")
        for i in range(device_count)
        for b in range(base + (1 if i < extra else 0))
    ]
    assert len(refs) == total_refs
    install_drivers(monkeypatch, FakeDriver("pixel", refs=refs))
    await _login(client)

    resp = await client.post("/jobs/launch/devices", data={"driver": "pixel"})

    assert resp.status_code == 200
    option_count = resp.text.count('<option value="device')
    assert option_count == device_count, (
        f"the device picker renders one <option> per device, not per ref: got {option_count}"
    )
    assert f"{device_count} devices, {total_refs} builds" in resp.text


async def test_loading_devices_is_an_explicit_action_that_lists_them(
    db_env, db_session_factory, client, monkeypatch
):
    driver = FakeDriver(
        "pixel",
        refs=[
            _ref("comet", "AP4A.250105.002", marketing_name="Pixel 9 Pro Fold"),
            _ref("comet", "CP2A.260705.006"),
            _ref("oriole", "AP2A.240905.003"),
        ],
    )
    install_drivers(monkeypatch, driver)
    await _login(client)

    resp = await client.post("/jobs/launch/devices", data={"driver": "pixel"})

    assert resp.status_code == 200
    assert driver.calls == 1
    assert "comet" in resp.text
    assert "Pixel 9 Pro Fold" in resp.text
    # The build the acquire stage would resolve, taken from `select_ref` rather than row order.
    assert "newest CP2A.260705.006" in resp.text
    assert "2 devices, 3 builds" in resp.text


async def test_an_empty_index_is_an_error_and_never_an_empty_device_list(
    db_env, db_session_factory, client, monkeypatch
):
    """A terms-walled index answers 200 with zero links. Rendering that as "no devices" sends
    the operator looking at the vendor instead of at their own configuration."""
    install_drivers(
        monkeypatch,
        FakeDriver("pixel", error=firmware_module.EmptyFirmwareIndexError("the wall answered")),
    )
    await _login(client)

    resp = await client.post("/jobs/launch/devices", data={"driver": "pixel"})

    assert "callout-danger" in resp.text
    assert "the index answered with no builds" in resp.text
    assert "EmptyFirmwareIndexError" in resp.text
    assert "no device list yet" not in resp.text
    assert "0 devices" not in resp.text


async def test_a_driver_answering_with_no_refs_at_all_is_still_an_error(
    db_env, db_session_factory, client, monkeypatch
):
    """The drivers raise for this themselves; the view refuses to trust that they always will,
    because the failure mode is a rendered fact about the vendor that is not true."""
    install_drivers(monkeypatch, FakeDriver("pixel", refs=[]))
    await _login(client)

    resp = await client.post("/jobs/launch/devices", data={"driver": "pixel"})

    assert "callout-danger" in resp.text
    assert "EmptyFirmwareIndexError" in resp.text
    assert "no device list yet" not in resp.text


async def test_motorola_with_no_configured_devices_names_that_rather_than_showing_none(
    db_env, db_session_factory, client, monkeypatch
):
    """The real driver, on the path that raises before it opens a client. lolinet publishes no
    index document, so an unconfigured Motorola has nothing to enumerate and that is a
    configuration fact, not "this vendor ships no phones"."""
    set_config(monkeypatch, MOTOROLA_DEVICES="")
    await _login(client)

    resp = await client.post("/jobs/launch/devices", data={"driver": "motorola"})

    assert resp.status_code == 200
    assert "MOTOROLA_DEVICES" in resp.text
    assert "no device list yet" not in resp.text


async def test_an_unacknowledged_terms_wall_is_named_rather_than_read_as_an_empty_source(
    db_env, db_session_factory, client, monkeypatch
):
    """The real Pixel driver, which checks the acknowledgement before it opens a client."""
    set_config(monkeypatch, PIXEL_TERMS_ACK_COOKIE_VALUE="")
    await _login(client)

    resp = await client.post("/jobs/launch/devices", data={"driver": "pixel"})

    assert "the terms for this source are not acknowledged" in resp.text
    assert "PIXEL_TERMS_ACK_COOKIE_VALUE" in resp.text


async def test_a_disabled_driver_is_refused_by_name_and_not_as_a_typo(
    db_env, db_session_factory, client, monkeypatch
):
    set_config(monkeypatch, DISABLED_FIRMWARE_DRIVERS="samsung")
    install_drivers(monkeypatch, FakeDriver("pixel"), FakeDriver("samsung"))
    await _login(client)

    resp = await client.post("/jobs/launch/devices", data={"driver": "samsung"})

    assert "that driver is switched off" in resp.text
    assert "no driver by that name" not in resp.text


async def test_the_index_is_fetched_once_and_re_used_for_the_next_render(
    db_env, db_session_factory, client, monkeypatch
):
    driver = FakeDriver("pixel", refs=[_ref("comet", "CP2A.260705.006")])
    install_drivers(monkeypatch, driver)
    await _login(client)

    first = await client.post("/jobs/launch/devices", data={"driver": "pixel"})
    second = await client.post("/jobs/launch/devices", data={"driver": "pixel"})

    assert driver.calls == 1
    assert "comet" in first.text
    assert "comet" in second.text


async def test_a_stale_index_is_listed_again_rather_than_served_forever(
    db_env, db_session_factory, client, monkeypatch
):
    """The cache exists to spare a vendor three requests for one form submission, not to pin
    a catalogue for the life of the process."""
    driver = FakeDriver("pixel", refs=[_ref("comet", "CP2A.260705.006")])
    install_drivers(monkeypatch, driver)
    monkeypatch.setattr(jobs_view, "DEVICE_INDEX_TTL_SECONDS", -1.0)
    await _login(client)

    await client.post("/jobs/launch/devices", data={"driver": "pixel"})
    await client.post("/jobs/launch/devices", data={"driver": "pixel"})

    assert driver.calls == 2


async def test_a_driver_name_nothing_registers_reads_as_a_typo_not_as_a_shutoff(
    db_env, db_session_factory, client, monkeypatch
):
    install_drivers(monkeypatch, FakeDriver("pixel"))
    await _login(client)

    resp = await client.post("/jobs/launch/devices", data={"driver": "pixle"})

    assert "no driver by that name" in resp.text
    assert "that driver is switched off" not in resp.text


async def test_an_htmx_request_gets_the_fragment_and_a_browser_post_gets_the_page(
    db_env, db_session_factory, client, monkeypatch
):
    """Without javascript the same form posts natively, so the route has to answer both."""
    install_drivers(monkeypatch, FakeDriver("pixel", refs=[_ref("comet", "CP2A.260705.006")]))
    await _login(client)

    fragment = await client.post(
        "/jobs/launch/devices", data={"driver": "pixel"}, headers={"HX-Request": "true"}
    )
    page = await client.post("/jobs/launch/devices", data={"driver": "pixel"})

    assert "<html" not in fragment.text
    assert "<html" in page.text
    assert "comet" in fragment.text
    assert "comet" in page.text


# --- launching a firmware job -------------------------------------------------------------


async def test_queueing_a_firmware_job_creates_it_and_lands_on_its_detail(
    db_env, db_session_factory, client, monkeypatch
):
    install_drivers(monkeypatch, FakeDriver("pixel", refs=[_ref("comet", "CP2A.260705.006")]))
    await _login(client)

    resp = await client.post(
        "/jobs/launch/firmware", data={"driver": "pixel", "device": "comet", "build": ""}
    )

    assert resp.status_code == 303
    async with db_session_factory() as session:
        from sqlalchemy import select

        job = (await session.execute(select(Job))).scalar_one()
    assert resp.headers["location"] == f"/jobs/{job.id}/detail"
    assert job.kind == "firmware_analysis"
    assert job.state == JobState.QUEUED
    # No url and no digest: `acquire` re-resolves the build when the job actually runs, so a
    # link that expires in the queue costs nothing.
    assert job.params == {"driver": "pixel", "device": "comet"}


async def test_a_named_build_reaches_the_job_params(
    db_env, db_session_factory, client, monkeypatch
):
    install_drivers(monkeypatch, FakeDriver("pixel", refs=[_ref("comet", "CP2A.260705.006")]))
    await _login(client)

    await client.post(
        "/jobs/launch/firmware",
        data={"driver": "pixel", "device": "comet", "build": "CP2A.260705.006"},
    )

    from sqlalchemy import select

    async with db_session_factory() as session:
        job = (await session.execute(select(Job))).scalar_one()
    assert job.params["build"] == "CP2A.260705.006"


async def test_a_firmware_target_the_job_validator_refuses_comes_back_on_the_form(
    db_env, db_session_factory, client, monkeypatch
):
    """`jobs.create_job` is the only validator. A blank device is a 422 there, and the screen
    re-renders with what the operator typed rather than clearing it."""
    install_drivers(monkeypatch, FakeDriver("pixel"))
    await _login(client)

    resp = await client.post("/jobs/launch/firmware", data={"driver": "pixel", "device": ""})

    assert resp.status_code == 422
    assert "that firmware target was refused" in resp.text
    assert '<option value="pixel" selected' in resp.text
    from sqlalchemy import select

    async with db_session_factory() as session:
        assert (await session.execute(select(Job))).scalars().all() == []


# --- launching a classification job -------------------------------------------------------


async def test_the_classification_control_says_it_spends_money_before_it_is_pressed(
    db_env, db_session_factory, client, monkeypatch
):
    install_drivers(monkeypatch, FakeDriver("pixel"))
    await _login(client)

    resp = await client.get("/jobs", headers={"accept": "text/html"})

    assert "this one spends money" in resp.text
    assert "$0.055" in resp.text
    assert "spending api credit" in resp.text


async def test_a_classification_run_carries_the_params_it_was_given(
    db_env, db_session_factory, client, monkeypatch
):
    install_drivers(monkeypatch, FakeDriver("pixel"))
    await _login(client)

    resp = await client.post(
        "/jobs/launch/classification",
        data={"packages": "com.example.one, com.example.two", "limit": "5", "reclassify": "on"},
    )

    assert resp.status_code == 303
    from sqlalchemy import select

    async with db_session_factory() as session:
        job = (await session.execute(select(Job))).scalar_one()
    assert job.kind == "classification"
    assert job.params["packages"] == ["com.example.one", "com.example.two"]
    assert job.params["limit"] == 5
    assert job.params["reclassify"] is True


async def test_an_omitted_package_list_and_limit_leave_the_defaults_alone(
    db_env, db_session_factory, client, monkeypatch
):
    """A blank field is dropped rather than handed over as `""`, so the params model applies
    its own default: no named packages, and the settings ceiling as the limit."""
    install_drivers(monkeypatch, FakeDriver("pixel"))
    await _login(client)

    await client.post("/jobs/launch/classification", data={"packages": "", "limit": ""})

    from sqlalchemy import select

    async with db_session_factory() as session:
        job = (await session.execute(select(Job))).scalar_one()
    assert job.params["packages"] == []
    # `validate_job_params` drops a None, so an absent limit means the settings ceiling.
    assert "limit" not in job.params
    assert job.params["reclassify"] is False


async def test_a_limit_that_is_not_a_number_is_refused_by_the_params_model(
    db_env, db_session_factory, client, monkeypatch
):
    install_drivers(monkeypatch, FakeDriver("pixel"))
    await _login(client)

    resp = await client.post("/jobs/launch/classification", data={"limit": "soon"})

    assert resp.status_code == 422
    assert "that classification run was refused" in resp.text


# --- polling ------------------------------------------------------------------------------


def test_poll_trigger_is_dropped_only_when_every_state_is_terminal():
    assert jobs_view.poll_trigger([JobState.SUCCEEDED, JobState.FAILED]) is None
    # An empty list has no terminal state, so it cannot satisfy "every state is terminal".
    # This line asserted the opposite of the name above it, and the behaviour matched: the
    # loop body never ran, so the fragment came back with no trigger at all.
    assert jobs_view.poll_trigger([]) == "every 3s"
    assert jobs_view.poll_trigger([JobState.SUCCEEDED, JobState.RUNNING]) == "every 3s"
    assert jobs_view.poll_trigger([JobState.QUEUED]) == "every 3s"
    assert jobs_view.poll_trigger([JobState.CLAIMED]) == "every 3s"


async def test_a_filtered_list_with_nothing_in_it_keeps_asking_for_itself(
    db_env, db_session_factory, client, monkeypatch
):
    """`_list_context` feeds `poll_trigger` the FILTERED rows, so an operator sitting on
    `?state=running` with nothing running got a fragment carrying no trigger and never saw a
    job queued afterwards until they reloaded by hand. An empty list is not a finished one."""
    install_drivers(monkeypatch, FakeDriver("pixel"))
    await _seed_job(db_session_factory, state=JobState.SUCCEEDED)
    await _login(client)

    resp = await client.get("/jobs/list/rows?state=running&kind=")

    assert resp.status_code == 200
    assert 'hx-trigger="every 3s"' in resp.text


async def test_a_list_whose_every_job_is_finished_stops_asking(
    db_env, db_session_factory, client, monkeypatch
):
    """The other leg, and the one the trigger exists for: a fragment that keeps polling after
    the last job finished re-queries Postgres for the rest of the tab's life over rows that
    cannot change again."""
    install_drivers(monkeypatch, FakeDriver("pixel"))
    await _seed_job(db_session_factory, state=JobState.SUCCEEDED)
    await _login(client)

    resp = await client.get("/jobs/list/rows?state=succeeded&kind=")

    assert resp.status_code == 200
    assert "hx-trigger" not in resp.text


async def test_a_live_run_panel_asks_for_itself_again(db_env, db_session_factory, client):
    job_id = await _seed_job(db_session_factory, state=JobState.RUNNING, stage="unpack")
    await _login(client)

    resp = await client.get(f"/jobs/{job_id}/detail/panel")

    assert resp.status_code == 200
    assert 'hx-trigger="every 3s"' in resp.text


async def test_a_finished_run_panel_stops_polling(db_env, db_session_factory, client):
    """A trigger that outlives the job re-queries Postgres for the life of the tab over a row
    that cannot change again."""
    job_id = await _seed_job(
        db_session_factory,
        state=JobState.SUCCEEDED,
        stage="rule_ladder",
        started_at=_utcnow() - timedelta(seconds=30),
        finished_at=_utcnow(),
    )
    await _login(client)

    resp = await client.get(f"/jobs/{job_id}/detail/panel")

    assert resp.status_code == 200
    assert "hx-trigger" not in resp.text
    assert f"/jobs/{job_id}/detail/panel" in resp.text


async def test_the_run_list_stops_polling_once_every_run_is_terminal(
    db_env, db_session_factory, client
):
    await _seed_job(db_session_factory, state=JobState.SUCCEEDED, stage="rule_ladder")
    await _seed_job(db_session_factory, state=JobState.FAILED, stage="unpack")
    await _login(client)

    resp = await client.get("/jobs/list/rows")

    assert resp.status_code == 200
    assert "hx-trigger" not in resp.text


async def test_the_run_list_keeps_polling_while_one_run_is_live(db_env, db_session_factory, client):
    await _seed_job(db_session_factory, state=JobState.SUCCEEDED, stage="rule_ladder")
    await _seed_job(db_session_factory, state=JobState.QUEUED)
    await _login(client)

    resp = await client.get("/jobs/list/rows")

    assert 'hx-trigger="every 3s"' in resp.text


async def test_the_panel_for_a_run_that_is_gone_stops_polling_instead_of_chasing_it(
    db_env, db_session_factory, client
):
    await _login(client)

    resp = await client.get(f"/jobs/{uuid.uuid4()}/detail/panel")

    # 200 and not 404: htmx swaps neither a 404 nor a 5xx, so a 404 here would leave the old
    # panel on screen with its trigger intact, polling a row that no longer exists.
    assert resp.status_code == 200
    assert "hx-trigger" not in resp.text
    assert "this run is gone" in resp.text


# --- the stage walk is the KIND's walk ----------------------------------------------------


async def test_a_firmware_run_walks_its_own_kinds_stages_and_not_the_whole_vocabulary(
    db_env, db_session_factory, client
):
    """`JOB_KIND_STAGES` decides, never `PIPELINE_STAGES`: a firmware job ends at
    `rule_ladder`, and showing the four stages it deliberately never runs would report every
    finished one as 6 of 10 forever."""
    job_id = await _seed_job(
        db_session_factory, state=JobState.SUCCEEDED, stage="rule_ladder", attempt=1
    )
    await _login(client)

    resp = await client.get(f"/jobs/{job_id}/detail", headers={"accept": "text/html"})

    assert '<span class="table-name">rule_ladder</span>' in resp.text
    for absent in ("llm", "corroborate", "triage", "branch"):
        assert f'<span class="table-name">{absent}</span>' not in resp.text
    assert "6 of 6 done" in resp.text


async def test_a_classification_run_walks_llm_then_corroborate(db_env, db_session_factory, client):
    job_id = await _seed_job(
        db_session_factory,
        kind=JobKind.CLASSIFICATION,
        state=JobState.RUNNING,
        stage="llm",
        params={"reclassify": False},
    )
    await _login(client)

    resp = await client.get(f"/jobs/{job_id}/detail", headers={"accept": "text/html"})

    assert '<span class="table-name">llm</span>' in resp.text
    assert '<span class="table-name">corroborate</span>' in resp.text
    assert '<span class="table-name">acquire</span>' not in resp.text
    assert "1 of 2 done" in resp.text


async def test_the_list_counts_completed_stages_against_this_kinds_walk(
    db_env, db_session_factory, client
):
    await _seed_job(db_session_factory, state=JobState.RUNNING, stage="unpack")
    await _login(client)

    resp = await client.get("/jobs/list/rows")

    assert "2 of 6" in resp.text
    assert "extract_facts" in resp.text
    # The bar agrees with the count beside it rather than sitting at a constant.
    assert 'class="progress-bar" style="width:33%"' in resp.text


async def test_a_capped_list_says_how_much_of_the_queue_it_is_showing(
    db_env, db_session_factory, client, monkeypatch
):
    """The partial state: the table settled, and it holds less than the queue does."""
    monkeypatch.setattr(jobs_view, "LIST_LIMIT", 2)
    for _ in range(3):
        await _seed_job(db_session_factory, state=JobState.SUCCEEDED, stage="rule_ladder")
    await _login(client)

    resp = await client.get("/jobs/list/rows")

    assert "showing 2 of 3 runs, newest 2 first" in resp.text


async def test_a_failed_run_marks_the_stage_it_stopped_on(db_env, db_session_factory, client):
    job_id = await _seed_job(
        db_session_factory,
        state=JobState.FAILED,
        stage="unpack",
        failure_reason="7z refused the image",
    )
    await _login(client)

    resp = await client.get(f"/jobs/{job_id}/detail", headers={"accept": "text/html"})

    assert "this run failed" in resp.text
    assert "7z refused the image" in resp.text
    assert '<span class="tag tag-danger">failed</span>' in resp.text


async def test_a_live_run_says_how_many_stages_have_a_recorded_run(
    db_env, db_session_factory, client
):
    """The partial state: the walk is known, most of it has not happened yet, and the screen
    says so rather than reading as finished."""
    job_id = await _seed_job(db_session_factory, state=JobState.RUNNING, stage=None)
    await _login(client)

    resp = await client.get(f"/jobs/{job_id}/detail", headers={"accept": "text/html"})

    assert "0 of 6 stages have a recorded run" in resp.text
    assert "the rest have not started." in resp.text


# --- the scratch lease --------------------------------------------------------------------


async def test_the_lease_card_names_the_job_holding_scratch(
    db_env, db_session_factory, client, monkeypatch
):
    install_drivers(monkeypatch, FakeDriver("pixel"))
    job_id = await _seed_job(db_session_factory, state=JobState.RUNNING, stage="acquire")
    async with db_session_factory() as session, session.begin():
        session.add(
            ScratchLease(
                id=1,
                holder_job_id=job_id,
                holder_worker_id="worker-7",
                acquired_at=_utcnow(),
                heartbeat_at=_utcnow(),
            )
        )
    await _login(client)

    resp = await client.get("/jobs", headers={"accept": "text/html"})

    assert "worker-7" in resp.text
    assert str(job_id)[:8] in resp.text
    assert ">held<" in resp.text


async def test_the_lease_card_reads_free_when_nobody_holds_it(
    db_env, db_session_factory, client, monkeypatch
):
    install_drivers(monkeypatch, FakeDriver("pixel"))
    async with db_session_factory() as session, session.begin():
        session.add(ScratchLease(id=1, holder_job_id=None, holder_worker_id=None))
    await _login(client)

    resp = await client.get("/jobs", headers={"accept": "text/html"})

    assert ">free<" in resp.text
    assert "nobody holds it" in resp.text


# --- the five screen states, and escaping -------------------------------------------------


async def test_an_empty_queue_offers_the_action_that_fills_it(
    db_env, db_session_factory, client, monkeypatch
):
    install_drivers(monkeypatch, FakeDriver("pixel"))
    await _login(client)

    resp = await client.get("/jobs", headers={"accept": "text/html"})

    assert "nothing has run yet" in resp.text
    assert "queue a firmware analysis above" in resp.text


async def test_a_filter_matching_nothing_says_runs_exist_and_offers_a_way_back(
    db_env, db_session_factory, client, monkeypatch
):
    install_drivers(monkeypatch, FakeDriver("pixel"))
    await _seed_job(db_session_factory, state=JobState.QUEUED)
    await _login(client)

    resp = await client.get("/jobs?state=failed", headers={"accept": "text/html"})

    assert "no run matches this filter" in resp.text
    assert "drop the filter" in resp.text
    assert "nothing has run yet" not in resp.text


async def test_a_state_filter_lists_only_that_state(db_env, db_session_factory, client):
    queued = await _seed_job(db_session_factory, state=JobState.QUEUED)
    failed = await _seed_job(db_session_factory, state=JobState.FAILED, stage="acquire")
    await _login(client)

    resp = await client.get("/jobs/list/rows?state=failed")

    assert str(failed)[:8] in resp.text
    assert str(queued)[:8] not in resp.text


async def test_the_poll_asks_for_the_same_filter_the_operator_chose(
    db_env, db_session_factory, client, monkeypatch
):
    """A poll url that drops the filter silently resets the table to everything three seconds
    after it was narrowed."""
    install_drivers(monkeypatch, FakeDriver("pixel"))
    await _seed_job(db_session_factory, state=JobState.QUEUED)
    await _login(client)

    resp = await client.get("/jobs?state=queued", headers={"accept": "text/html"})

    assert "/jobs/list/rows?state=queued&amp;kind=" in resp.text


async def test_a_filter_value_the_url_made_up_lists_everything_and_says_so(
    db_env, db_session_factory, client, monkeypatch
):
    install_drivers(monkeypatch, FakeDriver("pixel"))
    job_id = await _seed_job(db_session_factory, state=JobState.QUEUED)
    await _login(client)

    resp = await client.get("/jobs?state=nonsense", headers={"accept": "text/html"})

    assert "ignored the unknown filter value" in resp.text
    assert str(job_id)[:8] in resp.text


async def test_a_failure_reason_is_escaped_rather_than_rendered(db_env, db_session_factory, client):
    """A failure reason quotes subprocess output and vendor responses. None of it is this
    pipeline's bytes."""
    job_id = await _seed_job(
        db_session_factory,
        state=JobState.FAILED,
        stage="acquire",
        failure_reason="<script>alert(1)</script>",
        log_tail="<img src=x onerror=alert(2)>",
    )
    await _login(client)

    resp = await client.get(f"/jobs/{job_id}/detail", headers={"accept": "text/html"})

    assert "<script>alert(1)</script>" not in resp.text
    assert "&lt;script&gt;" in resp.text
    assert "<img src=x" not in resp.text


async def test_an_unknown_run_id_answers_404_with_a_way_back(db_env, db_session_factory, client):
    await _login(client)

    resp = await client.get(f"/jobs/{uuid.uuid4()}/detail", headers={"accept": "text/html"})

    assert resp.status_code == 404
    assert "no run with that id" in resp.text
    assert 'href="/jobs"' in resp.text


# --- the database not answering -----------------------------------------------------------
#
# These take the `client` fixture WITHOUT `db_env`, so `POSTGRES_HOST` keeps its unreachable
# default and the reads genuinely fail. Measured: an unresolvable host escapes SQLAlchemy as a
# bare `socket.gaierror`, which is why the handlers catch `OSError` beside `SQLAlchemyError`.


async def test_the_screen_survives_a_database_that_does_not_answer(client, monkeypatch):
    """The launch controls and the driver postures need no database, so one dead query must
    not take the whole screen down with it."""
    install_drivers(monkeypatch, FakeDriver("pixel"))
    await _login(client)

    resp = await client.get("/jobs", headers={"accept": "text/html"})

    assert resp.status_code == 200
    assert "this did not load" in resp.text
    assert "this one spends money" in resp.text
    assert '<option value="pixel"' in resp.text


async def test_a_dead_database_is_asked_once_rather_than_every_three_seconds(client):
    await _login(client)

    resp = await client.get("/jobs/list/rows")

    assert resp.status_code == 200
    assert "this did not load" in resp.text
    assert "hx-trigger" not in resp.text
    assert "try again" in resp.text


async def test_a_run_detail_survives_a_database_that_does_not_answer(client):
    await _login(client)

    resp = await client.get(f"/jobs/{uuid.uuid4()}/detail", headers={"accept": "text/html"})

    assert resp.status_code == 200
    assert "this did not load" in resp.text
    assert "hx-trigger" not in resp.text


# --- auth ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    ["/jobs", "/jobs/list/rows", "/jobs/00000000-0000-0000-0000-000000000000/detail"],
)
async def test_every_screen_route_is_behind_the_default_deny_middleware(client, path):
    resp = await client.get(path)
    assert resp.status_code == 401


@pytest.mark.parametrize(
    "path", ["/jobs/launch/devices", "/jobs/launch/firmware", "/jobs/launch/classification"]
)
async def test_every_launch_route_is_behind_the_default_deny_middleware(client, path):
    resp = await client.post(path, data={})
    assert resp.status_code == 401
