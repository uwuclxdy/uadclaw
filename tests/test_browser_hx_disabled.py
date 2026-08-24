"""The in-flight disabled state, observed firing in a real browser for the first time.

`hx-disabled-elt` had a test for its PRESENCE (`test_dashboard_jobs.py`) and had never
been seen to FIRE, because nothing in the suite executes htmx's JS during an in-flight
fetch. This boots the real app on
a loopback port and drives it with system chrome through playwright. The load-devices
index fetch — the multi-second vendor request the attribute exists for — is slowed
inside the driver, so the in-flight window is observable; the button being disabled
WHILE the request is pending and enabled again after the swap is the firing. With
`hx-disabled-elt` removed from the template this test goes red.

Opt-in like the heavy tier: skipped unless UADCLAW_BROWSER_TESTS=1, and skipped again
when playwright or a browser cannot be resolved. Run with -n0 for a lone run; under
the suite's -n auto the loadscope distribution keeps the single file on one worker,
so a parallel run is safe too. One browser, one port, and the per-checkout DB prefix
keeps the suite's parallel contract.
"""

import asyncio
import os
import shutil
import socket

import pytest

from uadclaw import firmware as firmware_module
from uadclaw.firmware import FirmwareDriver, FirmwareRef, TermsPosture, TermsRisk
from uadclaw.views import jobs as jobs_view

pytest.importorskip("playwright")

PASSWORD = "test-only-admin-password"
CHROME = shutil.which("google-chrome-stable")
# How long the index fetch takes: long enough that the disabled state is observable,
# short enough that the whole test stays inside the timeout.
INDEX_FETCH_DELAY_SECONDS = 1.5

pytestmark = [
    pytest.mark.timeout(120),
    pytest.mark.skipif(
        os.environ.get("UADCLAW_BROWSER_TESTS") != "1",
        reason="opt-in: set UADCLAW_BROWSER_TESTS=1 (needs playwright and a chrome)",
    ),
    pytest.mark.skipif(
        CHROME is None,
        reason="no chrome on PATH (google-chrome-stable)",
    ),
]


class SlowListingDriver(FirmwareDriver):
    """A driver whose index fetch takes INDEX_FETCH_DELAY_SECONDS, so the request is
    still in flight while the test looks at the button. `calls` counts real
    `list_available` runs, which is how a second submission would be detected."""

    def __init__(self) -> None:
        self.calls = 0

    def terms(self) -> TermsPosture:
        return TermsPosture(
            risk=TermsRisk.PUBLIC,
            source_url="https://example.invalid/pixel",
            summary="a source with nothing to accept.",
            acknowledged=True,
        )

    async def list_available(self) -> list[FirmwareRef]:
        self.calls += 1
        await asyncio.sleep(INDEX_FETCH_DELAY_SECONDS)
        return [
            FirmwareRef(
                driver="pixel",
                device="comet",
                build="AP4A.250105.002",
                url="https://example.invalid/comet-AP4A.250105.002.zip",
                marketing_name="Pixel 9 Pro Fold",
            )
        ]

    async def fetch(self, ref, dest_dir):  # pragma: no cover - never reached by this screen
        raise AssertionError("the jobs screen must never download firmware")


def _free_port() -> int:
    """Ask the kernel for a free loopback port. The bind is released before uvicorn
    takes it, so this is a hint rather than a claim; on a box with no concurrent
    listeners the gap is negligible and the server owns the port from then on."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_until(predicate, *, what: str, timeout: float = 15.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


async def test_load_devices_disables_the_button_for_the_in_flight_fetch(
    db_env, monkeypatch, tmp_path
):
    """The whole point of `hx-disabled-elt` on this screen, observed rather than pinned:
    the button answers the click by disabling for the duration of the index fetch, and
    the same button — it sits OUTSIDE the swap target — is enabled again after the
    swap. The presence test stays green with the attribute dead inside htmx's JS; this
    one cannot, which is the distinction."""
    import uvicorn
    from playwright.async_api import Error, async_playwright

    from uadclaw.app import create_app

    driver = SlowListingDriver()
    monkeypatch.setattr(
        firmware_module, "_driver_factories", lambda: {"pixel": lambda settings: driver}
    )
    # The index cache is process-local and other tests in this worker may have listed a
    # different pixel index: a cached answer would skip the driver entirely and the test
    # would observe a request that never happened.
    jobs_view._device_index_cache.clear()

    app = create_app()
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="off")
    )
    # A task on THIS loop rather than a thread of its own: the app's engine pool is
    # created on the loop the first request runs on, and conftest's `_reset_caches`
    # disposes that engine on the test loop — a thread-run server leaves its connections
    # on a loop that is already dead and dispose raises inside asyncpg.
    server_task = asyncio.create_task(server.serve())
    try:
        await _wait_until(lambda: server.started, what="uvicorn startup")
    except AssertionError:
        # A failed bind leaves the real exception in the task, where it would read as
        # a generic timeout and never be retrieved. Surface the actual cause.
        if server_task.done():
            server_task.result()
        raise
    try:
        base_url = f"http://127.0.0.1:{port}"
        async with async_playwright() as playwright:
            try:
                browser = await playwright.chromium.launch(
                    executable_path=str(CHROME), headless=True
                )
            except Error as exc:
                pytest.skip(f"could not launch chrome: {exc}")
            try:
                page = await browser.new_page()
                await page.goto(f"{base_url}/login")
                await page.fill('input[name="password"]', PASSWORD)
                await page.click('button[type="submit"]')

                await page.goto(f"{base_url}/jobs")
                # The form's select defaults to "choose one" and the handler answers an
                # empty driver instantly — the click only becomes the multi-second
                # request the attribute exists for once a driver is chosen.
                await page.select_option("#driver-select", "pixel")
                button = page.get_by_role("button", name="load devices")
                picker = page.locator("#device-picker")

                # The response is RECORDED, never awaited here: `expect_response`'s
                # context manager blocks until the response arrives, which is the whole
                # in-flight window the test exists to observe.
                responses: list[object] = []

                def record(resp) -> None:
                    if "/jobs/launch/devices" in resp.url:
                        responses.append(resp)

                page.on("response", record)
                await button.click()

                # In flight: the button is disabled and the picker has not been swapped.
                # This is the state the presence test never reaches, and the driver's
                # delay is what keeps it observable here.
                await page.wait_for_function(
                    """() => {
                        const button = [...document.querySelectorAll("button")]
                            .find((b) => b.textContent.includes("load devices"));
                        return Boolean(button && button.disabled);
                    }""",
                    # Timer polling rather than the default rAF: deterministic in
                    # headless chrome, where rAF can go quiet while nothing repaints.
                    polling=100,
                    timeout=5000,
                )
                await button.scroll_into_view_if_needed()
                shot = tmp_path / "hx-disabled-inflight.png"
                await page.screenshot(path=str(shot))
                print(f"in-flight screenshot: {shot}")
                assert "no device list yet" in await picker.inner_text()

                # The swap landed and the SAME button (outside the swap target) is
                # enabled again, still carrying its keybind for the next press. One
                # predicate for both, so an early re-enable cannot race the swap.
                await page.wait_for_function(
                    """() => {
                        const button = [...document.querySelectorAll("button")]
                            .find((b) => b.textContent.includes("load devices"));
                        const picker = document.querySelector("#device-picker");
                        return Boolean(
                            button && !button.disabled
                            && picker && picker.textContent.includes("Pixel 9 Pro Fold")
                        );
                    }""",
                    polling=100,
                    timeout=5000,
                )
                assert [resp.status for resp in responses] == [200]
                assert driver.calls == 1
            finally:
                await browser.close()
    finally:
        server.should_exit = True
        if not server_task.done():
            await server_task
