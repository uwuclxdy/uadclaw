"""Worker entrypoint. Starts, logs, stays alive.

The real job-claim loop lands in task 2; this scaffold only proves the process boots and
runs under the worker container's writable scratch mount.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def run() -> None:
    logger.info("worker starting")
    await asyncio.Event().wait()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())


if __name__ == "__main__":
    main()
