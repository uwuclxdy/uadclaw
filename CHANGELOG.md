# Changelog

## Unreleased — 2026-08-14

### Features
- All settings loaded from a mounted secrets directory first and the environment second, over async postgres with alembic migrations.
- The worker pool claimed jobs with fenced writes and heartbeats, reclaimed stale attempts, and leased scratch one job at a time.
- The pipeline acquired firmware through a per-OEM driver: Pixel, Xiaomi, Nothing, Motorola, Samsung and Oppo/OnePlus/Realme.
- A driver whose index was not already chronological sorted it before returning, so no device silently resolved to a stale build.
- Samsung downloads came from FUS, decrypted in place, and verified size and CRC against the values the service declared.
- The Samsung driver probed a configured model and CSC grid and warned when a model resolved to no build at all.
- The Oppo driver resolved operator-named models off the update endpoint, kept the catalogue gate as fallback, and filed builds under ids that carried the region.
- Firmware downloads refused a redirect that downgraded https to http and fast-failed on a size mismatch.
- A pinned build kept the checksum its source published, so the archive downloaded integrity-unverified only when the source published none, recorded with a warning.
- Unpacking dispatched on container magic, never the file name or OEM, and extracted selectively.
- The Samsung chain read its tar stream and LZ4 frames straight out of the zip member, so the 11.47 GB tar never landed on disk.
- EROFS images went through fsck.erofs, since 7z had no EROFS handler, and the unpacker rebuilt Motorola sparsechunk sets in name order.
- The pipeline contained untrusted names from images, zips and filesystems before they became paths.
- An image or partition that yielded no file at all failed the job, and only named appless partitions could stay empty.
- APK facts came out of the manifest and merged across devices, idempotently, with danger flags sticky-true.
- The corpus graph emitted only overlay-to-target and required-library edges; declared queries and unresolved libraries stayed evidence.
- The dex string table yielded content:// references as evidence, at a fraction of a full DEX parse.
- The ladder read its /etc inputs: privapp allowlists, static roles and platform shared libraries.
- The rule ladder set a removal floor the model could raise and never lower; lowering was unrepresentable rather than forbidden.
- The additions queue filtered against the operator-supplied upstream list, and refused a missing or empty list.
- Every package got a content-addressed evidence bundle, whose sha256 was what an upstream PR body linked to.
- The DeepSeek client asked in json mode with bounded retries and separated a spent reasoning budget from the documented empty-content bug.
- The response validator checked model answers and rejected them, never clamped, when they fell below the floor, named an unknown list, or carried dependency claims.
- Classification ran as its own job kind that walked the llm and corroborate stages, so a firmware job never spent on the model.
- The API parked a package it gave up on with the reason, and a failed search or fetch never cancelled its siblings.
- A paid search landed in the database even when the page fetch failed.
- Every proposal that claimed something faced an independent check: Brave search, a bounded page fetch, and a judge that refused a citation it was never given.
- Search results cached on the package name, so a re-classification reused them and spent no quota.
- The fetch boundary refused the verdict-carrying transition blocks, 6to4 and teredo, by name, and normalised IPv4-mapped addresses to the embedded IPv4.
- A server-rendered dashboard with htmx served login, corpus, jobs, telemetry, triage and emission screens behind default-deny auth.
- The corpus screen browsed, filtered and searched packages, with a per-package detail view of floor, edges, evidence and conflicts.
- The jobs screen launched firmware jobs, listed vendor indexes on press, and showed run detail.
- The telemetry screen showed the pool utilization figures the pool size is tuned against.
- The triage screen took keyboard input, showed one ranked candidate at a time, and logged every decision.
- The pipeline extracted launcher icons from APKs and served them to the dashboard, with a deterministic monogram chip for the rest.
- Approved packages emitted through their own job kind as an append-only splice in dominant key order, never a reformat.
- The batch committed onto a branch in the operator's clone without pushing, fetching or authenticating, and the PR body carried the disclosure.
- Emission read the approved set in one repeatable-read transaction, and refused a package whose ladder never ran.
- Branch emission jobs serialized against the clone, and a retry could never double-ship.
- A shipped approval kept its own state, and the batch filtered against the destination's own bytes.
- Emission verified HEAD attachment and the commit's parent on the live clone; a failed run discarded only its own file, never a human's in-flight edit.
- Emission refused invisible characters and non-package-name keys, and stripped git identity and config injection from the commit environment.

### Improvements
- The triage card grew into three columns with icons, and the corpus rows shared the same icon lane.
- The triage queue and card read one current decision off one snapshot, so a mid-read decision could not skew the count.
- A reviewer could edit a row the model declared unknown, and a stale human removal no longer wedged its job.
- The write path enforced the removal floor, so human edits could not land below it either.
- Every screen answered an unreachable database with a visible error, and a dead database host no longer hung requests for a minute.
- Driver terms and unacknowledged risk were visible on the jobs screen without a click.
- Credentials stayed out of job errors and log tails, and a blank secret file deferred to the environment.
- A firmware job that named no target failed at creation, and a misspelt target passed shape validation only to fail at run time when the driver could not resolve it.
- The dashboard neutralised a package name that was itself a path token before it reached the wire.
- Icon extraction became total: a malformed drawable failed only its own package.
- A firmware job parked on a retired stage completed instead of lingering.

### Internal
- The worker image carried the full unpacking toolchain, git, and an erofs-utils floor; the scratch dir moved onto a host bind mount.
- Compose mounted the upstream list, the upstream clone and the worker credentials where they are read.
- The Python floor rose to 3.12.13, the version the container image ships.
- CI gated uv sync, ruff and pytest, and built the Dockerfile.
- Tests gained Playwright browser coverage and heavy real-chain fixtures for the Samsung, Oppo and emission paths.
