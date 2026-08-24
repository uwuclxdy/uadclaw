# Jobs and worker

**How a pipeline run is claimed, walked stage by stage, and recovered after a crash.**

## Job kinds and the stage walk

Three job kinds exist (`models.JobKind`): `firmware_analysis`, `classification`, `branch_emission`. `models.JOB_KIND_STAGES` maps each kind to the stages it walks, in order.

| Job kind | Stages walked | Needs the scratch lease |
|---|---|---|
| `firmware_analysis` | acquire, unpack, extract_facts, corpus_graph, filter, rule_ladder | yes |
| `classification` | llm, corroborate | no |
| `branch_emission` | branch | no |

`models.PIPELINE_STAGES` lists all ten stage names the design defines: the six above, plus `llm`, `corroborate`, `triage` and `branch`. It is vocabulary, not a walk order. Nothing may iterate it to decide what a job runs next. `triage` sits in that vocabulary with no handler and no kind that walks it: it is the human review step done through the triage screen, not a pipeline stage.

### Why per-kind stages, not one global walk

The worker no-ops any stage with no registered handler. `worker.default_stage_handlers` maps every `PIPELINE_STAGES` name to a no-op by default, and only the names in `stages.pipeline_stage_handlers()` get a real one. If a single global list decided every job's walk, a `firmware_analysis` job would classify the whole corpus against a paid API the moment it finished unpacking, because the `llm` handler exists and nothing kind-scoped would stop the walk from reaching it. `FIRMWARE_ANALYSIS` ends at `rule_ladder` for that reason. It used to no-op through `llm`, `corroborate`, `triage` and `branch` and finish at `branch` instead, recording four stage runs that did nothing.

`classification` and `branch_emission` are separate kinds a human queues on purpose. Classification spends an LLM provider budget on both its stages, plus a Brave search quota on the second. Emission commits into somebody else's git clone.

`jobs.stages_for(kind)` raises `JobValidationError` for an unrecognized kind rather than defaulting to the full pipeline. A new kind cannot silently inherit the firmware walk this way. `jobs.next_stage(current, kind)` reads the same table to find the stage after `current`, or `None` once the kind's last stage is done. A stage recorded before a kind's walk shrank (a firmware job once reached `corroborate` while `FIRMWARE_ANALYSIS` still no-oped through it) is treated as finished rather than raised on, via `jobs._is_retired_stage`.

## Job params

Every job carries a `params` JSONB column, shaped by its kind. `jobs.JOB_KIND_PARAM_MODELS` validates it at creation, before the row exists:

| Job kind | Params model |
|---|---|
| `firmware_analysis` | `firmware.FirmwareJobParams` |
| `classification` | `classify.ClassificationJobParams` |
| `branch_emission` | `emission.BranchEmissionJobParams` |

`validate_job_params` treats a missing `params` the same as `{}`. `{"kind": "firmware_analysis"}` and `{"kind": "firmware_analysis", "params": {}}` get the same validation answer. A bad target fails at creation, as a 422, rather than after a worker has claimed the job and waited on the scratch lease only to discover the driver or device is unknown.

## Claim, heartbeat, fence

`jobs.claim_job` takes the oldest queued job with `SELECT ... FOR UPDATE SKIP LOCKED`, inside `session.begin()`. The row lock and the flip to `CLAIMED` commit as one unit, so a second worker's `SKIP LOCKED` cannot see the row as still queued before the first worker's update lands. Each claim increments `Job.attempt`. That number, paired with the worker's own id, is the identity every later write for this run has to match.

`jobs._fenced_job` loads the job row `FOR UPDATE` only when it is still owned by `(worker_id, attempt)`. Every mutating call in `jobs.py` goes through it: `mark_running`, `heartbeat_job`, `advance_stage`, `complete_job`, `fail_job`, `mark_artifacts_deleted`. Each returns `None` or `False` the instant ownership has moved on. A worker that got reclaimed stops touching the job at that point rather than racing the new owner.

`worker._heartbeat_loop` starts before the scratch lease is acquired, not after. A job can block on a contended lease for longer than the stale window. Starting the heartbeat only once that wait ends would let a perfectly healthy job get reclaimed while it waits its turn. The loop refreshes the job's own heartbeat unconditionally, and the lease's heartbeat only while the kind needs scratch.

## Stale reclaim

`jobs.reclaim_stale_jobs` requeues any `CLAIMED`/`RUNNING` job whose `heartbeat_at` is older than `lease_stale_after_seconds` (300s by default). `stage` is left untouched, so the next claim resumes after the last completed stage rather than from the beginning.

A job already at `max_job_attempts` (5 by default) is parked `FAILED` instead of requeued. Claim order is FIFO by `created_at`, so a job that reliably kills its worker would otherwise return to the front of the queue on every reclaim, forever, ahead of every healthy job.

`worker._periodic_sweep_loop` runs this reclaim plus `scratch.enforce_retention_ceiling` on an interval (`sweep_interval_seconds`, 60s by default), not only once at startup. A job or lease orphaned by a slot that died mid-run, without killing the whole process, still recovers this way.

## Stage handler contract

`worker.StageHandler` is `Callable[[StageContext], Awaitable[None]]`. `StageContext` carries `job_id`, `attempt`, `scratch_dir` (`None` for a kind that does not need scratch) and a `session_factory`. Every real handler lives in `stages.pipeline_stage_handlers()` and follows three rules:

- writes only inside `ctx.scratch_dir`, never anywhere else on disk;
- never writes a job row. Job state, stage progress and the fence belong to the worker, which calls `jobs.record_stage_started`/`record_stage_finished`/`advance_stage` around the handler in `worker._run_one_stage`;
- an unregistered stage name falls through to a no-op and does nothing.

`_run_one_stage` calls `record_stage_started` first, itself fenced. A `None` run id means the job was fenced out before the stage could start, and the handler never runs at all. On success it records the stage finished, then advances the job's `stage` column. On an exception it records the stage failed and calls `fail_job` with the traceback. A fenced write at either point returns a `FENCED` outcome that aborts the job loop locally, recording nothing further.

## Scratch: lease and retention

`scratch.py` owns the single-occupant scratch lease and its own retention, independent of the worker loop; callers pass their own `session_factory`.

**Lease.** `ScratchLease` is a singleton row (`id = 1`, pinned by a `CheckConstraint`). `acquire_scratch_lease` polls `_try_acquire_once` until it holds the lease or a timeout elapses, reclaiming a holder whose `heartbeat_at` has gone stale along the way and logging a `ScratchLeaseEvent`. It is re-entrant: a job already holding the lease under its own `(job_id, worker_id)` returns immediately, without waiting out the stale window. `release_scratch_lease` no-ops, logged, unless the caller is the current holder by both ids, so a fenced-out worker cannot release a lease a newer attempt has since acquired for the same job id.

**Retention, two halves.** On success, `worker._run_job` commits `SUCCEEDED` first and calls `scratch.cleanup_scratch_dir` second, so a cleanup failure never rolls a finished job back into looking failed. Scratch directories are keyed on `(job_id, attempt)`, not `job_id` alone: a reclaimed job restarts under a new attempt with its own directory, so the old fenced-out attempt's `rmtree` on eventual success can never delete files the new attempt is still writing.

The second half is `enforce_retention_ceiling`, an oldest-mtime-first eviction that bounds total bytes under `scratch_root` regardless of job state. A ceiling keyed only on `state == FAILED` would miss real bytes on disk:

- an orphaned directory with no matching job row
- a partial `rmtree`
- a crashed job re-queued under a new attempt

The scan reads `(job_id, state, attempt)` once, walks the filesystem outside any transaction, then re-checks each candidate's current state immediately before deleting it (`_is_still_evictable`). The walk itself can take seconds, long enough for a slot to claim that exact job and start writing into a new directory. The whole sweep is serialized cluster-wide with a Postgres advisory lock, so a periodic sweep and a just-failed job's own cleanup cannot each read a stale, pre-eviction view and under-evict.

## Utilization figures (`stats.py`)

`GET /stats`, behind auth like every route, returns `stats.compute_stats`, used to tune `worker_pool_size`:

| Figure | Source | Note |
|---|---|---|
| `queue_depth_now` | live count of `QUEUED` jobs | a gauge computed at read time |
| `stage_durations` | `JobStageRun` rows with `outcome = 'succeeded'` | a stage that fails fast must not read as "got fast" |
| `worker_utilization` | `Job.started_at`/`finished_at` grouped by `worker_id` | a still-running job counts as busy via `coalesce(finished_at, now)` |
| `lease` | `ScratchLeaseEvent` history | occupancy percentage, average and max wait, current holder |

The window always ends at "now" and looks back `stats_lookback_seconds`, never at the last time something finished. A wedged system, nothing finishing and nothing releasing, would otherwise read as 100% healthy. `ScratchLeaseEvent` exists because the mutable `ScratchLease` singleton row cannot answer "how long was it held historically" once a later acquire or release has overwritten it.

## Worker pool cycle

`worker.run_worker_pool` reclaims anything a previous, now-dead worker left stuck, then runs `worker_pool_size` `_worker_slot` loops plus one `_periodic_sweep_loop` concurrently, until a shutdown event is set. Each slot:

1. claims a job, or waits `job_claim_poll_interval_seconds` and retries when nothing is queued;
2. marks it running and starts the heartbeat task;
3. acquires the scratch lease and prepares a scratch directory, only if the kind needs one;
4. walks `jobs.next_stage` in a loop, running each stage through `_run_one_stage`, until it returns `None` or a stage fails;
5. on success, commits `SUCCEEDED` then cleans up scratch; on failure, records it; in `finally`, stops the heartbeat, releases any held lease, and runs the retention ceiling sweep.

Whether a kind needs scratch (`JOB_KIND_NEEDS_SCRATCH`) is a property of the kind rather than of the worker loop, so `classification` runs concurrently with a `firmware_analysis` job holding the single-occupant lease instead of queueing behind it. It only touches the database and the network.

## Stages by job kind

| Stage | Job kind | What it writes |
|---|---|---|
| `acquire` | firmware_analysis | resolves the firmware ref against the driver, downloads the archive into scratch, records it in `pipeline-state.json` |
| `unpack` | firmware_analysis | unpacks the archive into partition images, extracts APKs and config XMLs as artifacts, deletes the archive and the partition images |
| `extract_facts` | firmware_analysis | parses every extracted APK, records a `device_scans` row, upserts `package_observations`/`package_facts`, deletes the APKs |
| `corpus_graph` | firmware_analysis | parses the `/etc` config XMLs, records `device_scans.config_inputs`, writes `package_analysis.dependencies`/`needed_by`/`edges`/`evidence` corpus-wide |
| `filter` | firmware_analysis | loads `uad_lists.json`, writes `package_analysis.upstream_present`/`queued`/`filter_verdict`/`upstream_provenance` |
| `rule_ladder` | firmware_analysis | computes removal floors from the corpus and the `/etc` inputs, writes `package_analysis.floor`/`floor_rule`/`floor_reasons`/`privapp_*` |
| `llm` | classification | classifies queued candidates against the job's provider, writes `package_classification` (a proposal, or a parked row) |
| `corroborate` | classification | searches Brave, fetches source pages, judges each description against the job's provider, writes `package_corroboration` and `package_search_results` |
| `branch` | branch_emission | commits an approved vendor batch into the operator's clone, writes `branch_emission`/`branch_emission_package` |

See [Firmware-Drivers](Firmware-Drivers), [Unpacking](Unpacking), [Facts-and-Corpus](Facts-and-Corpus) and [Rule-Ladder](Rule-Ladder) for the firmware-side stages; [Classification](Classification) and [Corroboration](Corroboration) for the model stages; [Triage-and-Emission](Triage-and-Emission) for `triage` and `branch`.
