# Classification

**The model proposes one field, `description`. Everything else is derived, floor-bounded, or rejected outright.**

Classification is its own job kind (`JobKind.CLASSIFICATION`), separate from `firmware_analysis`. `models.JOB_KIND_STAGES` decides what a job runs, never the module-global `PIPELINE_STAGES` vocabulary: the worker no-ops any stage with no registered handler, so one global walk plus a registered `llm` handler would classify the entire corpus against a paid API the moment any firmware job finished unpacking. `JOB_KIND_STAGES[JobKind.FIRMWARE_ANALYSIS]` ends at `rule_ladder`. `JOB_KIND_STAGES[JobKind.CLASSIFICATION]` is `("llm", "corroborate")`, queued separately by a human. See [Jobs-and-Worker](Jobs-and-Worker) for the stage walk mechanics.

## The evidence bundle is the reproducibility anchor

DeepSeek has no seed and no determinism guarantee, not even on a cache hit. Nothing about the model's answer can be pinned. What CAN be pinned is exactly what it was asked, and that is `bundle.py`'s job.

`build_bundle` assembles one package's facts, provenance, graph edges, computed floor and upstream style anchors into a payload, then hashes it. `canonical_json` is the one serialization the hash is ever taken over: sorted keys, tightest separators, no timestamp, no run-scoped id. Two runs over the same corpus produce byte-identical bundles and the same `sha256`, whatever order the packages arrive in.

- **Pure.** `bundle.py` touches no database, no network, no clock, the same posture as `ladder.py`.
- **Content-addressed.** `EvidenceBundle.sha256` is `bundle_sha256(payload)`. `classifystore.existing_bundle_hashes` compares it against the hash of a freshly built bundle: an unchanged hash means the question was already asked, and `select_candidates` skips it without spending an API call.
- **Floor and evidence graph travel as data the prompt shows the model.** `dependencies` and `neededBy` never travel as anything the model may edit; see [Rule-Ladder](Rule-Ladder) for why those two fields are graph-owned.

`nearest_entries` picks up to `DEFAULT_ANCHOR_COUNT` (4) existing upstream entries as style anchors, ranked by longest shared dotted package-name prefix, ties broken by name. A package needs at least `MIN_ANCHOR_SHARED_SEGMENTS` (2) shared segments to get an anchor at all; a package with no namespace neighbor gets none rather than the alphabetically nearest strangers.

## `llm.py`: the only retry layer

`LlmClient` owns retry, backoff and the concurrency bound, and nothing else in this repo may add a second layer. The repo rule is one retry layer per concern: a second layer in the stage would multiply the caps, turning a documented 3 attempts into 9 silent ones against a paid API. One client serves every provider: `Settings.llm_providers` is the operator's provider table, and a classification job's `provider` param names which entry the client is built from (default `deepseek`), so a new provider is a table row plus a key file, never a second client.

It calls the OpenAI-format `/chat/completions` endpoint rather than the `/anthropic` one, because only that envelope returns `usage.prompt_cache_hit_tokens` and `usage.prompt_cache_miss_tokens`, which is the cost and cache measurement this project reads off real production rows. `response_format` is `{"type": "json_object"}`; DeepSeek has no `json_schema` strict mode, so the hand-written validator in `classify.py` is the real gate either way.

### Five failure classes

| Error | Trigger | Retryable |
|---|---|---|
| `LlmAuthError` | HTTP 401/403 | no |
| `LlmBalanceError` | HTTP 402 insufficient balance | no, will not resolve inside a retry window |
| `LlmBudgetError` | `finish_reason == "length"`, or empty content with reasoning at the ceiling | no, raising `max_tokens` is the fix |
| `LlmMalformedError` | empty content with `finish_reason == "stop"`, or a non-JSON body | yes |
| `LlmUnavailableError` | HTTP 429/500/503, or a transport failure | yes |

### The budget failure and the empty-content bug are opposite fixes

`_check_budget` separates a too-small `max_tokens` from DeepSeek's documented, unresolved empty-content bug, because both present as an unusable body and the envelope alone cannot distinguish them. Thinking is on by default and billed against `max_tokens`, so a call can spend its whole budget on reasoning and leave nothing for the JSON body. `_REASONING_PRESSURE_RATIO` (0.9) decides how close the reasoning spend has to get to the ceiling before an empty body reads as a budget failure rather than the malformed-response bug. Treating a budget failure as malformed burns the whole retry cap on a request that cannot succeed at that ceiling; treating the malformed-response bug as a budget failure never retries a request a bigger budget would not have fixed anyway. The deepseek entry's `max_tokens` defaults to 16384 and its `thinking` defaults to `true`; setting `thinking` `false` removes the reasoning spend entirely rather than halving it, and sends the DeepSeek-specific `{"thinking": {"type": "disabled"}}` wire field, which settings load refuses on any entry that is not deepseek, naming the entry.

`require_json_prompt` refuses to send a prompt that does not contain the word "json" and a `{`-shaped example, per DeepSeek's own JSON-mode guide, because without both the model may emit an unbounded whitespace stream until it hits `max_tokens`.

## `classify.py`: the validator

Pure, no database or network. This is the safety-critical half of the stage: the question it answers is not "can this response be made usable" but "is this response allowed to exist at all."

`ModelProposal` is the response shape (`description`, `list`, `removal`, `confidence`, `unknown_fields`, `reasoning_brief`), validated with `extra="forbid"`. Before that validation even runs, `_check_forbidden_keys` rejects a response carrying `dependencies`, `neededBy`, `needed_by` or `labels` at all: a wrong edge silently changes a removal rating, so a response that writes one of those keys is refused rather than having the key dropped.

### A below-floor answer is rejected, never clamped

`_check_removal` calls `ladder.is_below_floor`, never `ladder.raise_to_floor`, on a model response. A model that answered below the floor did not make a rounding error: it read evidence saying `coreApp="true"` and concluded the package is safe to remove, so its `description` and its `list` came out of that same misreading. Silently raising the number to the floor would keep the misreading and hide it behind a rating that now looks correct. `raise_to_floor` is never called on model output anywhere in this repository.

### `list` is decided for two rules only

Design intent was "deterministic from cert issuer plus partition plus prefix." Measured against the live 5372-entry upstream list, only two rules clear the bar for rejecting a model answer:

- an OEM name token or a matching certificate-issuer organization decides `Oem` (98.7% agreement, n=3147)
- the AOSP test-key marker (`android@android.com`) plus an `android.`/`com.android.` namespace decides `Aosp`

`derive_list` runs the OEM checks first, because an OEM's overlay of an AOSP package still lives in the AOSP namespace and is upstream's `Oem`; the reverse order mislabels 316 live entries. Everything else is `ListDerivation(value=None, ...)`, meaning undecided, and the model's answer stands. `_check_list` rejects a model `list` only when `derivation.decided` is true and the model disagrees.

### Other checks

- `description` must be `DESCRIPTION_MIN_CHARS` (20) to `DESCRIPTION_MAX_CHARS` (600) characters, no leading or trailing whitespace, no space adjacent to a `\n` escape (a literal upstream maintainer request on PR #1180). A description declared in `unknown_fields` is exempt from the lower bound and must read exactly `unknown`.
- `unknown_fields` may name only `description`. `removal` always has a value because the floor supplies one; `list`'s catch-all is `Misc`.
- `reasoning_brief` is capped at `REASONING_BRIEF_MAX_CHARS` (400).

Every accepted `Classification` carries a `provenance` map (`provenance_for`): `description`, `removal`, `confidence` and `reasoning_brief` are `llm:<model>`; `list` is the deciding rule's name when a rule decided it, `llm:<model>` otherwise; `dependencies`, `neededBy` and `floor` are stated as `graph:corpus` and `rule:ladder` even though this module never writes them, so their authority is auditable per row.

## `classifystore.py`: the enforcement seam

`package_classification` is a separate table from `package_analysis`, sharing no column with it, so a re-run of the classification stage cannot touch an edge or a floor even by mistake.

### `_upsert` is where the floor gate actually lives

The floor check used to live only in `classify.validate_response`, which held exactly as long as the model was the sole writer. Triage is a second writer: `store_human_edit` writes `human:`-provenance ratings with no `validate_response` anywhere in its path. `_upsert` is the one seam all three writers (`store_classification`, `park_package`, `store_human_edit`) pass through, so `_refuse_below_floor` raises `BelowFloorError` there rather than being re-implemented per caller.

### `removal_superseded`: when a human edit and a rising floor collide

Two rules this module owes can conflict on one field: a `human:`-owned value survives a model re-run untouched, and nothing may rank below its floor. The floor only rises, so a stored `human:` removal can go stale with nobody editing anything. `_preserved` checks every human-owned field against the current floor via `_floor_refusal`; a `removal` the floor would now refuse is dropped from what gets preserved rather than carried into a write the floor gate would refuse in turn. `_apply_superseded` records the drop as a `removal_superseded` provenance entry (`SUPERSEDED_BY = "rule:floor superseded"`) the triage card renders, so the reviewer whose edit was overruled can see that it was. Nothing else the human wrote is touched, because the floor constrains only `removal`.

### `park_package` clears a stale proposal, never a fresh one

A previous accepted proposal survives a park only when the park is against the SAME `bundle_sha256`. A park against a new bundle clears `description`, `uad_list`, `removal` and `confidence`, because carrying an old answer forward under a new evidence hash would assert the old rating answers evidence nobody validated it against, exactly the failure mode `raise_to_floor` never being called on model output is built to prevent. A field a human owns survives either way.

`store_human_edit` runs every edited field through the SAME `_check_description` that a model response goes through, imported across the module boundary rather than re-implemented, so a human-written description obeys the same space-adjacent-to-newline rule upstream's reviewer enforces.
