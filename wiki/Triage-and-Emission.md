# Triage and emission

**The human gate a proposal passes through, and what happens to the ones it approves.**

## The ranked queue

`triagestore.load_rows` returns every package the model has answered, ordered by `package_facts.device_count` descending, nulls last, then package name. `device_count` is a stored column rather than a join computed at read time, precisely to be this ranking signal: it counts phones that shipped the package, not firmware images, since `device_scans.device_key` is `<driver>:<device>` without the build.

The whole read runs on one `REPEATABLE READ` snapshot (`triagestore._begin_snapshot`), because it issues several statements and Postgres reads committed per statement. A decision landing between the joined-rows query, the decisions query and the shipped-branches query would otherwise be visible to one and not the others, and the screen would render a stale count beside fresh rows. `_begin_snapshot` has to be the first statement on a fresh session, so `load_board` and `load_approved` both call `load_rows` first.

Each `QueueRow` carries the package, its device count, whether it has an icon, the model's `removal`/`confidence`, the computed `floor`, the corroboration status, whether `package_facts.has_conflict` is set, whether the classification is parked, and its current decision.

### Views

`triagestore.VIEWS` is `("queue", "deferred", "decided", "shipped", "parked")`. `in_view` decides which one a row belongs to, based on `current_decision`:

| View | Membership |
|---|---|
| `queue` | no current decision, or the current decision is `edit`/`reopen` (not verdicts) |
| `deferred` | current decision is `defer` |
| `decided` | current decision is `approve`/`reject` and it has not shipped |
| `shipped` | current decision is `approve` and a `branch_emission_package` row names it |
| `parked` | the classification itself is parked (checked before every other view) |

Nothing is filtered out of the queue for looking weak. A conflicted package and an uncorroborated one both reach it, flagged rather than hidden: 134 of 147 packages shared between the Pixel 6 and Android 16 emulator corpora disagree on their signing certificate (ordinary key rotation or the AOSP test key facing the production key), and corroboration lands at 13.6% overall, 0 of 12 for `com.android.*` names. A screen that hid either signal would hide most of the corpus.

## The per-candidate card

`triagestore.load_candidate` builds one `Candidate`: the `package_facts` evidence rows (label, devices, partitions, flags, shared uid, cert issuer, libraries, declared queries, protected broadcasts, intent filters), the model's proposal, the computed floor and its reasons, the dependency edges, the corroboration verdict and its cited sources, and up to four upstream neighbours from `bundle.nearest_entries` (the same anchors the classification bundle showed the model).

`Candidate.missing` names the evidence classes the card does not have, rather than rendering a quietly incomplete layout: firmware facts, a minimum rating, sources, upstream neighbours, or a field the model declared it could not determine. Corroboration at 13.6% means most cards are missing sources, and that is an ordinary outcome the card states rather than hides.

## Decisions

`package_triage_decision` is append-only. Nothing in `triagestore` ever updates or deletes a row. `ACTIONS` is `("approve", "reject", "defer", "edit", "reopen")`; `VERDICTS` is `{"approve", "reject"}`. A `reject` requires a non-blank `reason`, enforced twice: once in `record_decision` (raises `ReasonRequired`) and once as a database `CheckConstraint` on the table itself, because a rejection with no reason is exactly the row that looks fine until somebody tries to learn from the log a month later.

Every decision is keyed on `bundle_sha256`, the classification's evidence hash at the moment of the decision, not on the package name alone. `current_decision` is the one function that decides whether a decision counts: the newest decision for a package, and only when its `bundle_sha256` still matches the bundle hash on the classification row today. A re-classification changes that hash, so an old decision stops being current on its own, with nothing deleting it. The package returns to the queue carrying its full history, and the append-only log keeps every rejection ever recorded against it, which is the only measurement this pipeline has of whether its review funnel is improving.

`edit` and `reopen` are not verdicts. An edit is a revision that still needs approval; `reopen` is the only way back out of a verdict, since the log is append-only and nothing else returns a decided package to the queue. `apply_edit` writes through `classifystore`, the only module that touches `package_classification`, so an edited `removal` below its computed floor is refused on the same seam a model's own writes pass through.

## Emission

### Vendor batches

`emissionstore.load_approved` reads the approved set for one vendor. `emission.vendor_for` reads the driver half of each approved package's `device_key`s (the `<driver>:` prefix), never the package name: a name-prefix heuristic once swept 70 Xiaomi packages into an Oppo bucket. One driver across every device key is that driver's vendor; more than one is `emission.SHARED_VENDOR` ("shared"). `emission.group_by_vendor` splits an approved batch this way and refuses a package that appears twice.

### `ApprovedPackage.floor` is not optional

`package_analysis.floor` is nullable (`NULL` means the rule ladder has not run for that package). `ApprovedPackage.floor` is a plain `str`. `emissionstore._refusal` is what enforces the gap: it refuses a row whose `floor` is `None` before an `ApprovedPackage` for it can be built, because a package with no computed floor has no bound for its rating to sit above. The dangerous alternative is writing `floor=row.floor or "Recommended"` at the construction site: `danger_rank("Recommended")` is 0, so that spelling produces a floor that is structurally present, semantically absent, and passes every downstream check silently.

### Append-only splice

`emission.insert_entries` appends approved entries to the bytes of `uad_lists.json` without re-serializing the document. The closing brace is located with `json.JSONDecoder().raw_decode`, not a backward `rfind("}")`, so nothing is guessed about a brace inside a description string, trailing whitespace, or a BOM. New entries are written in `emission.DOMINANT_KEY_ORDER` (`list, description, dependencies, neededBy, labels, removal`), matching the file's own majority key order rather than whatever a neighbouring entry happens to use. The spliced result is re-parsed and compared against the original mapping plus exactly the entries requested, a structural check rather than a claim of correctness by construction, since this is the one function in the repo that could otherwise hand back a corrupt `uad_lists.json`.

`emission.already_carried` is a pre-filter: it drops packages the destination file already has a key for, so a vendor can ship a second batch after its first has merged upstream. `insert_entries` itself still refuses whole if any of its packages is already carried, as a safety net rather than the normal path, because two entries under one key make a document whose meaning depends on the reader.

### Snapshot and ordering

`load_approved` rides the same `REPEATABLE READ` snapshot `triagestore.load_rows` opens, so the approved set and every package's shippable detail (description, floor, dependencies, device keys) are read as one instant. A decision landing mid-read here would otherwise commit a different batch than the one a reviewer saw approved.

The intent row is written and committed before any git command runs. `stages.branch_stage` calls `emissionstore.record_intent`, then `upstreamrepo.emit_branch`, then `emissionstore.record_commit`. Recording first can only lose the *outcome*, and the outcome is still readable off the clone afterward: the branch is there or it is not, and its committed bytes say whether it is this pipeline's. Emitting first would lose the *batch*, since nothing in the clone records which packages, ratings and floors went in, and `emit_branch` refuses a branch name that already exists, so a retry after a crash could not tell its own earlier emission from somebody else's branch of the same name.

### PR body

`emission.render_pr_body` states plainly that `list` and `description` are model-generated and human-reviewed, that `removal` is bounded below by the computed floor and never corrected up to it, and that `dependencies`/`neededBy` are never model output. It links each package to the evidence bundle it was classified from (a hash, or a link when `emission_bundle_base_url` is set).

## `upstreamrepo.py`: writing into somebody's live checkout

`upstreamrepo.py` is the only module in this repo that writes outside the project: it commits a branch into a git clone the operator supplies. It never pushes, fetches or authenticates; there is no GitHub credential anywhere in this stack. The deliverable is a local branch plus a PR body on the emission row, and a human runs `git push` from the clone afterward.

**Identified by content, not by remote.** The clone may be a fork, so nothing here checks its remote URL, remote name or default branch name. `inspect_repo` validates it by content instead: the configured `list_path` has to hold parseable, non-empty JSON at the resolved `base_ref`. This cuts both ways. The original, stale `0x192/universal-android-debloater` repo carries the same list at the same path and passes this check too, so an emission against it would splice into 2024 bytes; the recorded `base_commit` on the emission row is the only on-machine evidence of which clone was actually used.

**Every guarantee is asserted after the fact.** `inspect_repo` refusing a dirty tree does not bound what happens between that check and the commit. Three shapes were measured, each of which passed an earlier version of the guard:

| Attack shape | What made it look safe | Fix |
|---|---|---|
| The `@` HEAD synonym | `git check-ref-format --branch @` exits 0 and echoes `@` back; `git checkout @` resolves to the previously checked-out branch, so the checkout "succeeds" and moves nothing, landing the whole emission on the operator's own branch | `emit_branch` asserts `symbolic-ref --quiet HEAD` equals `refs/heads/<branch>` right after checkout, before a byte is written |
| Commit read-back bounding the commit, not the branch | A read-back that checks only the committed blob and touched paths passes even when a human commits mid-window: their commit rides underneath the emission's, invisible to a diff of the emission's own commit | `rev-list --parents -n 1 <head>` must equal exactly `[base_commit]`, which bounds the branch's lineage instead of one commit's own diff |
| The empty `status --porcelain` of an interrupted rebase | A clone stopped mid-`rebase -i` at an `edit` step stages nothing, so `status --porcelain` reads clean while the operation is still open | `_interrupted_operation` checks six named markers (`rebase-merge`, `rebase-apply`, `MERGE_HEAD`, `CHERRY_PICK_HEAD`, `REVERT_HEAD`, `BISECT_LOG`) via `--git-path`, before the dirty check |
| Recovery adopting a commit made in the crash window | Comparing only the list blob's digest adopts whatever commit sits at the branch tip; a human's unrelated commit into the same clone during that window doesn't touch the list file, so the blob digest still matches | `verify_emitted_branch` makes the same three read-backs `emit_branch` makes about its own commit: parent, touched paths, bytes |

### The three read-backs

Both `emit_branch` (about the commit it just made) and `verify_emitted_branch` (recovering a branch a crashed run already cut) make the same three checks, because bounding the file does not bound the branch:

| Read-back | Command | What it bounds |
|---|---|---|
| Parent | `git rev-list --parents -n 1 refs/heads/<branch>` equals exactly `[base_commit]` | the branch's lineage: exactly one commit, cut from exactly the recorded base |
| Touched paths | `git diff-tree --no-commit-id --name-only -r -z <head>` equals exactly `[list_path]` | the commit's diff: one file, nothing else added or amended in |
| Bytes | sha256 of `git cat-file blob <head>:<list_path>` equals the recorded `list_sha256` | the content: a hook that reformats `uad_lists.json` is a documented upstream rejection reason |

The parent check alone is not enough: `git commit --amend` preserves the parent while adding a file, so parent-plus-blob both pass a commit that carries something extra. All three run together.

`verify_emitted_branch` compares against the digest the *intent row* recorded, never a freshly re-derived batch, so a reviewer approving one more package between the crash and the retry cannot make a perfectly good branch look wrong.

### Why every ref is `refs/heads/<branch>`

Every check spells the ref as `refs/heads/<branch>` in full rather than the short branch name. Git resolves a bare name through `refs/tags/` first if a tag of that name exists, so a tag shadowing the branch would answer a short-name lookup instead of the branch itself. The dangerous direction is a rogue branch sitting under a clean-looking tag.

### Rollback

`upstreamrepo._restore` undoes a failed `emit_branch`. It never passes `--force`. It discards only `list_path`, the one file this emission owns, with a plain `checkout`, then switches HEAD back to the original ref with another plain `checkout`, then deletes the branch it created with `branch -D`. A forced switch would destroy a human's in-flight edit to any other file in the clone, with nothing holding it: no stash, no reflog, no object ever written. A plain checkout that refuses because a real conflict exists is treated as the correct outcome and reported, not overridden.

`_forget_if_rolled_back` (in `stages.py`) reads the clone back after a failed `emit_branch` and deletes the intent row only when the branch is confirmed gone. If the branch still exists, the intent is kept so the next run's recovery path (`_reconcile_emission`) can verify and adopt it rather than guess.

### Crash-resumability states

`stages.branch_stage` resolves four states on every run, based on what the `branch_emission` row and the clone each say:

| Row state | Branch present | Meaning | Action |
|---|---|---|---|
| no row | n/a | nothing happened yet | emit |
| row with a commit | n/a | this job is done | no-op |
| row with no commit | yes | the commit landed, recording did not | `_reconcile_emission` verifies it and records `reconciled=True` |
| row with no commit | no | ambiguous: died before committing, or committed-pushed-and-had-its-branch-deleted before recording | refused; a human clears the row by hand |

The last row is the one state genuinely ambiguous from inside the clone. Re-emitting would be the guess that ships a batch upstream twice, so the ambiguity resolves toward refusal. `_forget_if_rolled_back` is what keeps this from swallowing an ordinary failed attempt: a retryable failure whose branch did roll back deletes its own intent and lands in the first state on the next try, instead of the last one.

`branch_emission` jobs are serialized against each other with `stages._emission_clone_lock`, an `asyncio.Lock` inside the worker process. `worker_pool_size` can exceed 1 and `BRANCH_EMISSION` does not take the scratch lease (it writes the clone, not scratch), so without this lock two concurrent emission jobs would checkout-and-commit into the same clone and fail each other's read-backs.

See [Classification](Classification) for how a proposal reaches this queue, [Corroboration](Corroboration) for the sources shown on the card, [Rule-Ladder](Rule-Ladder) for the computed floor, and [Jobs-and-Worker](Jobs-and-Worker) for the `branch` stage's place in the job walk.
