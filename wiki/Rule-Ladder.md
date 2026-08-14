# Rule ladder

**The removal floor: a lower bound on how dangerous removing a package is, that the model and a human may raise and can never lower.**

`removal` reaches real phones through Canta, AppManager and android-debloat-list as well as uad-ng, so a wrong `Recommended` is a failure that propagates past this project. `ladder.py` computes a floor for every package before the model ever sees it. A later stage may decide a package is *more* dangerous than its floor. Nothing may decide it is less.

## Danger order

`Removal` is a `StrEnum` with four values: `Recommended`, `Advanced`, `Expert`, `Unsafe`. Comparing them as strings sorts `"Expert" < "Recommended"`, which inverts the whole ladder. Every comparison in this codebase goes through `danger_rank()` instead, which ranks position in the explicit `DANGER_ORDER` tuple.

| Order | Tier |
|---|---|
| 0 (least dangerous) | Recommended |
| 1 | Advanced |
| 2 | Expert |
| 3 (most dangerous) | Unsafe |

`strictest(*removals)` returns the most dangerous of several tiers using `danger_rank`, and it is the only way two tiers combine.

## The floor is unrepresentable to lower

Four properties, held by shape rather than by review discipline:

- `Removal` orders by `danger_rank()`, never by its own string comparison.
- `compute_floors()` is the only constructor of a `RemovalFloor`. It takes the whole corpus and has no per-package variant, because three rules below are corpus-wide.
- `RemovalFloor` is frozen and validates on construction that its `floor` equals the strictest tier among its own `fired` rules. A hand-built floor cannot understate its own evidence.
- `raise_to_floor(proposal, floor)` is the only function that combines a proposal with a floor. It returns `max(proposal, floor)` by `danger_rank`. There is no function anywhere in the module that returns a tier below a floor.

`is_below_floor(proposal, floor)` exists separately for callers that need to reject rather than raise. See [Classification](Classification) for why the classification stage calls that one and never `raise_to_floor`.

## Which rules set which floor

`_base_rules` walks one package's facts and appends every `FiredRule` that fires. `compute_floors` runs this over the whole corpus, then adds an overlay pass. Ties between rules that set the same floor break on table order (`FloorRule` enum order), so the reported rule is stable across runs.

| Rule | Floor | Fires when |
|---|---|---|
| `core_app` | Unsafe | `coreApp="true"`: AOSP puts the package in the minimalist boot environment |
| `sole_library_provider` | Unsafe | the corpus's only provider of a required library some other package declares |
| `sole_static_role_holder` | Unsafe | the corpus's only holder of a static role with no alternative |
| `deny_list` | Unsafe | package name matches the hardcoded jurisdiction deny list |
| `system_shared_uid` | Expert | `sharedUserId="android.uid.system"` |
| `persistent` | Expert | `persistent="true"`: the platform keeps it running |
| `sole_core_intent_handler` | Expert | the corpus's only handler of HOME, DIALER, SMS_DELIVER, or IME |
| `privileged` | Advanced | in `priv-app`, or listed in a `privapp-permissions` allowlist |
| `overlay_target` | target's floor, capped at Advanced | `<overlay android:targetPackage="X">` |
| `default` | Recommended | no rule fired |

Measured over both local corpora (Pixel 6 `oriole` plus the Android 16 emulator, 393 merged packages): Unsafe 27, Expert 17, Advanced 265, Recommended 84. No static-role rule fired on either build because `roles.xml` was absent from both, and no sole-library-provider rule fired because neither corpus has a library edge (see [Facts-and-Corpus](Facts-and-Corpus)).

### Why three rules need the whole corpus

`compute_floors(corpus, config=...)` takes a `Sequence[CorpusPackage]`, never one package, because three rules are statements about what ELSE is installed:

- **Sole library provider.** `_sole_library_consumers` maps a provider to the consumers that would break, computed by scanning every package's declared and required libraries across the corpus.
- **Sole static role holder.** `_sole_static_roles` checks, per role, whether more than one corpus package holds it.
- **Sole core intent handler.** `_sole_intent_surfaces` checks whether more than one package handles HOME, DIALER, SMS_DELIVER or IME.

A function that could answer any of these from one manifest alone would silently return a floor that is too low the moment a second provider exists elsewhere in the corpus. There is deliberately no such function.

### The deny list

`DENY_LIST` is five hardcoded regexes matched against the lower-cased package name, each carrying its own `reason`: cellbroadcast, emergency, the CMAS/ETWS/WEA acronyms, AMBER, and SAR. The alert acronyms and AMBER and SAR are segment-anchored (`(^|\.)wea(\.|$)`) rather than bare substrings, because as substrings they match "weather", "wearable", a Motorola wallpaper package, and the Hungarian word inside `telekom.hu.android.mobilvasarlas`. This clause is hardcoded rather than a model call: the upstream wiki defines `Unsafe` as including packages "illegal to remove in your country," and no static analysis answers a jurisdiction question.

### The overlay pass

`compute_floors` runs a second pass after computing every package's base floor. A package with `<overlay android:targetPackage="X">` inherits `min(base_floor[X], Advanced)`. Reading the target's BASE floor, not its final one, keeps a chain or a cycle of overlays from compounding. The cap can only raise this package's floor, never lower it, because floors combine with `strictest`.

### The privileged-permission allowlist is an integration score

AOSP refuses to boot when a package that is still PRESENT requests a privileged permission nobody allowlisted. That is a ROM-build error, not a debloat outcome: removing the app removes the request. So allowlist membership contributes the same `Advanced` floor that living in `priv-app` does, and the permission count travels alongside as context on the `FiredRule.detail` string. It never raises the floor to `Unsafe`. Inverting this reading is called out as the easiest correctness mistake available in this module.

## Provenance

Every field this pipeline emits carries a prefix naming which authority owns it.

| Prefix | Owner |
|---|---|
| `rule:` | a deterministic rule, including the ladder |
| `graph:` | the corpus graph (`dependencies`/`neededBy`) |
| `llm:` | the model |
| `human:` | a triage reviewer |

`RemovalFloor.as_json()` carries `rule` (which `FloorRule` set the reported floor) and `fired` (every rule that fired, strictest first), so a triage card can show the reviewer why a package is pinned where it is.

## The regression test

`test_no_public_ladder_call_can_return_below_the_floor` (`tests/test_ladder.py`) enumerates every public function in `ladder.py` that can hand back a `Removal`, by walking the module's own `vars()` and filtering on a helper that detects a rating-producing signature. Each one must be registered in the test's `RATING_CALLABLES` mapping with a boot-critical invocation, and every registered call must answer `Unsafe` when handed the most permissive proposal against a `core_app` floor. A new function added to `ladder.py` that can return a `Removal` fails this test until it is registered, so the safety property is enforced by the module's surface rather than by a reviewer remembering to check.

## Filters: what reaches the additions queue

`filters.py` decides which packages in the corpus are worth proposing at all, after the ladder has run. The measured baseline this collapses: a Google emulator image yields 228 packages, 152 already in `uad_lists.json`, and roughly 5 of the remaining 76 are worth an entry, a 15:1 junk ratio even on the best-covered vendor.

`filter_verdict` checks four conditions in order and records the first one that fires:

| Verdict | Fires when | Notes |
|---|---|---|
| `already_upstream` | the package name is already a key in `uad_lists.json` | still analyzed: a mechanically derived `neededBy` correction on an existing entry is worth proposing |
| `auto_generated_rro` | `auto_generated_rro_` appears anywhere in the package name | matched as a SUBSTRING because on real firmware the marker is a SUFFIX (`com.android.ons.auto_generated_rro_vendor__`); a prefix match finds 0 of 27 on a Pixel 6 |
| `emulator_only` | every device that shipped the package matches an emulator name marker (`emulator`, `sdk_gphone`, `goldfish`, `ranchu`) | decided on device provenance, never on the package name |
| `queued` | none of the above fired | reaches the additions queue |

Overlays are deliberately not dropped as a class. 720 of the 5372 live upstream entries match an overlay or RRO pattern, so upstream already accepts them; dropping the class outright would discard most of the corpus for nothing.

Measured over the merged 393-package corpus against the live 5372-entry upstream list: 273 already upstream, 5 auto-generated RRO, 67 emulator-only, 48 queued. See [Classification](Classification) for what happens to the 48.
