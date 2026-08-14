# uadclaw

**Automatic LLM-driven bloat classification of Android apps, from official OEM firmware to a reviewed UAD-ng pull request.**

This wiki is the full reference. The [README](https://github.com/uwuclxdy/uadclaw#readme) is the map: what the project is, how to install it, and the first run. Every page here is generated from the code in `src/uadclaw/`, so a page that disagrees with the source is a bug worth filing.

## Start here

| you want to | read |
|---|---|
| run the stack on a box | [Deployment](Deployment), then [Configuration](Configuration) |
| understand what decides a removal rating | [Rule ladder](Rule-Ladder), then [Classification](Classification) |
| add an OEM | [Firmware drivers](Firmware-Drivers), then [Unpacking](Unpacking) |
| change the pipeline | [Jobs and worker](Jobs-and-Worker), then [Development](Development) |
| audit what reaches upstream | [Triage and emission](Triage-and-Emission) |

## The ten stages

A firmware job walks `acquire` through `rule_ladder` and stops. Classification is its own job kind a human queues, because it spends money. Branch emission is a third.

| stage | job kind | owned by |
|---|---|---|
| acquire | firmware | [Firmware drivers](Firmware-Drivers) |
| unpack | firmware | [Unpacking](Unpacking) |
| extract_facts | firmware | [Facts and corpus](Facts-and-Corpus) |
| corpus_graph | firmware | [Facts and corpus](Facts-and-Corpus) |
| filter | firmware | [Rule ladder](Rule-Ladder) |
| rule_ladder | firmware | [Rule ladder](Rule-Ladder) |
| llm | classification | [Classification](Classification) |
| corroborate | classification | [Corroboration](Corroboration) |
| triage | none, it is the human gate | [Triage and emission](Triage-and-Emission) |
| branch | branch emission | [Triage and emission](Triage-and-Emission) |

## The invariants

Six rules the code enforces structurally. Breaking one is a safety regression rather than a tuning choice.

- **The rule ladder sets a floor the model may raise and never lower.** Lowering is unrepresentable, not merely forbidden. See [Rule ladder](Rule-Ladder).
- **A below-floor model answer is rejected, never clamped.** Correcting only the number keeps the misreading that produced it. See [Classification](Classification).
- **`dependencies` and `neededBy` are never model output.** They come from the corpus graph or from a human. See [Facts and corpus](Facts-and-Corpus).
- **A judge citing a source it was never handed refuses the whole response.** Dropping the bad URL and keeping the verdict is the tempting repair and it is the defect. See [Corroboration](Corroboration).
- **Unpacking dispatches on the bytes,** never on the OEM and never on a file's name. See [Unpacking](Unpacking).
- **Nothing pushes itself.** Emission commits a local branch into a clone the operator supplies and never pushes, fetches or authenticates. See [Triage and emission](Triage-and-Emission).

## Editing this wiki

The pages live in `wiki/` in the main repo and `.github/workflows/wiki.yml` mirrors them here on every push to `mommy`. An edit made in the wiki web UI is overwritten by the next sync, so send changes to `wiki/` instead.
