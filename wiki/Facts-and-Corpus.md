# Facts and corpus

**One APK's manifest becomes typed facts; facts merge across devices into one row per package; the merged corpus yields exactly two kinds of dependency edge and records everything else as evidence.**

## `facts.py`: manifest to typed facts

`parse_apk()` turns one extracted APK into an `ApkFacts` dataclass. Pure: no database, no filesystem beyond the file it is handed. Uses androguard rather than aapt2: aapt2 resolves everything needed but drags the Android SDK and a JRE into the worker image, and androguard is pure Python. Measured over all 312 APKs of `oriole-cp2a.260705.006.a1`: 312 parsed, zero errors.

Two measured shapes this module encodes rather than rediscovers:

- **`coreApp` is read with a null namespace.** AOSP parses it via `parser.getAttributeBooleanValue(null, "coreApp", false)`, an un-namespaced manifest root attribute. A reader walking only `android:`-prefixed attributes misses all 23 of them on a Pixel and computes a floor of `Recommended` for packages the platform will not boot without. See [Rule-Ladder](Rule-Ladder) for what that floor gates.
- **A missing `android:label` is a missing attribute, not a resolution failure.** `_label()` returns `LABEL_UNKNOWN` (`"unknown"`) when `get_app_name()` resolves to nothing or to an unresolved `@…` reference; there is no ARSC fallback to build. On the measured corpus, `get_app_name()` resolved 177 of 225 non-overlay packages and 18 of 87 overlays, and every spot-checked miss simply never declared a label at all.

androguard 4.x logs through loguru, which stdlib `logging.disable()` does not reach. `_loguru_logger.disable("androguard")` runs before androguard is imported, since its module-level loggers bind at import time; this silenced a measured 123 KB of DEBUG output across eight APKs in one baseline run.

The launcher icon (`icons.extract_icon`) and the dex `content://` scan (`_content_uri_authorities`) both run inside `parse_apk()`, not in a later stage, because `stages.extract_facts` deletes the APK the moment its facts land. This is the only point in the pipeline the file still exists.

`_authorities_in_dex()` reads only a dex member's string table (`string_ids_size`/`string_ids_off` plus each `string_data_item`), not a full androguard `DEX` parse. Measured 2026-08-14 over the 228-APK emulator corpus: 35 ms/APK against 459 ms/APK for the full parse. MUTF-8 is scanned as plain bytes safely because the `content://` pattern and every authority are pure ASCII, and a MUTF-8 multi-byte sequence always sets the high bit on every one of its bytes, so the ASCII pattern can neither appear inside one nor straddle one.

Fields recorded per APK: package name, label, version code, partition, `priv_app` (from the partition path, `/priv-app/` present in `device_path`, never from the manifest's own `FLAG_SYSTEM`), signing cert issuer/subject, `coreApp`, `sharedUserId`, `persistent`, `hasCode`, overlay target/static/priority, `<library>`/`<static-library>` declarations, `<uses-library>`/`<uses-static-library>` consumers split required from optional, `<protected-broadcast>`, provider authorities, `<queries><package>` names, dex `content://` authorities, intent filters with priorities, IME/device-admin/accessibility/carrier-service flags, sha256, and launcher icon bytes plus mime.

## `factstore.py`: observation and merge

One `package_observations` row per APK, keyed on `(device_key, build, device_path)`. One `package_facts` row per package name. `merge_observations()` rebuilds the merged row from every observation of that package on every call, rather than folding a new observation into whatever was already there, which is what makes re-scanning a build idempotent instead of order-dependent.

Three merge rules, each field assigned to exactly one:

| Rule | Fields | Behavior |
|---|---|---|
| Sticky-true (`STICKY_TRUE_FIELDS`) | `priv_app`, `core_app`, `persistent`, `has_code`, `overlay_static`, `is_input_method`, `is_device_admin`, `is_accessibility_service`, `is_carrier_service` | one device declaring `true` makes the merged value `true` forever, since these gate the rule-ladder floor and a merge must never lower it |
| Union (`UNION_FIELDS`) | `libraries`, `static_libraries`, `uses_libraries_required`, `uses_libraries_optional`, `protected_broadcasts`, `provider_authorities`, `queries_packages`, `content_uri_authorities`, `intent_filters` | additive evidence: declared on any device, present in the merged row |
| First-wins (`FIRST_WINS_FIELDS`) | `cert_issuer`, `cert_subject`, `shared_user_id`, `overlay_target` | identity scalars, taken from the lowest-sorting `(device_key, build, device_path)`, never the most recent writer |

`label` and the two `ICON_FIELDS` (`icon_bytes`, `icon_mime`) follow a variant close to first-wins: a declared label beats `LABEL_UNKNOWN` wherever it appears, and an icon-bearing observation beats an iconless one, before the sort order decides between remaining ties. Strict first-wins would let one silent device blank a name or an icon every other device carries.

`CONFLICT_FIELDS` (`cert_issuer`, `cert_subject`, `shared_user_id`, `core_app`, `persistent`, `overlay_target`) are the fields where two devices stating DIFFERENT values sets `has_conflict` and records every value with the devices that carried it. One device stating nothing is absence of evidence, not contradiction, and never raises the flag. Measured over the 147 packages shared by a Pixel 6 and the Android 16 emulator: 134 disagree on certificate, every one of them APK signature v3 key rotation or the AOSP test key facing Google's production key. So the flag is a "show the reviewer both values" marker, not a suspicion score, and the triage screen renders it rather than filtering on it.

## `etcconfig.py`: element dispatch

Parses `/etc/permissions/*.xml`, `/etc/sysconfig/*`, `/etc/default-permissions/*.xml`, and `roles.xml` for three rule-ladder inputs stated only there, never in an APK, via `_collect()`.

| Element | Feeds | Note |
|---|---|---|
| `<privapp-permissions package="X"><permission name="Y">` | `privapp_permissions` | an integration score, never a boot-risk flag; AOSP refuses to boot when a still-present priv-app requests an unallowlisted permission, so removing the app removes the request |
| `<role name="…" static="true" defaultHolders="a;b">` | `static_role_holders` | only `static="true"` roles count; a non-static role is user-reassignable and not a floor input |
| `<library name="…">` | `platform_libraries` | why most `<uses-library required="true">` names resolve to no corpus package: the provider is the platform, not a removable APK |

Dispatch runs on the element via `root.iter(...)`, never on the filename: AOSP lets a device implementer choose any file layout as long as every `priv-app` package is allowlisted somewhere, so a fixed-path assumption would silently produce an empty allowlist for a vendor filing it elsewhere.

Two structural guards protect every parse: `MAX_CONFIG_XML_BYTES` (8 MiB) refuses an oversized file before it is read, and `_DTD_PATTERN` refuses any document declaring a DTD or entity (`<!DOCTYPE`, `<!ENTITY`). `xml.etree` expands internal entities, the whole billion-laughs class, and no legitimate AOSP config XML carries one, verified across 246 real config files.

`merge_config_inputs()` unions several devices' `ConfigInputs` the same order-independent way `factstore` merges facts: a static role held on one device and a privileged-allowlist entry on another must both reach the same package's floor.

## `corpus.py`: two edge classes, everything else is evidence

`build_graph()` is pure and order-independent over a `Sequence[CorpusPackage]`, emitting exactly two edge kinds.

- **`EDGE_OVERLAY`**: `<overlay android:targetPackage="X">` means the overlay depends on X. Fully mechanical; an RRO is meaningless without its target.
- **`EDGE_LIBRARY`**: a provider declares `<library>` or `<static-library>`; a consumer declares `<uses-library required="true">` (the AOSP default when the attribute is simply absent) or `<uses-static-library>` (unwaivable by construction, since that element carries no `required` attribute to omit). The consumer depends on the provider.

Recorded as `PackageEvidence` and never emitted as an edge: `<queries><package name="X">` (a caller is expected to handle X's absence), a required library no corpus package provides (split into `required_libraries_from_platform` against `required_libraries_unresolved`), `content://` authorities referenced in dex and matched by authority string only (`content_uri_authorities_in_corpus` / `_absent`), and shared provider authorities two packages both declare.

Measured over both local corpora: the library edge class yields zero on real firmware. Every `<uses-library required="true">` name (8 distinct on the Pixel, 4 on the emulator) resolves to a platform-declared Java shared library in `/etc/permissions/*.xml`, not to any APK in the image, and the one `<static-library>` each image declares (`com.google.android.trichromelibrary`) has zero `<uses-static-library>` consumers on either corpus. That is the corpus's own shape rather than a broken lookup, so a future firmware that ships its own library-providing APK is what would first produce a library edge.

`corpusstore.py` is where a `PackageFact` ORM row becomes a `CorpusPackage` value object (`corpus_package()`) and where a `CorpusGraph` gets written back, split into disjoint column sets so three separate stages can each own their own columns: `store_graph()` writes `dependencies`/`needed_by`/`edges`/`evidence`, `store_filter_verdicts()` writes queue membership, `store_floors()` writes the [Rule-Ladder](Rule-Ladder) floor. Each names only its own columns on upsert, so re-running one stage can never blank another's output on the same `package_analysis` row. `load_corpus()` defers the `icon_bytes` column: measured against a throwaway database holding the real 312-package Pixel corpus, that column alone accounts for 34.5% of the row payload a full-column query would transfer, for something nothing downstream of the load reads.

## Icons

`icons.py` renders one APK's launcher icon to bytes, pure, no database, no scratch, called from inside `parse_apk()` before the APK closes. Two rules shape the module: dispatch on the bytes (a member named `.png` is whatever the magic says it is; `_raster_mime()` checks PNG, JPEG, and WEBP magic, never the zip member's extension), and nothing out of the APK is copied verbatim into the emitted SVG. Every value is parsed into a float, a colour, or a member of a fixed enum, then re-serialized from that parsed value; `android:pathData` is re-emitted token by token from a fixed command set (`_PATH_COMMANDS`). An attribute that fails to parse, or an element outside the allowlisted support set, refuses the whole icon rather than rendering it partially: a half-drawn icon is a wrong claim about a package rather than a smaller one.

Measured over the 312-APK Pixel corpus at `max_dpi=320`: 223 declare no resolvable icon at all, 48 yield a PNG, 5 a WEBP, and 36 a binary-XML drawable this module renders. A separate 40-APK sample (`monogram.py`'s own measurement) found 25 declaring no `android:icon` at all, which is the source of the "62% of packages have no icon" figure in this repo's module map.

Two axes bound recursion separately. `_MAX_DRAWABLE_DEPTH` (8) bounds reference resolution, how many `@ref` hops one icon may take; `_MAX_ELEMENT_DEPTH` (32) bounds element nesting inside one decoded document (`<group>` inside `<group>`), which the reference bound never touches. `MAX_DRAWABLE_BYTES` (256 KiB) and `MAX_ICON_BYTES` (64 KiB) cap input and output size respectively.

`ApkDrawables` is the `DrawableSource` a caller hands to the pure renderer, resolving a `@ref` against androguard's parsed resource table with a bound of `_MAX_REF_HOPS = 4`. androguard hands enum attributes over as their AOSP integer spelling rather than the source XML's own words (`android:fillType="evenOdd"` arrives as `"1"`), so every lookup table here (`_FILL_RULES`, `_LINE_CAPS`, `_LINE_JOINS`) keys on both spellings.

## Monogram

`monogram.py` derives the fallback chip a package gets when it has no renderable icon. Derived, never stored: the same property matters more than any other, that a given package name produces the same chip everywhere and forever, since two screens showing one package two different ways would teach a reviewer to trust neither.

`monogram_for(package)` takes the last dotted component of the package name (the part that actually distinguishes two packages sharing `com.google.android.`), keeps its first two alphanumeric characters, and falls back to the whole name if that component carries fewer than two. The colour index comes from `hashlib.sha256(package.encode()).digest()[0] % MONOGRAM_COLOURS` (8 colours), deliberately never `hash()`, which is salted per process and would repaint every chip on every restart. `templates/partials/pkg_icon.html` is the one place either branch, real icon or monogram, is rendered.

## What could not be verified against source

- The exact byte counts and dates cited for the measurements above (the 34.5% icon-column payload figure, the 123 KB androguard log measurement, the dex-scan timing comparison) are carried from gitignored operator notes rather than something this reconciliation re-ran.
- Whether a future OEM corpus ever produces a nonzero `EDGE_LIBRARY` count is unmeasured by construction; the zero recorded here describes only the two local corpora measured to date.
