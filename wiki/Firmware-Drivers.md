# Firmware drivers

**Six OEM drivers behind one interface: what builds exist, how to fetch one, what the operator agrees to by enabling it.**

## The interface and the registry

`src/uadclaw/firmware.py` defines `FirmwareDriver`, an abstract class every driver implements and nothing more:

- `terms() -> TermsPosture`: what enabling this driver commits the operator to, evaluated against current configuration so `acknowledged` reflects reality rather than intent.
- `list_available() -> list[FirmwareRef]`: every build the source currently offers, oldest first where the source is itself ordered. Raises `EmptyFirmwareIndexError` instead of returning an empty list, because a source answering 200 with nothing usable reads exactly like an empty catalogue.
- `fetch(ref, dest_dir) -> DownloadedArchive`: download `ref` into `dest_dir` and return the digest of what actually arrived.

Nothing downstream is keyed on the OEM. [Unpacking](Unpacking) dispatches on container format, so a vendor changing archive layout needs no driver change.

`_driver_factories()` maps six names to six classes: `motorola`, `nothing`, `oppo`, `pixel`, `samsung`, `xiaomi`. `get_driver(name, settings)` resolves one by name, raising `UnknownFirmwareDriverError` for a typo and `FirmwareDriverDisabledError` for a driver the operator turned off through `DISABLED_FIRMWARE_DRIVERS`, so a typo and a deliberate shutoff never read the same in a failed job's reason. `enabled_driver_names(settings)` filters the registry against that setting. Full settings reference: [Configuration](Configuration).

`FirmwareRef` is the value one driver hands back for one downloadable build: `driver`, `device`, `build`, `url`, plus optional `sha256`, `md5`, `size`, `android_version`, `marketing_name`, and `archive_suffix`. `device` matches `^[A-Za-z0-9_-]{1,64}$` and `build` matches `^[A-Za-z0-9._-]{1,64}$`, because both reach a filename: `archive_filename` builds it from these validated fields, never from the URL's own basename. `TermsPosture.risk` is a `TermsRisk` enum the dashboard renders: `public`, `acknowledgement`, `restricted`, `reverse_engineered`, loosely ordered by how likely the source is to object.

## Newest-build resolution

`select_ref(refs, device=..., build=None)` picks one build out of an index listing. Handed no explicit `build`, it resolves "newest" from the date embedded in the build id, via `build_date()`: a regex matching the `YYMMDD` field an AOSP build id carries in its middle segment (`CP2A.260705.006` parses to 2026-07-05). A build id with no such field falls back to its row position in the index, ranked behind every dated build.

Only Pixel's build ids carry that date field. Measured against the real Pixel index, taking the last row per device agrees with the date for 56 of 58 devices and is wrong for two (`crosshatch`, `blueline`, both long EOL), which is why date parsing exists at all rather than trusting row order outright.

Every other driver sorts its own index oldest-first before returning it, so `select_ref`'s row-order fallback lands on the newest build without needing a date it cannot parse:

- Xiaomi sorts by the index's own `date` field.
- Nothing reverses GitHub's newest-first release order.
- Motorola sorts by the mirror's own file mtime.
- Samsung has no "newest" to resolve for one model at all: two CSCs are two regional firmware lines, not two builds of one, so the operator's own `SAMSUNG_REGIONS` order decides which CSC `select_ref` lands on.
- Oppo sorts by the catalogue's own `build_timestamp`.

## The six drivers

| Driver | Index source | Ref identity | Terms risk |
|---|---|---|---|
| `pixel` | `developers.google.com/android/images` | codename + build id (`CP2A.260705.006`) | acknowledgement |
| `xiaomi` | `XiaomiFirmwareUpdater/miui-updates-tracker`'s `data/latest.yml` | codename + version | public |
| `nothing` | `spike0en/nothing_archive` GitHub releases | codename + build from the release tag | restricted |
| `motorola` | `mirrors.lolinet.com` h5ai JSON API | device + `<build_id>_<channel>` | restricted |
| `samsung` | `fota-cloud-dn.ospserver.net/firmware/{CSC}/{MODEL}/version.xml` | model + `<PDA>_<CSC>` | reverse_engineered |
| `oppo` | `roms.danielspringer.at/api/ota.php?latest=1` plus the OPlus component-OTA endpoint | model + `<ota_version>_<REGION>` | reverse_engineered |

Every driver honours `DISABLED_FIRMWARE_DRIVERS`, `FIRMWARE_HTTP_TIMEOUT_SECONDS` (60s default, read/connect only, no total deadline since a slow-but-progressing multi-GB transfer is not a failure), and `MAX_FIRMWARE_ARCHIVE_BYTES` (32 GiB default, sized against Samsung's 19.25 GB encrypted `SM-S928B` build plus headroom).

## Pixel

Reads `developers.google.com/android/images` (`PIXEL_INDEX_URL`). The index sits behind a client-side terms wall: a bare GET answers HTTP 200 with about 75 KB of prose and zero download links, the same shape a healthy-but-empty index would have. `Cookie: devsite_wall_acks=nexus-image-tos` (`PIXEL_TERMS_ACK_COOKIE_VALUE`) returns the real page, roughly 1.2 MB carrying 2293 factory zips as measured 2026-08-11. The wall sits on the index only: a bare fetch of a factory zip on `dl.google.com` answers 200 with no redirect, so `fetch()` sends no cookie there.

A ref is identified by device codename plus the build id parsed off the same URL, uppercased. `parse_factory_index()` parses each `<tr>` row independently rather than with one page-wide regex, so a row that breaks shape can never pair its neighbour's checksum with its own URL. Zero parsed links raises `EmptyFirmwareIndexError`, naming the terms-wall fix directly, since that is what the wall looks like unacknowledged, not an empty catalogue.

Measured gotchas: one link of 2293 carries an uppercase build id in its URL (`razorg-JLS36C-factory-834eab41.zip`), so the parser accepts both cases and canonicalizes to uppercase. `select_ref`'s newest-build logic exists because taking the last row disagrees with the true date for those two long-EOL devices.

## Xiaomi

Reads `XiaomiFirmwareUpdater/miui-updates-tracker`'s `data/latest.yml` (`XIAOMI_INDEX_URL`), the only machine-readable index of Xiaomi's own CDN that exists; Xiaomi publishes none. A ref is identified by `codename` plus `version`.

- **Only `Recovery` rows are offered.** `OPENABLE_METHOD = "Recovery"`. The 1,522 `Fastboot` rows are `.tgz`, matching no magic in [Unpacking](Unpacking)'s dispatch table, so listing them would hand the acquire stage builds that always fail one stage later.
- **The CDN host in the index lies about reachability.** `cdn_url()` rewrites `bigota.d.miui.com` and `ultimateota.d.miui.com` (`BROKEN_CDN_HOSTS`) to `cdnorg.d.miui.com` (`WORKING_CDN_HOST`). `bigota` answers HEAD 200 with a correct Content-Length; a plain GET intermittently returns a CloudFront 503 instead, so a HEAD-based reachability check reports a healthy source that cannot be downloaded. The rewrite runs both at index parse time and again in `fetch()`, because a ref can arrive from a job's params carrying whatever URL an operator pasted out of the tracker.
- **The index publishes md5, not sha256**, for 3,454 of 3,550 entries. `download_to_file` is handed the algorithm the source actually used rather than the pipeline's own preference.
- **Row order is not chronological.** `parse_latest_index()` sorts by the index's own `date` field before returning it: 297 of 1,270 devices with a Recovery build list a non-newest build last in the raw file, and no Xiaomi version string (`V14.0.44.0.TGOMIXM`) carries a parseable date to fall back on.
- 164 of the 2,028 Recovery rows publish a plain `http://` link, all pre-2016 devices; `FirmwareRef.url`'s `^https://` pattern refuses them, and they are logged and dropped rather than downloaded in the clear.

## Nothing

Reads `spike0en/nothing_archive` GitHub releases, unauthenticated on purpose (`NOTHING_RELEASES_URL`), capped at GitHub's 60 requests/hour/IP. A ref is identified by the release tag's `<Codename>_<Build>` split (`FroggerPro_B4.1-260723-1820`).

The APK-bearing partitions ship as a split 7z volume set, `<PREFIX>-image-logical.7z.001` through `.006`. `fetch()` streams every volume into one file (`_join_volumes()`) and deletes each volume as it is consumed, so the driver hands `unpack.py` a single archive with a single digest and nothing is left on disk for retention to miss.

- **The published hashes never cover the download.** `<TAG>-hash.sha256` lists the inner image files, not the archives, so it can only be checked after extraction; every Nothing archive comes back `integrity_verified=False`.
- **The asset prefix is not assumed to equal the release tag.** 10 of 229 releases name their assets after something else (`Spacewar-T1.5-230619-0042_1.5.5-image-logical.7z.001` under tag `Spacewar_T1.5-230619-0042`); requiring the two to match dropped all ten silently.
- **7 of 229 releases publish `-image.7z` alongside the volume set**, not instead of it, so `_asset_volumes()` makes the logical set win outright rather than pairing two archives that each claim to be volume 1.
- A gap in the volume run is refused rather than joined short: `simg2img`-style silent truncation is exactly the failure this guards.

## Motorola

Reads `mirrors.lolinet.com`'s h5ai JSON API (`POST /_h5ai/public/index.php`, `MOTOROLA_MIRROR_URL`). Motorola publishes no index document, only a browsable tree, so `MOTOROLA_DEVICES` names the codenames to crawl rather than the driver enumerating the whole mirror. A ref is identified by device plus `<build_id>_<channel>` (`V1TRS35H.60-33-7_RETAIL`).

- **One device can publish the same build id under several channels.** Measured across 164 files on three devices, 16 of 32 distinct (device, build) pairs appear in more than one channel; `V1TRS35H.60-33-5` is published by 9 of `rtwo`'s 10 channels. The channel therefore rides in `build`, never in `device`, or one phone would count as ten toward `package_facts.device_count`, the triage ranking signal.
- **The h5ai API answers with the whole ancestor chain, not just the children.** One channel listing came back with 68 items of which 5 were that channel's own files; `_direct_children()` filters a raw listing down to `base`'s direct entries before anything else touches it.
- Two filename shapes share the tree, and the leading field count differs between them (a model SKU present on some, absent on others), so `_FILENAME_RE` anchors on the trailing `_subsidy-` or `_<8 hex>_release-keys` marker rather than a fixed field index, which parses 164 of 164 measured files against 116 for a field-index approach.
- `_MAX_LISTINGS_PER_DEVICE` (96) bounds the crawl per device against a mirror layout that grows deeper or wider than the measured `official/<channel>/` shape.

## Samsung

Two unrelated halves. The version index, `fota-cloud-dn.ospserver.net/firmware/{CSC}/{MODEL}/version.xml` (`SAMSUNG_INDEX_URL`), is public and answers with no auth and no cookie. A 403 means that CSC/model pair does not exist, an S3-style `AccessDenied` body rather than a terms wall, which is what makes the index usable as an existence oracle: `SamsungDriver` probes the configured `SAMSUNG_MODELS` times `SAMSUNG_REGIONS` grid, capped at `MAX_INDEX_PROBES = 128` requests, and keeps the 200s.

The binary comes from FUS (`neofussvr.sslcs.cdngc.net`), reverse-engineered and entirely private. A ref is identified by model plus `<PDA>_<CSC>` (`S911USQS8FZG1_XAA`), because Samsung publishes exactly one build per model/CSC pair; there is no "newest" inside one pair to resolve at all.

- **Every published Python FUS client is dead against the live server.** `samloader` and its forks fail deceptively: the nonce endpoint still answers 200 with a `NONCE:` header, so a reachability probe passes, but the published static keys no longer decrypt that nonce. This driver transcribes the current protocol from `topjohnwu/samloader-rs`: `Smart`-infixed endpoints, `User-Agent: SMART 2.0`, and the server nonce used raw, never decrypted, as the `LOGIC_CHECK` input.
- **`BINARY_CRC` is the CRC32 of the encrypted `.enc4` body**, not the plaintext. An earlier project note recorded Samsung as publishing no digest at all; that was wrong. Measured 949352961 over the full 11,565,187,312-byte `SM-S911U`/`XAA` download, exactly matching what the inform response declared. The decrypted archive's own CRC32 (3102581189) matches nothing.
- The archive decrypts AES-128-ECB in the driver, in place, before it ever reaches [Unpacking](Unpacking): an encrypted blob has no magic number, so there is nothing for a byte-dispatch table to read.
- No IMEI, TAC, or device identifier of any kind appears anywhere in the working request. An earlier plan for this driver recorded a synthesized IMEI as the blocker; that belonged to a protocol Samsung has since retired.
- `_MODEL_TYPE_RE` (`^[0-9]{1,4}$`) is derived from exactly two devices that both answer `9`; the range past that pair is unmeasured.

## Oppo

Two sources, each answering a different question. `roms.danielspringer.at/api/ota.php?latest=1` (`OPPO_CATALOGUE_URL`) is the catalogue: one row per model and region carrying `ota_version`, `md5`, `size`, `build_timestamp`, and a durable `/downloadCheck` gate URL. The OPlus component-OTA endpoint (`component-otapc-{sg,cn,in,eu}.allawn{os,tech}.com/update/v3`) is the resolver: given a synthesized query it hands back one package and a fresh URL, valid roughly ten minutes. A ref is identified by model plus `<ota_version>_<REGION>`, because 12 of 95 distinct `(model, ota_version)` pairs in the catalogue carry two different archives at different sizes, one per region.

- **`realOtaVersion`, never `otaVersion`, is the package's real identity.** `RMX3706` asked at its real `A.48` answered `otaVersion: RMX3706_11.A.22` and `realOtaVersion: RMX3706_11.C.22` for one package; the rewritten `otaVersion` keeps the branch letter that was asked about, which is precisely the field a "did I get the build I named" comparison would reach for first. `resolved_package()` reads `realOtaVersion` first, falls back to `otaVersion` when it is absent, and cross-checks `componentVersion` against whichever was chosen.
- **The catalogue is multi-OEM, and only some of it belongs to this driver.** Of 179 rows measured 2026-08-12, only 107 are OPlus gates on the four `allawn*` hosts; 70 belong to Xiaomi and 2 to OnePlus-NA. `parse_catalogue()` filters by gate host (`REGION_BY_GATE_HOST`), never by a device-name heuristic, and checks each of the four hosts for emptiness separately.
- **The endpoint answers the next build in a chain, never the newest.** Asked at its own most recent answer, `PLK110` at `A.68` returns `2004`, a fixed point four builds short of the catalogue's `A.72`. So the catalogue's own `build_timestamp` decides what "newest" means; the endpoint is never asked to.
- **The endpoint is optional.** `_resolve_direct()` catches every transport, protocol, or build-mismatch failure and falls back to the catalogue's stored gate URL rather than failing the whole `acquire` stage; five modern export models (`CPH2797`, `CPH2659`, `CPH2653`, `RMX5011`, `RMX5210`) answer `2004` to every query tried.
- `OPPO_MODELS` entries (`MODEL:REGION`) reach models the catalogue never carried at all, walking the five branch letters observed anywhere (`A`, `F`, `C`, `H`, `J`) until one answers 200.

## What could not be verified against source

- The precise host-rewrite reasoning for Xiaomi (why `cdnorg` in particular serves reliably) rests on a probe result recorded in `docs/domain-knowledge.md` rather than anything assertable from the driver code alone; the code enforces the rewrite but not the underlying CDN behaviour.
- Xiaomi's, Nothing's, Motorola's, Samsung's, and Oppo's exact request/response byte counts and probe dates are taken from `docs/domain-knowledge.md` and `docs/pipeline-design.md`, which are gitignored operator notes rather than something this reconciliation could re-run.
