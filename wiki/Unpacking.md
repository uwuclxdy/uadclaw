# Unpacking

**Container format is decided from the bytes at every step, never from the OEM or the file's name, down to selective extraction of the paths worth keeping.**

`src/uadclaw/unpack.py` is the module. A firmware driver hands it an archive; it hands `facts.py` a tree of extracted APKs and config files.

## Magic-number dispatch

`_classify_header()` reads the first 8192 bytes (`_HEADER_BYTES`) of a source and returns one of eleven `ContainerFormat` values, matched at fixed offsets:

| Format | Magic | Offset |
|---|---|---|
| Sparse image | `3a ff 26 ed` | 0 |
| Zip | `50 4b 03 04` | 0 |
| 7-Zip | `37 7a bc af 27 1c` | 0 |
| A/B OTA payload | `CrAU` | 0 |
| LZ4 frame | `04 22 4d 18` | 0 |
| Tar | `ustar` | 257 |
| GPT disk | `EFI PART` | 512 |
| Super image (LP geometry) | `gDla` | 4096 |
| EROFS | `e2 e1 f5 e0` | 1024 |
| ext4 | `53 ef` | 1024 + 0x38 |

Order inside `_classify_header()` matters: sparse is checked before zip because a sparse-wrapped super image still needs un-sparsing first, and the tar magic is checked before the ext4 offset because `ustar` at byte 257 is a longer, more specific match than a two-byte ext4 magic a tar's first file entry could carry by coincidence at the same relative offset.

A Pixel factory zip's partition images are named `*.img` and are raw ext4; `super_empty.img` is a 5,080-byte metadata template carrying the LP geometry magic and is skipped by name (`_NON_FILESYSTEM_IMAGES`) before it can be misdetected as a real super image needing `lpunpack`.

## The chain, per format

`unpack_to_partitions()` recurses through `_unpack()`, capped at `_MAX_UNPACK_DEPTH = 6` levels so a pathological or hostile nesting becomes a named error instead of a worker unpacking until the disk fills.

| Format | Handler | External tool |
|---|---|---|
| Zip | `_unpack_zip()`: a nested zip wins over top-level `.img` members, then `payload.bin`, then sparsechunk sets, then `.img` members; anything left is sniffed as a tar | none, stdlib `zipfile` |
| 7-Zip | `_unpack_7z()`, format forced with `-t7z` | `7z` |
| Tar | `_unpack_tar()` / `_extract_tar_partitions()`, streamed via `tarfile`'s `r\|` mode so a multi-GB member is never written whole to disk first | none, stdlib `tarfile` |
| LZ4 frame | `_unpack_lz4()` / `_decoded_chunks()`, streamed through `lz4.frame.LZ4FrameDecompressor` | none, `lz4` Python package |
| A/B OTA payload | `_unpack_payload()` | `payload-dumper-go` |
| Sparse image | `_unpack_sparse()` | `simg2img` |
| Super image | `_unpack_super()` | `lpunpack` |
| GPT disk | `_unpack_gpt()`, parsed in-process by `read_gpt_partitions()` | none |
| ext4 | listed with `7z l -slt -text`, extracted selectively via `_extract_ext4()` | `7z` |
| EROFS | `fsck.erofs --extract`, filtered afterward by `_harvest_erofs_staging()` | `fsck.erofs` |

## External tools

| Tool | Provides | Arch | Debian |
|---|---|---|---|
| `7z` | ext4 listing/extraction, 7z archives | `7zip` | `7zip` (p7zip 16.02 cannot read ext4 at all) |
| `fsck.erofs` | EROFS extraction | `erofs-utils` | `erofs-utils` |
| `simg2img` | Android sparse to raw | `android-tools` | `android-sdk-libsparse-utils` |
| `lpunpack` | super image to partition images | `android-tools` | no package provides it; the worker image builds it from `nmeum/android-tools` |
| `payload-dumper-go` | A/B OTA payload to partition images | not packaged anywhere; vendored onto `PATH` in the worker image | not packaged |

`_tool_path()` resolves each on `PATH` (or at a configured path, `PAYLOAD_DUMPER_PATH` for `payload-dumper-go`) and raises `MissingToolError` naming what provides it. 7-Zip has no EROFS handler at all: pointed at an EROFS image it falls through to its gzip reader and lists one file where `fsck.erofs --extract` recovers 105, raising nothing, which is measured and is why the container format is forced on every 7z call (`-text`, `-t7z`) instead of left to 7z's own sniffing.

## Samsung's chain

The deepest chain here: zip, tar, LZ4 frame, sparse, super, EROFS. Member names carry no format information at all (tars are named `.tar.md5`, images inside them `.img.lz4`), so every step reads magic bytes and the suffixes are never consulted. The tar is walked as a stream straight out of the zip member (`_stream_zip_tar()`), because writing an 11.47 GB `AP_` member to disk first would put a copy nothing reads twice on scratch that has to hold the rest of the job too.

## Motorola's sparsechunk sets

Motorola splits its super image across `super.img_sparsechunk.0` through `.13` inside its firmware zip; none of those names end in `.img`, so `_sparsechunk_sets()` groups and orders the set by parsing the numeric suffix (`_SPARSECHUNK_RE`), the only record of chunk order that exists anywhere. Every member is still refused unless `detect_format()` says its bytes are a sparse image: the name decides order, the magic decides identity, and the two disagreeing is refused rather than trusted. A gap in the run (`.0`, `.2` with `.1` missing) raises instead of rebuilding a short image.

## `_NO_APP_PARTITIONS` and `PARTITIONS_ALLOWED_EMPTY`

Two lists answer two different questions, and this pipeline had a bug once from conflating them.

`_NO_APP_PARTITIONS` names real filesystems known to carry no preinstalled app, so unpacking one is skipped entirely rather than paying a full decompression for zero yield: `userdata`, `cache`, `metadata`, `persist`, `omr`, `misc`, `vm-bootsys`, `dspso`. `userdata` is the largest single cost avoided: Samsung's `AP_`/`USERDATA_` pair ships a fresh `/data` image at 1.85 GB of LZ4.

`PARTITIONS_ALLOWED_EMPTY` names partitions that ARE unpacked and extracted, and are allowed to yield zero artifacts once extracted: `system_other`, `cache`, `metadata`, `userdata`, `prism`, `optics`, `odm`, `dsp`, `my_bigball`, `my_carrier`, `my_engineering`, `my_heytap`, `my_manifest`, plus every partition ending `_dlkm` (checked by `partition_may_be_empty()`).

`extract_artifacts()` separates the two meanings of "empty" and treats them differently:

- **An image yielding no file at all is always an error, for every partition, `PARTITIONS_ALLOWED_EMPTY` included.** Samsung's `odm` stub still holds 10 files (`build.prop`, `passwd`, four selinux files); no vendor ships an image with literally nothing in it.
- **An image that yields files matching no artifact pattern is waivable, only for the named partitions.** `odm` carries 4 APKs on Nothing's FroggerPro but is a 10-file stub with none on Samsung's SM-S911U, which is why the waiver is per-partition rather than per-vendor.

The Oppo `dsp` and `my_*` entries (measured on PLK110, 2026-08-13) hold ColorOS metadata and DSP firmware, 1 to 17 files each and no artifact among them. `my_carrier` takes the same trade `prism` and `optics` already took: a carrier build could put an app there and this pipeline would not see it.

## A/B slots

`lpunpack` writes every partition the super metadata declares, both slots. On a retrofitted A/B device the B slot has extents but no content until the first OTA lands: measured on Motorola `rtwo`, 5 of 6 `_b` images `lpunpack` produced are zero bytes, matching no magic. `system_b` on the same device is a real 1.6 MB EROFS image. `unpopulated_slot_b()` drops an unreadable `_b` image only when its `_a` sibling is present in the same set; widening that to "skip anything unreadable" would let a genuinely lost partition disappear with nothing raised. `_resolve_partition_identities()` then normalizes A/B slot names, keeps slot A when both slots resolved (a factory image populates slot A), and raises `DuplicatePartitionError` on any other identity collision, since nothing here can guess which of two same-named images a device would mount.

## Path safety

Four functions handle every name this pipeline did not write itself: GPT partition names are raw UTF-16 out of a disk image, zip member names come from a downloaded archive's central directory, and filesystem paths come out of an extracted image.

- **`safe_component(name, context)`** reduces one untrusted name to a single harmless path component. Everything outside `[A-Za-z0-9._-]` becomes `_`; a name that reduces to nothing usable (empty, `.`, `..`) is refused rather than silently replaced with a default.
- **`unique_path(directory, filename)`** returns a path in `directory` no earlier caller took, appending `-2`, `-3`, and so on. Without it, two zip members sharing a basename (`a/system.img`, `b/system.img`) would let the second write clobber the first with no error at all.
- **`ensure_within(root, candidate, context)`** asserts `candidate` resolves inside `root` immediately before a write happens. Checked against `root`, never against `dest.parent`: the parent of an escaping path is itself outside `root`, so containment checked against the parent passes for exactly the input it should catch.
- **`safe_archive_path(path, context)`** validates a path read out of an archive or filesystem listing before it is joined to a destination. An absolute path or a `..` segment is refused outright, never sanitized, because normal firmware (5,000+ entries measured across five partitions) has neither.

## Artifact extraction

`extract_artifacts()` pulls the paths worth keeping via `ARTIFACT_PATTERNS`, matched with `fnmatch` against the partition-relative path: `*.apk`, `*/etc/permissions/*.xml`, `*/etc/sysconfig/*`, `*/etc/default-permissions/*.xml`, `*/privapp-permissions*.xml`, `*/default-permissions*.xml`, `*/roles.xml`. APEX payloads (`*.apex`) are deliberately excluded: unpacking them costs every device the same walk for packages triage never shows.

Extraction is selective by construction rather than a bulk unpack: an ext4 partition is listed once with `7z l -slt -text`, the wanted paths are chosen in Python, and only those are handed to `7z x`. That order sidesteps two measured traps: `7z x -r 'sysconfig/*'` extracts zero files against an image the listing proves holds dozens, and `debugfs -R "ls -l /app"` walks nothing on an image whose partition carries a nested `system/` root.

`canonical_device_path()` normalizes that nested-root difference: `system.img` entries carry a `system/` prefix inside the image; `product.img`, `system_ext.img`, and `vendor.img` are rooted directly at `app/`, `etc/`, `priv-app/`. Both spellings survive: `image_path` exactly as the image recorded it, which is what a ground-truth manifest is keyed on, and `device_path` normalized to the file's absolute path on a running device, which is what [Facts-and-Corpus](Facts-and-Corpus) keys its output on.

## What could not be verified against source

- Exact byte counts, device names, and measurement dates cited above (the Motorola `rtwo` chunk counts, the Samsung `AP_` tar size, the PLK110 partition file counts) are carried from `docs/domain-knowledge.md`, a gitignored operator notebook this reconciliation could read but not independently re-run.
