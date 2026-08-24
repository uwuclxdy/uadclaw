"""What reaches the additions queue.

This is where the 15:1 junk ratio collapses. The measured baseline: a Google emulator image
yields 228 packages, 152 of them already in `uad_lists.json`, 76 missing, and of those roughly
five are worth an entry. Google is the best-covered vendor in that list, so that is the noise
FLOOR, not a bad case.

Three filters, and one deliberate non-filter:

- already in `uad_lists.json` → out of the additions queue (it is still analysed: a
  mechanically derived `neededBy` on an existing entry is a correction worth proposing);
- `auto_generated_rro_*` → dropped. Matched as a SUBSTRING, because on real firmware the
  marker is a suffix: `com.android.ons.auto_generated_rro_vendor__`, never a prefix. A prefix
  match finds 0 of the 27 on a Pixel 6 and 0 of the 20 on the emulator, and reads exactly like
  a corpus that happens to have none;
- present only on emulator devices → dropped, decided on DEVICE PROVENANCE rather than on the
  package name. A name heuristic would be guessing; "every device that shipped it was an
  emulator" is a fact the corpus already carries;
- **overlays are not dropped as a class.** 720 of the 5372 live upstream entries match an
  overlay or RRO pattern, so upstream accepts them, and there are 87 on a Pixel 6 and 102 on
  the emulator. Dropping the class would discard most of the corpus for nothing.

The verdicts are ordered, and the first one that fires is recorded: "already upstream" beats
"auto-generated RRO" so that the queue's drop reasons stay readable as a funnel.
"""

import enum
from collections.abc import Sequence

from uadclaw.corpus import CorpusPackage
from uadclaw.upstream import UpstreamList

# Measured on both corpora: the marker is a suffix on the target's name, so this is matched
# anywhere in the package name and never as a prefix.
AUTO_GENERATED_RRO_MARKER = "auto_generated_rro_"

# A device key is `<driver>:<device>`. These mark the device half of a key that is an emulator
# image rather than a phone: `emulator` covers the key this repo's corpora use
# (`google:emulator-a16`), the other three are the AOSP emulator's own device names.
EMULATOR_DEVICE_MARKERS: tuple[str, ...] = ("emulator", "sdk_gphone", "goldfish", "ranchu")


class FilterVerdict(enum.StrEnum):
    """Why a package is, or is not, in the additions queue."""

    QUEUED = "queued"
    ALREADY_UPSTREAM = "already_upstream"
    AUTO_GENERATED_RRO = "auto_generated_rro"
    EMULATOR_ONLY = "emulator_only"


def is_emulator_device(device_key: str) -> bool:
    """Whether a `<driver>:<device>` key names an emulator image rather than a phone."""
    _, _, device = device_key.partition(":")
    name = (device or device_key).lower()
    return any(marker in name for marker in EMULATOR_DEVICE_MARKERS)


def is_auto_generated_rro(package: str) -> bool:
    return AUTO_GENERATED_RRO_MARKER in package


def filter_verdict(item: CorpusPackage, *, upstream: UpstreamList) -> FilterVerdict:
    """The one reason this package is or is not queued, first rule that fires."""
    if item.package in upstream:
        return FilterVerdict.ALREADY_UPSTREAM
    if is_auto_generated_rro(item.package):
        return FilterVerdict.AUTO_GENERATED_RRO
    if item.devices and all(is_emulator_device(device) for device in item.devices):
        return FilterVerdict.EMULATOR_ONLY
    return FilterVerdict.QUEUED


def queue_verdicts(
    corpus: Sequence[CorpusPackage], *, upstream: UpstreamList
) -> dict[str, FilterVerdict]:
    return {item.package: filter_verdict(item, upstream=upstream) for item in corpus}


def survival(verdicts: dict[str, FilterVerdict]) -> dict[str, int]:
    """The funnel, for the log line and for the measured comparison against the 15:1
    baseline: how many packages each verdict accounts for."""
    counts = {str(verdict): 0 for verdict in FilterVerdict}
    for verdict in verdicts.values():
        counts[str(verdict)] += 1
    counts["total"] = len(verdicts)
    return counts
