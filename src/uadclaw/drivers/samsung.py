"""Samsung firmware: a public version index, and a binary behind the reverse-engineered FUS.

The two halves are unrelated and only the second is private, which is what makes this driver
smaller than it looks.

**The index is public and answers an existence question.**
`fota-cloud-dn.ospserver.net/firmware/{CSC}/{MODEL}/version.xml` answers 200 with no auth and
no cookie. A **403 there means that CSC/model pair does not exist**, not a terms wall: the body
is an S3-style `<Error><Code>AccessDenied</Code>`, and `SM-A546B` answers 403 under DBT, XEO,
BTU and ATT while answering 200 under EUX. That makes the endpoint an existence oracle, which
is how a pipeline owning no device enumerates: probe a configured model list against a
configured CSC list and keep the 200s. Only the CSC-qualified URL shape works; the flat
`/{MODEL}/version.xml` answers 403 for everything.

**The FUS protocol changed and every Python client on the internet is dead against it.**
`samloader` and its forks fail deceptively — the nonce endpoint still answers 200 with a
`NONCE:` header, so a reachability probe passes — because the published static keys no longer
decrypt that nonce. This driver transcribes the current protocol from `topjohnwu/samloader-rs`
(Apache-2.0), verified live 2026-08-11 against `SM-S911U/XAA` and `SM-S928B/EUX`, both `S00`:

- the endpoints carry a `Smart` infix and the User-Agent is `SMART 2.0`;
- **the server nonce is never decrypted**. Its raw base64 string is the nonce for
  `LOGIC_CHECK`, and the signature is one AES-128-ECB block over `nonce[:16]` right-padded
  with ASCII `0`. Every published Python client tries to decrypt it first, which is why they
  all now produce an empty string and read as a network problem;
- the request carries **no IMEI, no TAC and no device identifier of any kind**, and needs no
  credential. An earlier plan for this task recorded a TAC table as the blocker; that belongs
  to a protocol Samsung has retired;
- `LATEST_FW_VERSION` is ABSENT from the inform response and `BINARY_SW_VERSION` is present.
  Old clients read the former, so they derive the decryption key from nothing.

**The archive is decrypted here, before it is handed on.** `uadclaw.unpack` dispatches on
magic bytes and a `.enc4` has none, so handing it over encrypted would break that rule at the
first step. The decrypt is AES-128-ECB over the whole file, streaming and IN PLACE — 11.5 GB
does not get a second copy — and ECB is what makes that safe: every block stands alone, so a
16-byte-aligned resume needs no state carried across it. The plaintext is proven rather than
assumed: the PKCS#7 padding is validated strictly and the result must open with the zip magic,
because a wrong key yields a file of exactly the right size and complete garbage, which no
exception would otherwise report.

**FUS does publish an integrity value, and it covers the encrypted body.** `BINARY_CRC` is the
CRC32 of the `.enc4` — measured 2026-08-11 over the full 11,565,187,312-byte `SM-S911U`/`XAA`
download, whose CRC32 is 949352961, exactly what the inform response declared; the plaintext's
is 3102581189 and matches nothing. So it is checked, in the decrypt pass that already reads
every ciphertext byte, and a Samsung archive comes back `integrity_verified=True` rather than
trusting bytes nobody vouched for. It is a CRC rather than a cryptographic digest — it catches
a corrupt transfer, not a hostile one — which is the same standing Xiaomi's md5 has here.

**What lands on disk is a Samsung factory zip, which `uadclaw.unpack` cannot yet open.** Its
six members are `.tar.md5` archives (a tar with an md5 line appended) holding LZ4-framed
images — measured on `SM-S911U/XAA`: the AP member is 11,465,093,243 bytes of tar whose first
entry is `boot.img.lz4`, magic `04 22 4d 18`. Neither tar nor LZ4 is in the dispatch table, so
`acquire` succeeds and `unpack` refuses. That is a separate task against `uadclaw.unpack`, not
something a driver may work around: dispatch is on the bytes, and inventing a Samsung-shaped
path through the unpacker is exactly what the module forbids.
"""

import asyncio
import hashlib
import logging
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree

import httpx
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from uadclaw.firmware import (
    DownloadedArchive,
    EmptyFirmwareIndexError,
    FirmwareDownloadError,
    FirmwareDriver,
    FirmwareError,
    FirmwareInputError,
    FirmwareRef,
    TermsPosture,
    TermsRisk,
    download_to_file,
)
from uadclaw.settings import Settings

logger = logging.getLogger(__name__)

# The FUS endpoints. Not settings: an operator cannot repair a protocol change by repointing a
# host, and these two only mean anything alongside the request shapes below. The public index
# IS a setting (`SAMSUNG_INDEX_URL`), because it is a plain document fetch like every other
# driver's source.
FUS_URL = "https://neofussvr.sslcs.cdngc.net"
# samloader-rs downloads over plain http. https serves the same path — verified 2026-08-11 by
# a ranged GET answering 206 with a correct Content-Range — so this pipeline never pulls
# firmware in the clear, the same call the Xiaomi driver makes about its own index rows.
DOWNLOAD_URL = "https://cloud-neofussvr.samsungmobile.com/NF_SmartDownloadBinaryForMass.do"
USER_AGENT = "SMART 2.0"

# Credited in samloader-rs `fus/src/auth.rs` to @henr1kas and @ducthoe. Not a secret and not
# this project's: it is the published constant that makes the handshake verifiable.
AUTH_AES_KEY = bytes.fromhex("422e73733617ae2b198940fd4e32b0a5")
AES_BLOCK_BYTES = 16

ENCRYPTED_SUFFIX = ".enc4"
_ZIP_MAGIC = b"PK\x03\x04"
_DECRYPT_CHUNK_BYTES = 4 * 1024 * 1024

# A Samsung model (`SM-S928B`) and a CSC (`XAA`, `EUX`). Both reach a URL, so both are pinned
# to a charset that cannot spell a path segment or a query separator.
_MODEL_RE = re.compile(r"^[A-Z0-9-]{3,32}$")
_REGION_RE = re.compile(r"^[A-Z0-9]{3}$")
# The two fields of the inform response that reach the download URL. They arrive inside a
# document downloaded from a third party, so they are refused rather than sanitised: a real
# one is `/neofus/911/` and `SM-S911U_2_20260708221800_qd55e39o59_fac.zip.enc4`.
# The floor is 25, not 1: the download-authorisation LOGIC_CHECK is computed over
# `filename[-25:-9]`, and a shorter name slices to something shorter than 16 characters
# without raising, so the request goes out wrong and FUS refuses it for a reason no message
# names. A real name is 48 characters.
_BINARY_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{25,160}$")
# The four-part version and DEVICE_MODEL_TYPE both come out of a downloaded document and both
# are interpolated into an outgoing XML body. Every other field that does is refused rather
# than escaped; these two were the exception until a reviewer said so.
_VERSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}(?:/[A-Za-z0-9._-]{0,64}){1,3}$")
_MODEL_TYPE_RE = re.compile(r"^[0-9]{1,4}$")
# An unsigned 32-bit CRC in decimal, which is what `BINARY_CRC` carries (949352961).
_CRC32_RE = re.compile(r"^(?:[0-9]{1,10})$")
# The lookahead is what refuses a `.` or `..` COMPONENT: the charset alone admits both, and a
# `/neofus/../911/` reaching a server that normalises its own paths asks for a resource this
# pipeline never named. Same call `unpack.safe_archive_path` makes about an image listing.
_MODEL_PATH_RE = re.compile(r"^(?:/(?!\.{1,2}/)[A-Za-z0-9._-]{1,64})+/$")
# The PDA half of a build id (`S911USQS8FZG1`) and the whole `<PDA>_<CSC>` this driver spells.
_BUILD_RE = re.compile(r"^(?P<pda>[A-Za-z0-9]{1,40})_(?P<region>[A-Z0-9]{3})$")

# One request per (model, CSC) pair, so the grid is the request count. A ceiling rather than an
# unbounded crawl of somebody else's index: 8 models against 8 CSCs is already 64 requests.
MAX_INDEX_PROBES = 128

# Every XML document here comes off a third-party server. Same threat and the same mitigation
# `etcconfig` applies to a vendor image's config files: a byte ceiling, and a hard refusal of
# any document carrying a DTD, which is the whole billion-laughs class removed without a
# parser swap. The real documents are 1.7-8 KB.
MAX_RESPONSE_BYTES = 1024 * 1024
_RESPONSE_CHUNK_BYTES = 64 * 1024
_DTD_SNIFF_BYTES = 8 * 1024
_DTD_PATTERN = re.compile(r"<!DOCTYPE|<!ENTITY", re.IGNORECASE)


class SamsungProtocolError(FirmwareError):
    """FUS answered, and what it answered is not what the protocol says it should be.

    Its own type because the FUS half is reverse-engineered and will break without warning,
    and a job's failure reason has to separate "Samsung changed the protocol" from "the
    network is down" — the first needs this module rewritten, the second needs a retry.
    """


async def _read_capped(response: httpx.Response, *, context: str) -> str:
    """The body, refused once it passes the ceiling rather than after it is all in memory.

    `response.text` on a streamed response would buffer whatever the server chose to send,
    and these are third-party servers reached by a worker with no memory bound of its own.
    Reading it in chunks with a running total is what makes the ceiling in `parse_xml` a
    ceiling on what ARRIVES — the same order `etcconfig` does it in, where the file's size is
    checked before the read rather than after.
    """
    received = 0
    chunks: list[bytes] = []
    async for chunk in response.aiter_bytes(_RESPONSE_CHUNK_BYTES):
        received += len(chunk)
        if received > MAX_RESPONSE_BYTES:
            raise SamsungProtocolError(
                f"{context}: the response passed the {MAX_RESPONSE_BYTES}-byte ceiling for a "
                f"FUS document after {received} bytes (a real one is 1.7-8 KB); refusing to "
                "read the rest of it"
            )
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


def parse_xml(text: str, *, context: str) -> ElementTree.Element:
    """Parse a server response, refusing anything carrying a DTD.

    `xml.etree` expands internal entities, which is the billion-laughs class. No legitimate
    FUS or FOTA document has a `<!DOCTYPE>` (measured across every response this driver was
    built against), so refusing one costs nothing.
    """
    if len(text) > MAX_RESPONSE_BYTES:
        raise SamsungProtocolError(
            f"{context}: the response is {len(text)} characters, past the "
            f"{MAX_RESPONSE_BYTES}-character ceiling for a FUS document (a real one is 1.7-8 "
            "KB); refusing to parse it"
        )
    if _DTD_PATTERN.search(text[:_DTD_SNIFF_BYTES]):
        raise SamsungProtocolError(
            f"{context}: the response carries a DTD or an entity declaration. No Samsung "
            "document does; refusing to hand it to an entity-expanding parser."
        )
    try:
        return ElementTree.fromstring(text)  # noqa: S314 - DTD refused above, per etcconfig
    except ElementTree.ParseError as exc:
        raise SamsungProtocolError(f"{context}: the response is not parseable XML ({exc})") from exc


def auth_signature(nonce: str) -> str:
    """The `signature` half of the `Authorization` header.

    NOTE the shape: the server nonce is NOT decrypted. Its raw base64 string is truncated or
    right-padded with ASCII `0` to exactly 16 bytes, encrypted as ONE AES-128-ECB block and
    hex-encoded. No IV, no CBC, no PKCS#7. This is the single biggest departure from every
    published Python client, all of which try to decrypt this value first and get an empty
    string back — which reads as a network problem rather than as a protocol change.
    """
    block = (nonce[:AES_BLOCK_BYTES].encode() + b"0" * AES_BLOCK_BYTES)[:AES_BLOCK_BYTES]
    encryptor = Cipher(algorithms.AES(AUTH_AES_KEY), modes.ECB()).encryptor()  # noqa: S305
    return (encryptor.update(block) + encryptor.finalize()).hex()


def logic_check(inp: str, nonce: str) -> str:
    """Index into `inp` with the low nibble of each character of `nonce`.

    Unchanged in shape from the retired protocol, but it is now fed the RAW base64 nonce
    rather than a decrypted one, and an out-of-range index emits `.` rather than raising.
    """
    return "".join(inp[ord(c) & 0xF] if (ord(c) & 0xF) < len(inp) else "." for c in nonce)


def normalize_version(version: str) -> str:
    """The four-part `BINARY_SW_VERSION` FUS wants, out of the three-part form the index
    publishes: `version.xml` states `PDA/CSC/PHONE` and FUS wants the PDA repeated."""
    parts = version.split("/")
    if len(parts) == 3:
        parts.append(parts[0])
    if len(parts) >= 3 and not parts[2]:
        parts[2] = parts[0]
    return "/".join(parts)


def _latest_offer(
    text: str, *, model: str, region: str, source_url: str
) -> tuple[ElementTree.Element, str]:
    """The `<latest>` element and its validated four-part version.

    One function rather than two, because BOTH are read on both paths — `list_available`
    wants the element's `o` attribute and `fetch` wants the version — and validating in only
    one of them is how the check ends up on the path nobody attacks. (It did, briefly: the
    version charset was enforced in `latest_version` while `parse_version_index` reached
    around it to the same element.)
    """
    root = parse_xml(text, context=f"samsung index[{model}/{region}]")
    latest = root.find("./firmware/version/latest")
    if latest is None or not (latest.text or "").strip():
        raise EmptyFirmwareIndexError(
            f"latest_version: {source_url} answered 200 with no <latest> build for "
            f"{model}/{region}. That pair exists (a pair that does not answers 403), so an "
            "empty index means the document's shape changed, not that Samsung published "
            "nothing for it."
        )
    version = normalize_version((latest.text or "").strip())
    if not _VERSION_RE.match(version):
        # This string is interpolated into the outgoing FUS body. Refused rather than escaped,
        # the same call every other externally-supplied field in this module gets.
        raise SamsungProtocolError(
            f"latest_version: {source_url} published {version!r} for {model}/{region}, which is "
            "not a Samsung version string (`PDA/CSC/PHONE`, alphanumerics and `._-`); refusing "
            "to build a request body out of it"
        )
    return latest, version


def latest_version(text: str, *, model: str, region: str, source_url: str) -> str:
    """The four-part `BINARY_SW_VERSION` this (model, CSC) pair currently offers.

    Only `<latest>` is read. The `<upgrade>` list beside it carries older versions ordered by
    an `rcount` attribute whose meaning is not measured, and whether FUS still resolves one is
    unproven — so offering them would be a catalogue of builds that may not download.
    """
    return _latest_offer(text, model=model, region=region, source_url=source_url)[1]


def parse_version_index(text: str, *, model: str, region: str, source_url: str) -> FirmwareRef:
    """The one build this (model, CSC) pair currently offers, as a ref.

    The CSC goes into the ref's BUILD rather than its device, the same call the Motorola
    driver makes about its channels: two CSCs of one model are two regional firmware lines of
    one phone, and putting the CSC in `device` would make that phone count twice toward
    `package_facts.device_count`, which is the triage ranking signal.

    `url` is the INDEX document rather than a download URL, because a Samsung build has no
    download URL until an authenticated inform call produces one. `fetch` re-reads this same
    document to resolve it, so the ref names where the offer came from and nothing else.
    """
    latest, version = _latest_offer(text, model=model, region=region, source_url=source_url)
    # `o="16"` on that element is the Android version, and it is the only place either
    # document states one.
    android = latest.attrib.get("o")
    try:
        return FirmwareRef(
            driver=SamsungDriver.name,
            device=model,
            build=f"{version.split('/')[0]}_{region}",
            url=source_url,
            android_version=android or None,
        )
    except ValueError as exc:
        raise SamsungProtocolError(
            f"parse_version_index: {source_url} published {version!r} for {model}/{region}, "
            f"which does not validate as a build id: {exc}"
        ) from exc


@dataclass(frozen=True, slots=True)
class SamsungBinary:
    """What one `BinaryInform` call resolved: where the binary is, and the key that opens it."""

    version: str
    filename: str
    model_path: str
    size: int
    key: bytes
    model_type: str
    region: str
    # `BINARY_CRC` is the CRC32 of the ENCRYPTED body, measured 2026-08-11: the full 11.5 GB
    # `SM-S911U`/`XAA` download hashes to 949352961, which is exactly what the inform response
    # declared, while the plaintext's CRC (3102581189) is not. So Samsung does publish an
    # integrity value for the download, and it is the only one either half of this protocol
    # offers. None when the response omits it rather than assumed.
    crc32: int | None = None

    @property
    def download_url(self) -> str:
        # `quote` over the two validated fields as well: they are the only operator-invisible
        # values in this URL, and a validated charset plus an encoder is what keeps a future
        # widening of either regex from becoming a query-injection.
        return f"{DOWNLOAD_URL}?file={quote(self.model_path + self.filename, safe='/')}"


def require_s00(text: str, *, context: str) -> ElementTree.Element:
    """Parse a FUS response and refuse anything but a success status.

    **FUS declines with HTTP 200 and a status in the body.** That is why every call here reads
    the status rather than the status code — including the download-authorisation call, whose
    only symptom otherwise arrives one request later as a 401 on the binary itself, which
    reads like an expired URL and is not one.
    """
    root = parse_xml(text, context=context)
    status = root.find("./FUSBody/Results/Status")
    status_text = (status.text or "").strip() if status is not None else ""
    if status_text not in {"S00", "200"}:
        raise SamsungProtocolError(
            f"{context}: FUS answered status {status_text or '(none)'}, not S00. It declines "
            "with HTTP 200 and a status in the body, so this is a refusal rather than a "
            "transport failure; re-list the index before retrying this build."
        )
    return root


def parse_binary_inform(text: str, *, model: str, region: str) -> SamsungBinary:
    """The resolved binary, or a named protocol failure.

    The decryption key is `md5(logic_check(BINARY_SW_VERSION, LOGIC_VALUE_FACTORY))`, and both
    inputs arrive in THIS response, so nothing extra is fetched to decrypt what it resolves.
    """
    root = require_s00(text, context=f"parse_binary_inform[{model}/{region}]")
    # Scoped to `Put` rather than harvested from the whole document: `BINARY_SW_VERSION`
    # appears in `Results` as well, and a whole-document walk picks whichever comes last —
    # which is right by document order alone, and this is the field the decryption key is
    # derived from.
    body = root.find("./FUSBody/Put")
    if body is None:
        raise SamsungProtocolError(
            f"parse_binary_inform: FUS answered S00 for {model}/{region} with no FUSBody/Put "
            "block, which is where every field the download needs lives"
        )
    fields = {
        element.tag: (element.find("Data").text or "").strip()
        for element in body
        if element.find("Data") is not None and element.find("Data").text is not None
    }

    def require(*names: str) -> str:
        for name in names:
            if fields.get(name):
                return fields[name]
        raise SamsungProtocolError(
            f"parse_binary_inform: FUS answered S00 for {model}/{region} without "
            f"{' or '.join(names)}. The response carried {len(fields)} fields, so the protocol "
            "changed rather than the request failing."
        )

    # `BINARY_SW_VERSION` first and `LATEST_FW_VERSION` second: the latter is what the retired
    # clients read and it is ABSENT from this response, so a client reading only it derives
    # its key from nothing and decrypts to garbage of exactly the right size.
    version = require("BINARY_SW_VERSION", "LATEST_FW_VERSION")
    logic_value = require("LOGIC_VALUE_FACTORY", "LOGIC_VALUE_HOME")
    filename = require("BINARY_NAME")
    model_path = require("MODEL_PATH")
    model_type = require("DEVICE_MODEL_TYPE")
    if not _BINARY_NAME_RE.match(filename) or not _MODEL_PATH_RE.match(model_path):
        raise SamsungProtocolError(
            f"parse_binary_inform: FUS named {model_path!r} + {filename!r} for {model}/{region}, "
            "which is not the shape a Samsung binary path has (`/neofus/911/` plus a plain "
            "filename of at least 25 characters); refusing to build a request URL out of it"
        )
    # Both reach an outgoing FUS body: the version through the inform LOGIC_CHECK and the
    # model type through the download authorisation.
    if not _VERSION_RE.match(version) or not _MODEL_TYPE_RE.match(model_type):
        raise SamsungProtocolError(
            f"parse_binary_inform: FUS answered version {version!r} and model type "
            f"{model_type!r} for {model}/{region}. Both are interpolated into the next request "
            "body, so a character outside their charsets is refused rather than escaped."
        )
    size_text = require("BINARY_BYTE_SIZE")
    try:
        size = int(size_text)
    except ValueError as exc:
        raise SamsungProtocolError(
            f"parse_binary_inform: BINARY_BYTE_SIZE is {size_text!r} for {model}/{region}, "
            "not a byte count"
        ) from exc
    if size <= 0 or size % AES_BLOCK_BYTES:
        raise SamsungProtocolError(
            f"parse_binary_inform: FUS declares {size} bytes for {model}/{region}, which is not "
            "a positive multiple of the 16-byte AES block every `.enc4` is padded to"
        )
    declared_crc = fields.get("BINARY_CRC", "")
    if declared_crc and not _CRC32_RE.match(declared_crc):
        raise SamsungProtocolError(
            f"parse_binary_inform: BINARY_CRC is {declared_crc!r} for {model}/{region}, not the "
            "unsigned 32-bit CRC of the encrypted body that this field has always carried"
        )
    return SamsungBinary(
        crc32=int(declared_crc) if declared_crc else None,
        version=version,
        filename=filename,
        model_path=model_path,
        size=size,
        key=hashlib.md5(logic_check(version, logic_value).encode()).digest(),  # noqa: S324
        model_type=model_type,
        region=require("BINARY_LOCAL_CODE"),
    )


def decrypt_archive(
    path: Path, key: bytes, *, context: str, expected_crc32: int | None = None
) -> str:
    """AES-128-ECB decrypt `path` in place, strip its PKCS#7 padding, return the plaintext's
    sha256. Blocking: every caller runs it through `asyncio.to_thread`.

    In place rather than into a sibling, because the alternative is a second 11.5 GB file for
    the length of the decrypt. ECB is what makes that safe: no block depends on another, so
    the plaintext of a block is written exactly where its ciphertext was, and the cipher can
    be resumed at any 16-byte boundary with no state carried across it.

    Two structural checks, both of which a wrong key fails and neither of which any exception
    would otherwise report — a wrong key produces a file of exactly the right size and
    complete garbage:

    - the padding must be valid PKCS#7 (measured: 14 bytes of `0x0e` on the real archive);
    - the result must open with the zip magic.

    `expected_crc32` is `BINARY_CRC` from the inform response, which is the CRC32 of the
    ENCRYPTED body. It is checked HERE because this is the pass that already reads every
    ciphertext byte — computing it during the download would mean two passes over 11.5 GB, and
    computing it afterwards is impossible once the file has been decrypted in place.
    """
    size = path.stat().st_size
    if size == 0 or size % AES_BLOCK_BYTES:
        raise FirmwareDownloadError(
            f"{context}: {path.name} is {size} bytes, not a positive multiple of the 16-byte "
            "AES block. The download is truncated or this is not a `.enc4` at all."
        )
    decryptor = Cipher(algorithms.AES(key), modes.ECB()).decryptor()  # noqa: S305
    digest = hashlib.sha256()
    ciphertext_crc = 0
    tail = b""
    with path.open("r+b") as fh:
        offset = 0
        while offset < size:
            fh.seek(offset)
            chunk = fh.read(min(_DECRYPT_CHUNK_BYTES, size - offset))
            ciphertext_crc = zlib.crc32(chunk, ciphertext_crc)
            plain = decryptor.update(chunk)
            fh.seek(offset)
            fh.write(plain)
            offset += len(plain)
            if offset < size:
                digest.update(plain)
            else:
                # The final block is held back: how much of it is padding is only known once
                # the whole file has been read, and the digest must cover the plaintext this
                # pipeline keeps rather than the padded form that was on the wire.
                tail = plain[-AES_BLOCK_BYTES:]
                digest.update(plain[:-AES_BLOCK_BYTES])
        decryptor.finalize()

        if expected_crc32 is not None and ciphertext_crc != expected_crc32:
            # Checked before the padding, because this one names the right suspect: a body
            # that did not arrive intact is a transfer failure, and the padding check below
            # would report the same corruption as a wrong decryption key.
            raise FirmwareDownloadError(
                f"{context}: the encrypted body hashes to CRC32 {ciphertext_crc}, but FUS "
                f"declared {expected_crc32}. The download is corrupt rather than mis-keyed; "
                "retry the job."
            )
        padding = tail[-1]
        if not 1 <= padding <= AES_BLOCK_BYTES or tail[-padding:] != bytes([padding]) * padding:
            raise FirmwareDownloadError(
                f"{context}: {path.name} decrypted to invalid PKCS#7 padding "
                f"({tail[-AES_BLOCK_BYTES:]!r}). The key is wrong or the body was corrupted: "
                "FUS derives it from the same response that named this file, so re-list the "
                "index and retry rather than reusing this ref."
            )
        digest.update(tail[: AES_BLOCK_BYTES - padding])
        fh.truncate(size - padding)

        fh.seek(0)
        magic = fh.read(len(_ZIP_MAGIC))
    if magic != _ZIP_MAGIC:
        raise FirmwareDownloadError(
            f"{context}: {path.name} decrypted to {magic!r}, not the zip magic. A wrong key "
            "yields a file of exactly the right size and complete garbage, so this is checked "
            "rather than left for a later stage to report as an unopenable archive."
        )
    return digest.hexdigest()


class FusSession:
    """The nonce every FUS request is signed with.

    Mutable and short-lived on purpose: the server issues a fresh `NONCE` header on nearly
    every response and the NEXT request must be signed with it, so this is rotated in place
    rather than passed around as a value.
    """

    def __init__(self) -> None:
        self.nonce = ""
        self.signature = ""

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": (
                f'FUS nonce="{self.nonce}", signature="{self.signature}", '
                'nc="", type="", realm="", newauth="1"'
            ),
            # `Kies2.0_FUS` belongs to the retired protocol.
            "User-Agent": USER_AGENT,
        }

    def rotate(self, response: httpx.Response) -> None:
        nonce = response.headers.get("NONCE") or response.headers.get("nonce")
        if nonce:
            self.nonce = nonce
            self.signature = auth_signature(nonce)


def binary_inform_body(*, model: str, region: str, version: str, nonce: str) -> str:
    return (
        "<FUSMsg>"
        "<FUSHdr><ProtoVer>1.0</ProtoVer><SessionID>0</SessionID><MsgID>1</MsgID></FUSHdr>"
        "<FUSBody><Put><CmdID>1</CmdID>"
        "<ACCESS_MODE><Data>1</Data></ACCESS_MODE>"
        "<BINARY_NATURE><Data>1</Data></BINARY_NATURE>"
        "<REQUEST_TYPE><Data>2</Data></REQUEST_TYPE>"
        f"<LOGIC_CHECK><Data>{logic_check(version, nonce)}</Data></LOGIC_CHECK>"
        f"<BINARY_SW_VERSION><Data>{version}</Data></BINARY_SW_VERSION>"
        f"<BINARY_LOCAL_CODE><Data>{region}</Data></BINARY_LOCAL_CODE>"
        f"<BINARY_MODEL_NAME><Data>{model}</Data></BINARY_MODEL_NAME>"
        "</Put><Get><CmdID>2</CmdID><BINARY_SW_VERSION></BINARY_SW_VERSION></Get>"
        "</FUSBody></FUSMsg>"
    )


def binary_init_body(binary: SamsungBinary, *, nonce: str) -> str:
    """The download authorisation FUS wants before it serves any byte of the binary.

    `LOGIC_CHECK` is computed over `filename[-25:-9]` — the 16 characters ending just before
    `.zip.enc4`. Slicing from the END is what makes it independent of how long the model name
    is; an index from the front is right for one model and silently wrong for the next.
    """
    checked = binary.filename[len(binary.filename) - 25 : len(binary.filename) - 9]
    return (
        "<FUSMsg>"
        "<FUSHdr><ProtoVer>1.0</ProtoVer><SessionID>0</SessionID><MsgID>1</MsgID></FUSHdr>"
        "<FUSBody><Put>"
        f"<BINARY_NAME><Data>{binary.filename}</Data></BINARY_NAME>"
        f"<BINARY_SW_VERSION><Data>{binary.version}</Data></BINARY_SW_VERSION>"
        f"<DEVICE_LOCAL_CODE><Data>{binary.region}</Data></DEVICE_LOCAL_CODE>"
        f"<DEVICE_MODEL_TYPE><Data>{binary.model_type}</Data></DEVICE_MODEL_TYPE>"
        f"<LOGIC_CHECK><Data>{logic_check(checked, nonce)}</Data></LOGIC_CHECK>"
        "</Put></FUSBody></FUSMsg>"
    )


class SamsungDriver(FirmwareDriver):
    name = "samsung"

    def __init__(self, settings: Settings, *, client: httpx.AsyncClient | None = None) -> None:
        self._index_url = settings.samsung_index_url.rstrip("/")
        self._models = settings.samsung_model_names
        self._regions = settings.samsung_region_names
        self._timeout = settings.firmware_http_timeout_seconds
        self._max_archive_bytes = settings.max_firmware_archive_bytes
        # Injected only by tests (a mock transport); production builds one per call so no
        # connection pool outlives the stage that opened it.
        self._client = client

    def terms(self) -> TermsPosture:
        return TermsPosture(
            risk=TermsRisk.REVERSE_ENGINEERED,
            source_url=self._index_url,
            summary=(
                "Samsung's own servers, in two halves. The version index is public and "
                "unauthenticated. The binary comes from FUS, a private endpoint Samsung "
                "publishes no terms for and no documentation of, reached by reproducing its "
                "client's handshake — no credential is presented and nothing is circumvented, "
                "but this is a protocol nobody was invited to speak, which is a materially "
                "different posture from a public CDN and will break without warning."
            ),
            # Nothing to acknowledge: there is no gate and no acceptance Samsung offers to
            # record. `risk` is what carries the exposure here, not this flag.
            acknowledged=True,
        )

    def _open_client(self) -> tuple[httpx.AsyncClient, bool]:
        """The client plus whether the caller owns closing it."""
        if self._client is not None:
            return self._client, False
        return httpx.AsyncClient(timeout=httpx.Timeout(self._timeout), follow_redirects=True), True

    def _index_url_for(self, model: str, region: str) -> str:
        return f"{self._index_url}/{region}/{model}/version.xml"

    def _validated_targets(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if not self._models or not self._regions:
            raise FirmwareInputError(
                "SamsungDriver: no models or no CSC regions configured. Samsung publishes no "
                "catalogue of either — the index answers one CSC/model pair at a time and 403s "
                "for a pair that does not exist — so enumeration is a probe over a grid the "
                "operator names: SAMSUNG_MODELS=SM-S911U,SM-S928B and SAMSUNG_REGIONS=XAA,EUX. "
                "Disable the driver with DISABLED_FIRMWARE_DRIVERS=samsung instead."
            )
        bad_models = [model for model in self._models if not _MODEL_RE.match(model)]
        bad_regions = [region for region in self._regions if not _REGION_RE.match(region)]
        if bad_models or bad_regions:
            raise FirmwareInputError(
                f"SamsungDriver: SAMSUNG_MODELS carries {bad_models!r} and SAMSUNG_REGIONS "
                f"carries {bad_regions!r}, which are not a Samsung model (`SM-S928B`) and a "
                "three-character CSC (`XAA`). Both reach a URL, so both are refused rather "
                "than transliterated."
            )
        probes = len(self._models) * len(self._regions)
        if probes > MAX_INDEX_PROBES:
            raise FirmwareInputError(
                f"SamsungDriver: {len(self._models)} model(s) against {len(self._regions)} "
                f"CSC(s) is {probes} index requests, past the {MAX_INDEX_PROBES} ceiling. The "
                "grid multiplies, so narrow one axis rather than raising this."
            )
        return self._models, self._regions

    async def list_available(self) -> list[FirmwareRef]:
        """Every configured (model, CSC) pair that exists, in the operator's configured order.

        That order is load-bearing and is the reason it is the operator's: no Samsung PDA
        version carries a parseable date, so `select_ref` resolves "newest" as the LAST row
        for a device — and across two CSCs of one model there is no "newer", because they are
        two regional firmware lines rather than two builds of one. A model configured against
        several CSCs therefore resolves to the last CSC in `SAMSUNG_REGIONS` unless the job
        names the build outright.
        """
        models, regions = self._validated_targets()
        client, owned = self._open_client()
        refs: list[FirmwareRef] = []
        missing: list[str] = []
        try:
            for model in models:
                for region in regions:
                    document = await self._index_document(client, model=model, region=region)
                    if document is None:
                        missing.append(f"{model}/{region}")
                    else:
                        refs.append(
                            parse_version_index(
                                document,
                                model=model,
                                region=region,
                                source_url=self._index_url_for(model, region),
                            )
                        )
        finally:
            if owned:
                await client.aclose()
        if missing:
            # INFO, not a warning: most of a model x CSC grid legitimately does not exist —
            # a model is sold under a handful of CSCs and the cross product names the rest.
            logger.info(
                "samsung: %d of %d configured pair(s) do not exist and are not on offer: %s",
                len(missing),
                len(models) * len(regions),
                ", ".join(missing),
            )
        # A MODEL that resolved nothing is a different thing from a pair that did, and it is
        # the shape a typo takes: the operator believes they are tracking a phone and the grid
        # quietly carries none of it. Same call the Motorola driver makes per device.
        for model in models:
            if not any(ref.device == model for ref in refs):
                logger.warning(
                    "samsung: %r resolved to no build under any of %d configured CSC(s) (%s); "
                    "check the model name and its CSCs",
                    model,
                    len(regions),
                    ", ".join(regions),
                )
        if not refs:
            raise EmptyFirmwareIndexError(
                f"SamsungDriver.list_available: not one of {len(models) * len(regions)} "
                f"configured pair(s) exists ({', '.join(missing)}). A 403 from this index means "
                "the CSC and the model do not go together, so check the pairing: SM-A546B is "
                "published under EUX and 403s under DBT, XEO, BTU and ATT."
            )
        logger.info(
            "samsung index: %d build(s) across %d model(s)",
            len(refs),
            len({ref.device for ref in refs}),
        )
        return refs

    async def _index_document(
        self, client: httpx.AsyncClient, *, model: str, region: str
    ) -> str | None:
        """This pair's `version.xml`, or None when the pair does not exist.

        **A 403 is not a failure here.** It is how this index says "no such CSC and model",
        with an S3-style `<Error><Code>AccessDenied</Code>` body; coding it as an error would
        make a grid with one wrong pairing fail whole.
        """
        url = self._index_url_for(model, region)
        try:
            async with client.stream("GET", url) as response:
                if response.status_code == httpx.codes.FORBIDDEN:
                    return None
                if response.status_code != httpx.codes.OK:
                    await response.aread()
                    raise FirmwareError(
                        f"SamsungDriver: {url} answered HTTP {response.status_code}; expected "
                        "200, or 403 for a CSC and model that do not go together"
                    )
                return await _read_capped(response, context=f"samsung index[{model}/{region}]")
        except httpx.HTTPError as exc:
            raise FirmwareError(f"SamsungDriver: {url} is unreachable: {exc}") from exc

    async def _post(
        self, client: httpx.AsyncClient, session: FusSession, path: str, body: str = ""
    ) -> str:
        """One FUS call, with its status checked HERE rather than by each caller.

        **FUS declines with HTTP 200 and a `<Status>` in the body**, so a caller that discards
        the response discards the refusal. Checking it in the one place every call goes
        through is what makes that unrepresentable: the download-authorisation call had no
        other symptom than a 401 on the binary one request later, which reads as an expired
        URL. Measured: every response carries the element, `GenerateNonce` included (it
        answers `<Status>200</Status>` with a 228-byte body).

        `parse_binary_inform` checks the status again on its own text, and that duplicate
        parse of a 4 KB document once per job is deliberate — it is the unit the fixture tests
        drive, and it has to be correct without a caller having gone first.
        """
        url = f"{FUS_URL}/{path}"
        try:
            async with client.stream(
                "POST", url, headers=session.headers(), content=body
            ) as response:
                if response.status_code != httpx.codes.OK:
                    await response.aread()
                    raise SamsungProtocolError(
                        f"SamsungDriver: {url} answered HTTP {response.status_code}. FUS "
                        "answers 401 once a nonce has expired and 403 when the handshake is "
                        "wrong, and both mean this module's transcription of the protocol no "
                        "longer matches the server."
                    )
                text = await _read_capped(response, context=f"SamsungDriver[{path}]")
                session.rotate(response)
        except httpx.HTTPError as exc:
            raise FirmwareError(f"SamsungDriver: {url} is unreachable: {exc}") from exc
        require_s00(text, context=f"SamsungDriver[{path}]")
        return text

    async def resolve(
        self, client: httpx.AsyncClient, ref: FirmwareRef
    ) -> tuple[FusSession, SamsungBinary]:
        """Turn a ref back into a downloadable binary, re-reading the public index first.

        Public, and separate from `fetch`, because it is the whole authenticated handshake and
        costs four small requests: the live test proves the protocol still works by resolving
        a real build and reading 64 KB of it, which is the entire auth chain end to end for
        the price of a page rather than of an 11.5 GB transfer.

        The index is re-read rather than trusted from the ref for the same reason the Nothing
        driver re-reads its release: a Samsung ref carries no download URL — none exists until
        an authenticated inform call produces one — so the four-part version FUS wants has to
        come from somewhere, and the somewhere is the document that published this build. A
        ref whose build the index no longer offers is refused rather than silently resolved to
        whatever is current, because the job named a build.
        """
        named = _BUILD_RE.match(ref.build)
        if named is None:
            raise FirmwareInputError(
                f"SamsungDriver.fetch: {ref.build!r} is not a Samsung build id. This driver "
                "spells one `<PDA>_<CSC>` (`S911USQS8FZG1_XAA`), because a build is only "
                "identified by the pair."
            )
        model, region, pda = ref.device, named.group("region"), named.group("pda")
        if not _MODEL_RE.match(model):
            raise FirmwareInputError(
                f"SamsungDriver.fetch: {model!r} is not a Samsung model name (`SM-S928B`)"
            )
        document = await self._index_document(client, model=model, region=region)
        if document is None:
            raise FirmwareError(
                f"SamsungDriver.fetch: the index no longer publishes {model}/{region} at all "
                "(HTTP 403, which is how it says the pair does not exist); re-list before "
                "fetching"
            )
        # The whole four-part version, not a reconstruction from the PDA: the CSC and PHONE
        # parts differ from it (`S911USQS8FZG1/S911UOYN8FZG1/S911USQS8FZG1`), and FUS checks
        # the LOGIC_CHECK derived from the whole string.
        version = latest_version(
            document, model=model, region=region, source_url=self._index_url_for(model, region)
        )
        if version.split("/")[0] != pda:
            raise FirmwareInputError(
                f"SamsungDriver.fetch: {model}/{region} now publishes "
                f"{version.split('/')[0]}_{region}, not the {ref.build} this ref names. "
                "Samsung's index carries only the current build per pair, so an older one "
                "cannot be resolved: re-list the index and requeue."
            )

        session = FusSession()
        await self._post(client, session, "NF_SmartDownloadGenerateNonce.do")
        if not session.nonce:
            # Every later request is signed with this. Absent, they go out as `nonce=""`,
            # `signature=""` and a LOGIC_CHECK of the empty string — which FUS rejects one
            # call later, blaming the call that was built correctly.
            raise SamsungProtocolError(
                "SamsungDriver.fetch: NF_SmartDownloadGenerateNonce.do answered 200 without a "
                "NONCE header. Every later request is signed with that value, so there is "
                "nothing to sign with and the protocol has changed."
            )
        inform = await self._post(
            client,
            session,
            "NF_SmartDownloadBinaryInform.do",
            binary_inform_body(model=model, region=region, version=version, nonce=session.nonce),
        )
        binary = parse_binary_inform(inform, model=model, region=region)
        if binary.region != region:
            raise SamsungProtocolError(
                f"SamsungDriver.fetch: asked FUS for {model}/{region} and it resolved a binary "
                f"for {binary.region}; refusing to download firmware for a region nobody asked "
                "for"
            )
        # The download authorisation. Its status is checked inside `_post` with every other
        # FUS response, which is the point: declined, it answers HTTP 200 and its only other
        # symptom is a 401 on the binary one request later.
        await self._post(
            client,
            session,
            "NF_SmartDownloadBinaryInitForMass.do",
            binary_init_body(binary, nonce=session.nonce),
        )
        return session, binary

    async def fetch(self, ref: FirmwareRef, dest_dir: Path) -> DownloadedArchive:
        if ref.driver != self.name:
            raise FirmwareInputError(
                f"SamsungDriver.fetch: ref belongs to driver {ref.driver!r}, not {self.name!r}; "
                "route it to the driver that produced it"
            )
        if ref.md5 is not None:
            # Refused rather than dropped. Every other driver hands `published_digest()` to
            # `download_to_file`, and doing that here would check an md5 of the CIPHERTEXT
            # against a digest an operator computed over an archive — a mismatch that reads as
            # a corrupt download and is not one. `sha256` is honoured, against the plaintext,
            # and the source's own CRC32 over the ciphertext is checked regardless.
            raise FirmwareInputError(
                f"SamsungDriver.fetch: {ref.build} pins an md5. FUS serves an encrypted body, "
                "so a digest can only be checked against the decrypted archive, and this "
                "driver computes sha256 there — pin `sha256` instead."
            )
        client, owned = self._open_client()
        try:
            session, binary = await self.resolve(client, ref)
            if binary.size > self._max_archive_bytes:
                # FUS states the size before the first byte, so the ceiling is enforced here
                # rather than only mid-stream by `download_to_file`. Otherwise a build past it
                # costs a full-ceiling transfer, and that much of a scratch disk shared with
                # every other job, before anything says no.
                raise FirmwareDownloadError(
                    f"SamsungDriver.fetch: FUS declares {binary.size} bytes for {ref.build}, "
                    f"past the {self._max_archive_bytes}-byte ceiling. Raise "
                    "MAX_FIRMWARE_ARCHIVE_BYTES if a firmware is genuinely that large; nothing "
                    "was downloaded."
                )
            dest = dest_dir / ref.archive_filename
            encrypted = dest.with_name(dest.name + ENCRYPTED_SUFFIX)
            logger.info(
                "samsung: %s resolves to %s (%d bytes, encrypted)",
                ref.build,
                binary.filename,
                binary.size,
            )
            try:
                await download_to_file(
                    client,
                    binary.download_url,
                    encrypted,
                    headers=session.headers(),
                    max_bytes=self._max_archive_bytes,
                )
                arrived = encrypted.stat().st_size
                if arrived != binary.size:
                    # Cheaper and more specific than the CRC below, which only fires after
                    # 11.5 GB has been read again: a short body that happens to end on a
                    # 16-byte boundary would otherwise be reported by whichever check runs
                    # next, and neither of those names the transfer.
                    raise FirmwareDownloadError(
                        f"SamsungDriver.fetch: {binary.filename} arrived as {arrived} bytes, but "
                        f"FUS declared {binary.size}. The transfer was truncated (or the build "
                        "was replaced mid-download); retry the job."
                    )
                sha256 = await asyncio.to_thread(
                    decrypt_archive,
                    encrypted,
                    binary.key,
                    context=f"SamsungDriver.fetch[{ref.build}]",
                    expected_crc32=binary.crc32,
                )
                if ref.sha256 is not None and sha256 != ref.sha256:
                    # An operator can pin a digest in a job's params. For every other driver
                    # that pin is checked against the download; here the download is
                    # ciphertext, so it is checked against the PLAINTEXT this driver keeps —
                    # which is the artifact the digest identifies everywhere else.
                    raise FirmwareDownloadError(
                        f"SamsungDriver.fetch: {ref.build} decrypted to sha256 {sha256}, but "
                        f"this job pinned {ref.sha256}. The pin is checked against the "
                        "decrypted archive, since FUS serves ciphertext and publishes no "
                        "digest of either form."
                    )
                encrypted.replace(dest)
            except BaseException:
                # Cancellation included: a reclaim lands here holding gigabytes that nothing
                # else is tracking, in a file no later stage would recognise.
                encrypted.unlink(missing_ok=True)
                raise
        finally:
            if owned:
                await client.aclose()
        # Verified when something OUTSIDE this pipeline said what the bytes should be: FUS's
        # own `BINARY_CRC` over the encrypted body, or a sha256 an operator pinned in the
        # job's params. Both are checked above; this only records which of them was available.
        verified = binary.crc32 is not None or ref.sha256 is not None
        if not verified:
            logger.warning(
                "%s decrypted to %d bytes, sha256=%s: this response carried no BINARY_CRC and "
                "the job pinned no digest, so the archive is integrity-unverified — what IS "
                "proven is that it decrypted to valid PKCS#7 padding and the zip magic",
                dest.name,
                dest.stat().st_size,
                sha256,
            )
        else:
            logger.info(
                "%s decrypted to %d bytes, sha256=%s, verified against %s",
                dest.name,
                dest.stat().st_size,
                sha256,
                " and ".join(
                    filter(
                        None,
                        [
                            f"the CRC32 FUS published ({binary.crc32})" if binary.crc32 else "",
                            "the sha256 this job pinned" if ref.sha256 else "",
                        ],
                    )
                ),
            )
        return DownloadedArchive(path=dest, sha256=sha256, integrity_verified=verified)
