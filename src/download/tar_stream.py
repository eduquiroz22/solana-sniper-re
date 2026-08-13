"""Streaming USTAR/POSIX TAR reader for selective member extraction."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Iterable
from urllib.request import Request, urlopen

BLOCK = 512
USER_AGENT = "solana-sniper/phase1-tar-stream"


def parse_ustar_header(hdr: bytes) -> dict:
    """Parse a 512-byte ustar header. Returns name, size, typeflag, etc."""
    if len(hdr) < BLOCK:
        raise ValueError(f"TAR header too short: {len(hdr)} bytes")
    if hdr == b"\x00" * BLOCK:
        return {"name": "", "size": 0, "typeflag": "end", "raw_empty": True}

    def _field(start: int, length: int) -> bytes:
        return hdr[start : start + length]

    def _c_str(b: bytes) -> str:
        return b.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()

    def _octal(b: bytes) -> int:
        s = _c_str(b).strip()
        if not s:
            return 0
        # GNU sparse / binary size: high bit set in first byte
        if len(b) >= 1 and b[0] & 0x80:
            n = 0
            for byte in b[1:]:
                n = (n << 8) | byte
            return n
        try:
            return int(s, 8)
        except ValueError:
            return 0

    name = _c_str(_field(0, 100))
    prefix = _c_str(_field(345, 155))
    if prefix:
        name = f"{prefix}/{name}" if name else prefix
    size = _octal(_field(124, 12))
    typeflag = chr(hdr[156]) if hdr[156] else "0"
    magic = _c_str(_field(257, 6))
    linkname = _c_str(_field(157, 100))

    return {
        "name": name,
        "size": size,
        "typeflag": typeflag,
        "magic": magic,
        "linkname": linkname,
        "raw_empty": False,
    }


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def discard_bytes(stream, n: int, chunk_size: int = 1024 * 1024) -> int:
    """Read and discard n bytes; return bytes actually discarded."""
    left = n
    got = 0
    while left > 0:
        chunk = stream.read(min(chunk_size, left))
        if not chunk:
            break
        got += len(chunk)
        left -= len(chunk)
    return got


def padded_size(size: int) -> int:
    if size <= 0:
        return 0
    rem = size % BLOCK
    return size if rem == 0 else size + (BLOCK - rem)


# Back-compat aliases used by early scripts
_discard = discard_bytes
_padded_size = padded_size


def iter_tar_members(
    stream,
    *,
    max_transfer_bytes: int | None = None,
    max_members: int | None = None,
    on_member: Callable[[dict], None] | None = None,
    discard_bodies: bool = True,
) -> list[dict]:
    """
    Iterate TAR members from a binary stream.

    By default discards file bodies (does not keep them). Tracks cumulative
    transfer (headers + bodies read). Stops when max_transfer_bytes or
    max_members would be exceeded on the next body.
    """
    members: list[dict] = []
    offset = 0
    transferred = 0
    empty_count = 0

    while True:
        if max_members is not None and len(members) >= max_members:
            break
        if max_transfer_bytes is not None and transferred + BLOCK > max_transfer_bytes:
            break

        hdr = _read_exact(stream, BLOCK)
        if len(hdr) < BLOCK:
            break
        transferred += len(hdr)
        header_offset = offset
        offset += BLOCK

        info = parse_ustar_header(hdr)
        if info.get("raw_empty"):
            empty_count += 1
            if empty_count >= 2:
                break
            continue
        empty_count = 0

        size = int(info["size"])
        padded = padded_size(size)
        next_transfer = transferred + padded

        if max_transfer_bytes is not None and next_transfer > max_transfer_bytes:
            # Would exceed budget to consume this body — stop without advancing body
            info_out = {
                **info,
                "offset": header_offset,
                "data_offset": header_offset + BLOCK,
                "padded_size": padded,
                "transferred_before_body": transferred,
                "truncated": True,
                "reason": "max_transfer_bytes",
            }
            members.append(info_out)
            if on_member:
                on_member(info_out)
            break

        if discard_bodies and padded:
            discarded = discard_bytes(stream, padded)
            transferred += discarded
            offset += discarded
            if discarded < padded:
                info["incomplete_body"] = True
        elif not discard_bodies and padded:
            # Caller must consume; we still track offset conceptually
            pass

        info_out = {
            **info,
            "offset": header_offset,
            "data_offset": header_offset + BLOCK,
            "padded_size": padded,
            "transferred_after": transferred,
            "truncated": False,
        }
        members.append(info_out)
        if on_member:
            on_member(info_out)

        # GNU long name / long link: typeflag L/K — name is in next data; we already
        # discarded body. For TOC purposes name may be "././@LongLink"; acceptable.

    return members


def extract_members_from_url(
    url: str,
    member_names: Iterable[str],
    dest_dir: str | Path,
    *,
    max_transfer_bytes: int | None = None,
    approve_large: bool = False,
    timeout: float = 300.0,
    progress: bool = True,
) -> dict:
    """
    Stream a TAR from url and write only wanted members under dest_dir.

    Stops after the last wanted member is extracted when possible (no need to
    finish the archive). Enforces max_transfer_bytes unless approve_large.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    wanted = {n.lstrip("./") for n in member_names}
    wanted_basenames = {Path(n).name for n in wanted}
    remaining = set(wanted) | set(wanted_basenames)
    written: dict[str, dict] = {}
    transferred = 0
    members_seen: list[dict] = []

    def _match(name: str) -> str | None:
        n = name.lstrip("./")
        if n in wanted:
            return n
        base = Path(n).name
        if base in wanted_basenames:
            # Prefer full relative if present
            for w in wanted:
                if Path(w).name == base:
                    return w
            return base
        return None

    def _check_budget(need: int) -> None:
        nonlocal transferred
        if max_transfer_bytes is None:
            return
        if transferred + need > max_transfer_bytes and not approve_large:
            raise RuntimeError(
                f"TAR stream would exceed max_transfer_bytes="
                f"{max_transfer_bytes} (transferred={transferred}, need+={need}). "
                "Pass approve_large / --i-approve-large-download."
            )

    # Prefer httpx streaming; fall back to urllib
    stream = None
    closer = None
    try:
        try:
            import httpx

            client = httpx.Client(follow_redirects=True, timeout=timeout)
            req = client.build_request("GET", url, headers={"User-Agent": USER_AGENT})
            resp = client.send(req, stream=True)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code} for {url}")

            class _HttpxReader:
                def __init__(self, r):
                    self._it = r.iter_bytes()
                    self._buf = bytearray()

                def read(self, n: int) -> bytes:
                    while len(self._buf) < n:
                        try:
                            chunk = next(self._it)
                        except StopIteration:
                            break
                        self._buf.extend(chunk)
                    out = bytes(self._buf[:n])
                    del self._buf[:n]
                    return out

            stream = _HttpxReader(resp)

            def closer():
                resp.close()
                client.close()

        except ImportError:
            ureq = Request(url, headers={"User-Agent": USER_AGENT})
            resp = urlopen(ureq, timeout=timeout)
            stream = resp

            def closer():
                resp.close()

        empty_count = 0
        offset = 0
        while remaining:
            _check_budget(BLOCK)
            hdr = _read_exact(stream, BLOCK)
            if len(hdr) < BLOCK:
                break
            transferred += len(hdr)
            header_offset = offset
            offset += BLOCK

            info = parse_ustar_header(hdr)
            if info.get("raw_empty"):
                empty_count += 1
                if empty_count >= 2:
                    break
                continue
            empty_count = 0

            size = int(info["size"])
            padded = padded_size(size)
            name = info["name"]
            typeflag = info["typeflag"]
            members_seen.append(
                {
                    "name": name,
                    "size": size,
                    "offset": header_offset,
                    "typeflag": typeflag,
                }
            )

            match = _match(name) if typeflag in ("0", "\0", "") else None
            if match and typeflag in ("0", "\0", ""):
                _check_budget(padded)
                out_name = Path(match).name
                out_path = dest_dir / out_name
                written_bytes = 0
                with out_path.open("wb") as out:
                    left = size
                    while left > 0:
                        chunk = stream.read(min(1024 * 1024, left))
                        if not chunk:
                            break
                        out.write(chunk)
                        written_bytes += len(chunk)
                        left -= len(chunk)
                        transferred += len(chunk)
                    # padding
                    pad = padded - size
                    if pad:
                        transferred += discard_bytes(stream, pad)
                offset += padded
                written[out_name] = {
                    "path": str(out_path),
                    "size": written_bytes,
                    "tar_name": name,
                    "tar_offset": header_offset,
                }
                remaining.discard(match)
                remaining.discard(out_name)
                remaining.discard(name.lstrip("./"))
                remaining.discard(Path(name).name)
                if progress:
                    sys.stderr.write(
                        f"  extracted {out_name} ({written_bytes:,} bytes)\n"
                    )
                if not remaining:
                    break
            else:
                _check_budget(padded)
                transferred += discard_bytes(stream, padded)
                offset += padded
                if progress and name:
                    sys.stderr.write(f"  skip {name} ({size:,} bytes)\n")

    finally:
        if closer:
            closer()

    return {
        "url": url,
        "dest_dir": str(dest_dir),
        "written": written,
        "members_seen": members_seen,
        "transferred_bytes": transferred,
        "remaining_wanted": sorted(remaining),
        "complete": len(remaining) == 0,
    }
