#!/usr/bin/env python3
"""Stream TAR headers from the start; discard bodies; write TOC metadata."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    ensure_dirs,
    format_bytes,
    load_config,
    max_auto_download_bytes,
    write_json,
)
from src.download.tar_stream import iter_tar_members  # noqa: E402

USER_AGENT = "solana-sniper/tar-toc-probe"


def open_url_stream(url: str, timeout: float = 300.0):
    try:
        import httpx

        client = httpx.Client(follow_redirects=True, timeout=timeout)
        req = client.build_request("GET", url, headers={"User-Agent": USER_AGENT})
        resp = client.send(req, stream=True)
        if resp.status_code >= 400:
            resp.close()
            client.close()
            raise RuntimeError(f"HTTP {resp.status_code} for {url}")

        class _Reader:
            def __init__(self):
                self._it = resp.iter_bytes()
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

            def close(self):
                resp.close()
                client.close()

        return _Reader()
    except ImportError:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        return urlopen(req, timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-transfer-mib",
        type=float,
        default=180.0,
        help="Stop after transferring this many MiB (default 180)",
    )
    parser.add_argument("--max-members", type=int, default=32)
    parser.add_argument(
        "--i-approve-large-download",
        action="store_true",
        help="Allow transfer budget above max_auto_download_bytes",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    cfg = load_config()
    paths = ensure_dirs(cfg)
    urls = cfg.get("urls") or {}
    tar_url = urls.get(
        "half_year_dataset_tar",
        "http://65.21.203.147:48102/half_year_dataset.tar",
    )

    max_transfer = int(args.max_transfer_mib * 1024 * 1024)
    limit = max_auto_download_bytes(cfg)
    if max_transfer > limit and not args.i_approve_large_download:
        print(
            f"Refusing TOC probe budget {format_bytes(max_transfer)} "
            f"> limit {format_bytes(limit)}. "
            "Lower --max-transfer-mib or pass --i-approve-large-download."
        )
        return 2

    print(f"Streaming TAR TOC from {tar_url}")
    print(f"Max transfer: {format_bytes(max_transfer)}; max members: {args.max_members}")

    stream = open_url_stream(tar_url, timeout=args.timeout)
    try:

        def on_member(m: dict) -> None:
            flag = "TRUNCATED" if m.get("truncated") else "ok"
            print(
                f"  [{flag}] offset={m.get('offset')} size={m.get('size')} "
                f"name={m.get('name')}"
            )

        members = iter_tar_members(
            stream,
            max_transfer_bytes=max_transfer,
            max_members=args.max_members,
            on_member=on_member,
            discard_bodies=True,
        )
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()

    transferred = 0
    for m in members:
        if m.get("transferred_after") is not None:
            transferred = max(transferred, int(m["transferred_after"]))
        elif m.get("transferred_before_body") is not None:
            transferred = max(transferred, int(m["transferred_before_body"]))

    out_obj = {
        "url": tar_url,
        "max_transfer_bytes": max_transfer,
        "max_members": args.max_members,
        "members": [
            {
                "name": m.get("name"),
                "size": m.get("size"),
                "offset": m.get("offset"),
                "data_offset": m.get("data_offset"),
                "padded_size": m.get("padded_size"),
                "typeflag": m.get("typeflag"),
                "truncated": m.get("truncated", False),
                "reason": m.get("reason"),
            }
            for m in members
        ],
        "approx_transferred_bytes": transferred,
        "note": (
            "Bodies discarded; network still transfers member payloads up to the stop point. "
            "Server may ignore HTTP Range — stream is from archive start."
        ),
    }
    out = paths["metadata"] / "tar_toc.json"
    write_json(out, out_obj)
    print(f"\nMembers listed: {len(members)}")
    print(f"Approx transferred: {format_bytes(transferred)}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
