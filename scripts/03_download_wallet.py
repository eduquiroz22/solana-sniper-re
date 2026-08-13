#!/usr/bin/env python3
"""Download target-wallet activity files (small; default under 1 GiB)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    ensure_dirs,
    format_bytes,
    load_config,
    max_auto_download_bytes,
    sha256_file,
)
from src.download.http_utils import download_file, head_request  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-jsonl",
        action="store_true",
        help="Also download activity_txs.jsonl.gz",
    )
    parser.add_argument(
        "--i-approve-large-download",
        action="store_true",
        help="Allow downloads exceeding max_auto_download_bytes",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    cfg = load_config()
    paths = ensure_dirs(cfg)
    dest_dir = paths["wallet"]
    urls = cfg.get("urls") or {}
    base = urls.get("wallet_base", "http://154.12.118.112:48114/").rstrip("/") + "/"
    wf = cfg.get("wallet_files") or urls.get("wallet_files") or {
        "activity": "5brv79e_activity.parquet",
        "activity_txs_index": "5brv79e_activity_txs_index.parquet",
        "activity_txs_jsonl": "5brv79e_activity_txs.jsonl.gz",
    }

    names = [wf["activity"], wf["activity_txs_index"]]
    if args.with_jsonl:
        names.append(wf["activity_txs_jsonl"])

    limit = max_auto_download_bytes(cfg)
    print(f"Destination: {dest_dir}")
    print(f"Files: {names}")

    results = []
    for name in names:
        url = base + name
        dest = dest_dir / name
        head = head_request(url, timeout=args.timeout)
        cl = head.get("content_length")
        print(f"\n→ {name}")
        print(f"  HEAD status={head.get('status')} size={format_bytes(cl)}")
        if cl is not None and cl > limit and not args.i_approve_large_download:
            print(
                f"  SKIP: {format_bytes(cl)} exceeds limit {format_bytes(limit)} "
                "(pass --i-approve-large-download)"
            )
            results.append({"name": name, "skipped": True, "reason": "too_large"})
            continue
        info = download_file(
            url,
            dest,
            max_bytes=limit,
            resume=True,
            approve_large=args.i_approve_large_download,
            timeout=args.timeout,
        )
        digest = sha256_file(dest)
        size = dest.stat().st_size
        print(f"  saved {format_bytes(size)} sha256={digest}")
        results.append(
            {
                "name": name,
                "url": url,
                "dest": str(dest),
                "size": size,
                "sha256": digest,
                **{k: info.get(k) for k in ("bytes_written", "resumed", "skipped")},
            }
        )

    print("\n=== Summary ===")
    for r in results:
        if r.get("skipped") and r.get("reason"):
            print(f"  {r['name']}: skipped ({r['reason']})")
        else:
            print(f"  {r['name']}: {format_bytes(r.get('size'))} sha256={r.get('sha256')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
