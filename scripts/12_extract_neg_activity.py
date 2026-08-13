#!/usr/bin/env python3
"""Stream-extract not_bought_deployers_activity.parquet from the TAR (~23.5 GiB on disk).

Network is larger than 23 GiB: the TAR has no HTTP Range, so we must transfer
every member before it (bought_* + not_bought jsonl/index, ~15 GiB prefix).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import ensure_dirs, format_bytes, load_config, max_auto_download_bytes  # noqa: E402
from src.download.tar_stream import extract_members_from_url  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--i-approve-large-download", action="store_true")
    parser.add_argument("--timeout", type=float, default=86400.0)
    args = parser.parse_args()

    cfg = load_config()
    paths = ensure_dirs(cfg)
    dest_dir = paths["negatives"]
    member = (cfg.get("tar_members") or {}).get(
        "negatives_activity", "not_bought_deployers_activity.parquet"
    )
    tar_url = (cfg.get("urls") or {}).get("half_year_dataset_tar")
    # prefix ~15 GiB + ~23.5 GiB body ≈ full archive
    estimate = 39_000_000_000
    limit = max_auto_download_bytes(cfg)

    print("=== extract not_bought_deployers_activity ===")
    print(f"URL: {tar_url}")
    print(f"Member: {member}")
    print(f"Dest: {dest_dir}")
    print(f"Disk file ~23.5 GiB; network worst-case ~{format_bytes(estimate)} (no Range)")
    print(f"Auto limit: {format_bytes(limit)}")

    if not args.i_approve_large_download:
        print("Refusing without --i-approve-large-download")
        return 2
    if not args.yes:
        print("Refusing without --yes")
        return 0

    result = extract_members_from_url(
        tar_url,
        [member],
        dest_dir,
        max_transfer_bytes=estimate + 2_000_000_000,
        approve_large=True,
        timeout=args.timeout,
        progress=True,
    )
    print(f"Complete: {result.get('complete')}")
    print(f"Transferred: {format_bytes(result.get('transferred_bytes'))}")
    for name, meta in (result.get("written") or {}).items():
        print(f"  {name}: {format_bytes(meta.get('size'))} -> {meta.get('path')}")
    if result.get("remaining_wanted"):
        print(f"Missing: {result['remaining_wanted']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
