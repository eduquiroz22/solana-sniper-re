#!/usr/bin/env python3
"""Download positive (bought) deploy artifacts via TAR stream."""

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
from src.download.tar_stream import extract_members_from_url  # noqa: E402


def _tar_method(cfg: dict, dest_dir: Path, args: argparse.Namespace) -> int:
    urls = cfg.get("urls") or {}
    tar_url = urls.get(
        "half_year_dataset_tar",
        "http://65.21.203.147:48102/half_year_dataset.tar",
    )
    members = list(
        (cfg.get("tar_members") or {}).get("positives")
        or [
            "bought_deploy_txs.jsonl.gz",
            "bought_deploy_txs_index.parquet",
            "bought_deployers_activity.parquet",
        ]
    )
    estimate = int(
        (cfg.get("tar_members") or {}).get("positives_estimate_bytes", 710_000_000)
    )
    limit = max_auto_download_bytes(cfg)

    print("=== TAR stream-extract method ===")
    print(f"URL: {tar_url}")
    print(f"Members: {members}")
    print(f"Estimated transfer: {format_bytes(estimate)} (docs; under 1 GiB)")
    print(f"Auto limit: {format_bytes(limit)}")

    if estimate > limit and not args.i_approve_large_download:
        print(
            "Estimate exceeds max_auto_download_bytes without "
            "--i-approve-large-download; aborting."
        )
        return 2

    if not args.yes:
        print(
            "\nRefusing to start network transfer without --yes "
            "(prints estimate only). Re-run with --yes to extract."
        )
        return 0

    max_transfer = estimate + 64 * 1024 * 1024  # slack
    if max_transfer > limit and not args.i_approve_large_download:
        max_transfer = limit

    print("\nStreaming extract...")
    try:
        result = extract_members_from_url(
            tar_url,
            members,
            dest_dir,
            max_transfer_bytes=max_transfer,
            approve_large=args.i_approve_large_download,
            timeout=args.timeout,
            progress=True,
        )
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Transferred: {format_bytes(result['transferred_bytes'])}")
    print(f"Complete: {result['complete']}")
    if result["remaining_wanted"]:
        print(f"Missing members: {result['remaining_wanted']}")
    for name, meta in result["written"].items():
        path = Path(meta["path"])
        digest = sha256_file(path) if path.is_file() else None
        print(f"  {name}: {format_bytes(meta['size'])} sha256={digest}")
    return 0 if result["complete"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually perform TAR extract",
    )
    parser.add_argument(
        "--i-approve-large-download",
        action="store_true",
        help="Allow transfers above max_auto_download_bytes",
    )
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    cfg = load_config()
    paths = ensure_dirs(cfg)
    return _tar_method(cfg, paths["positives"], args)


if __name__ == "__main__":
    raise SystemExit(main())
