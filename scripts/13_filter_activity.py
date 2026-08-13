#!/usr/bin/env python3
"""Keep only activity rows for signers in positives + negative sample (lazy scan)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import ensure_dirs, format_bytes, load_config  # noqa: E402


def main() -> int:
    import polars as pl

    cfg = load_config()
    paths = ensure_dirs(cfg)
    raw = paths["negatives"] / "not_bought_deployers_activity.parquet"
    bought = paths["positives"] / "bought_deployers_activity.parquet"
    if not raw.is_file():
        print(f"Missing {raw}")
        return 1

    pos = pl.read_parquet(paths["positives"] / "bought_deploy_txs_index.parquet")
    neg = pl.read_parquet(paths["samples"] / "negative_200k.parquet")
    signers = (
        pl.concat(
            [
                pos.select(pl.col("tx_signer").alias("wallet")),
                neg.select(pl.col("tx_signer").alias("wallet")),
            ]
        )
        .drop_nulls()
        .unique()
    )
    wallets = signers["wallet"].to_list()
    print(f"Unique signers to keep: {len(wallets):,}")

    out = paths["processed"] / "activity_signers_filtered.parquet"
    cols = [
        "wallet",
        "timestamp",
        "event_type",
        "tx_hash",
        "launchpad",
        "token_address",
        "from_address",
        "to_address",
    ]
    schema_names = pl.scan_parquet(raw).collect_schema().names()
    use = [c for c in cols if c in schema_names]
    print(f"Scanning {raw} ({format_bytes(raw.stat().st_size)}) cols={use}")
    (
        pl.scan_parquet(raw)
        .select(use)
        .filter(pl.col("wallet").is_in(wallets))
        .sink_parquet(out)
    )
    print(f"Wrote {out} ({format_bytes(out.stat().st_size)})")

    # Also a compact bought-activity slice for the same signers (already local)
    if bought.is_file():
        out_b = paths["processed"] / "bought_activity_signers_filtered.parquet"
        bnames = pl.scan_parquet(bought).collect_schema().names()
        buse = [c for c in cols if c in bnames]
        (
            pl.scan_parquet(bought)
            .select(buse)
            .filter(pl.col("wallet").is_in(wallets))
            .sink_parquet(out_b)
        )
        print(f"Wrote {out_b} ({format_bytes(out_b.stat().st_size)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
