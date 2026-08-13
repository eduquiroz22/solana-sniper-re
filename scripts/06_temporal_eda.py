#!/usr/bin/env python3
"""Temporal EDA on positives / wallet activity; propose train/valid/test cuts."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import ensure_dirs, load_config, write_json  # noqa: E402


def _to_epoch_series(df, col_candidates: list[str]):
    import polars as pl

    for c in col_candidates:
        if c in df.columns:
            s = df[c]
            # If datetime, convert; if int/float assume epoch seconds (or ms)
            dtype = s.dtype
            if dtype == pl.Datetime or str(dtype).startswith("Datetime"):
                return s.dt.epoch("s").alias("ts")
            # cast
            out = s.cast(pl.Int64, strict=False)
            # heuristic ms
            med = out.drop_nulls().median()
            if med is not None and med > 10_000_000_000:
                out = (out / 1000).cast(pl.Int64)
            return out.alias("ts")
    return None


def _histograms(ts_series):
    import polars as pl

    df = pl.DataFrame({"ts": ts_series}).drop_nulls()
    if df.is_empty():
        return {"by_day": [], "by_week": [], "n": 0}

    df = df.with_columns(
        [
            pl.from_epoch(pl.col("ts"), time_unit="s").alias("dt"),
        ]
    ).with_columns(
        [
            pl.col("dt").dt.strftime("%Y-%m-%d").alias("day"),
            pl.col("dt").dt.strftime("%G-W%V").alias("week"),
        ]
    )
    by_day = (
        df.group_by("day")
        .len()
        .sort("day")
        .rename({"len": "count"})
        .to_dicts()
    )
    by_week = (
        df.group_by("week")
        .len()
        .sort("week")
        .rename({"len": "count"})
        .to_dicts()
    )
    qs = df.select(
        [
            pl.col("ts").min().alias("min"),
            pl.col("ts").quantile(0.5).alias("p50"),
            pl.col("ts").quantile(0.7).alias("p70"),
            pl.col("ts").quantile(0.85).alias("p85"),
            pl.col("ts").max().alias("max"),
            pl.len().alias("n"),
        ]
    ).to_dicts()[0]
    return {"by_day": by_day, "by_week": by_week, "quantiles": qs, "n": qs["n"]}


def _fmt_ts(ts) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        return None


def _propose_cuts(hist: dict, bot_start: str) -> dict:
    """
    Propose temporal cuts without writing them into config.

    Prefer bot activity window starting ~bot_start. Use quantiles of observed
    timestamps when available: train_end≈p70, valid_end≈p85.
    """
    qs = hist.get("quantiles") or {}
    train_end = _fmt_ts(qs.get("p70"))
    valid_end = _fmt_ts(qs.get("p85"))
    t_min = _fmt_ts(qs.get("min"))
    t_max = _fmt_ts(qs.get("max"))

    # Guard: train_end should not precede bot start if bot window is the signal
    proposal = {
        "bot_activity_start": bot_start,
        "observed_min": t_min,
        "observed_max": t_max,
        "train_end": train_end,
        "valid_end": valid_end,
        "rationale": (
            "Quantile-based cuts on available blockTime/timestamp "
            f"(train_end≈p70, valid_end≈p85). Bot activity begins ~{bot_start}; "
            "consider aligning train start to bot_activity_start rather than Jan 1. "
            "NOT written into config.yaml automatically — copy manually after review."
        ),
        "suggested_splits": {
            "train": f"{bot_start} → {train_end}",
            "valid": f"{train_end} → {valid_end}",
            "test": f"{valid_end} → {t_max}",
        },
    }
    return proposal


def _load_frame(path: Path):
    import polars as pl

    return pl.read_parquet(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    _ = args

    cfg = load_config()
    paths = ensure_dirs(cfg)
    bot_start = cfg.get("bot_activity_start", "2026-03-12")
    urls = cfg.get("urls") or {}
    wf = urls.get("wallet_files") or {}

    targets = {
        "bought_deploy_txs_index": {
            "path": paths["positives"] / "bought_deploy_txs_index.parquet",
            "time_cols": ["blockTime", "block_time", "timestamp", "deploy_blockTime"],
        },
        "wallet_activity": {
            "path": paths["wallet"] / wf.get("activity", "5brv79e_activity.parquet"),
            "time_cols": [
                "blockTime",
                "block_time",
                "timestamp",
                "ts",
                "time",
                "datetime",
            ],
        },
    }

    try:
        import polars as pl  # noqa: F401
    except ImportError:
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError:
            print("Need polars or pyarrow. pip install polars pyarrow")
            return 1
        print("polars not installed; attempting limited pyarrow path...")

    report: dict = {"sources": {}, "proposed_cuts": None}
    any_data = False

    for name, meta in targets.items():
        path = meta["path"]
        print(f"\n=== {name} ===")
        if not path.is_file():
            print(f"  missing: {path}")
            report["sources"][name] = {"exists": False, "path": str(path)}
            continue
        any_data = True
        try:
            import polars as pl

            df = pl.scan_parquet(str(path)).collect()
            ts = None
            for c in meta["time_cols"]:
                if c in df.columns:
                    ts = _to_epoch_series(df, [c])
                    used = c
                    break
            else:
                # try any column with time-like name
                candidates = [
                    c
                    for c in df.columns
                    if "time" in c.lower() or c.lower() in ("slot", "blockslot")
                ]
                ts = _to_epoch_series(df, candidates) if candidates else None
                used = candidates[0] if candidates else None

            if ts is None:
                print(f"  columns: {df.columns}")
                print("  no usable time column found")
                report["sources"][name] = {
                    "exists": True,
                    "path": str(path),
                    "columns": list(df.columns),
                    "error": "no_time_column",
                }
                continue

            hist = _histograms(ts)
            print(f"  time_col={used} n={hist.get('n')}")
            qs = hist.get("quantiles") or {}
            print(
                f"  range: {_fmt_ts(qs.get('min'))} → {_fmt_ts(qs.get('max'))} "
                f"p70={_fmt_ts(qs.get('p70'))} p85={_fmt_ts(qs.get('p85'))}"
            )
            print(f"  weeks: {len(hist.get('by_week') or [])} buckets")
            report["sources"][name] = {
                "exists": True,
                "path": str(path),
                "time_col": used,
                "hist": hist,
            }
        except ImportError:
            # pyarrow-only minimal
            import pyarrow.parquet as pq
            import pyarrow.compute as pc

            table = pq.read_table(path)
            print(f"  columns: {table.column_names}")
            report["sources"][name] = {
                "exists": True,
                "path": str(path),
                "columns": list(table.column_names),
                "num_rows": table.num_rows,
                "note": "Install polars for full temporal histograms",
            }
        except Exception as exc:  # noqa: BLE001
            print(f"  error: {exc}")
            report["sources"][name] = {
                "exists": True,
                "path": str(path),
                "error": str(exc),
            }

    if not any_data:
        print(
            "\nNo positives index or wallet activity found. "
            "Download data first (scripts 03/04). Exiting 0."
        )
        write_json(paths["metadata"] / "temporal_eda.json", report)
        return 0

    # Prefer bought index hist for cuts; else wallet
    base_hist = None
    for key in ("bought_deploy_txs_index", "wallet_activity"):
        src = report["sources"].get(key) or {}
        if src.get("hist"):
            base_hist = src["hist"]
            break

    if base_hist:
        proposal = _propose_cuts(base_hist, bot_start)
        report["proposed_cuts"] = proposal
        print("\n=== Proposed temporal cuts (NOT written to config) ===")
        for k, v in (proposal.get("suggested_splits") or {}).items():
            print(f"  {k}: {v}")
        print(f"  train_end={proposal.get('train_end')} valid_end={proposal.get('valid_end')}")
        print(proposal.get("rationale"))

    out = paths["metadata"] / "temporal_eda.json"
    write_json(out, report)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
