#!/usr/bin/env python3
"""Factory / deployer-history features for cold deploys (as-of t_decision)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl

from src.common import ensure_dirs, load_config  # noqa: E402

N_CHUNKS = 8


def _log(msg: str) -> None:
    print(msg, flush=True)


def _attach_tx_style(deploys: pl.DataFrame, paths: dict[str, Path]) -> pl.DataFrame:
    extra = [
        "tx_index",
        "cu",
        "fee_lamports",
        "n_accounts",
        "n_signers",
        "n_ix",
        "n_inner_ix",
        "n_logs",
        "has_pump_program",
        "cu_limit",
        "cu_price_micro",
        "n_pump_ix",
        "n_sol_transfers",
        "max_xfer_lamports",
        "tip_lamports",
        "has_jito_tip",
        "has_service_tip",
        "has_any_tip",
        "has_buy_same_tx",
        "has_create_v2",
        "payer_sol_pre",
        "payer_sol_post",
    ]
    pos_tx = pl.read_parquet(paths["processed"] / "pos_tx_features.parquet")
    neg_tx = pl.read_parquet(paths["processed"] / "neg_tx_features.parquet")
    cols = ["tx_hash"] + [c for c in extra if c in pos_tx.columns and c in neg_tx.columns]
    style = pl.concat([pos_tx.select(cols), neg_tx.select(cols)]).unique(
        subset=["tx_hash"], keep="first"
    )
    out = deploys.join(style, on="tx_hash", how="left")
    if "payer_sol_pre" in out.columns and "payer_sol_post" in out.columns:
        out = out.with_columns(
            (pl.col("payer_sol_pre") - pl.col("payer_sol_post")).alias("sol_spent_lamports")
        )
    return out


def build_activity_features(cold: pl.DataFrame, paths: dict[str, Path]) -> pl.DataFrame:
    tmp = paths["processed"] / "_tmp_cold_hyp"
    tmp.mkdir(parents=True, exist_ok=True)
    bought = paths["processed"] / "bought_activity_signers_filtered.parquet"
    neg = paths["processed"] / "activity_signers_filtered.parquet"
    signers = cold.select(pl.col("tx_signer").alias("wallet")).drop_nulls().unique()
    _log(f"cold signers={signers.height:,}")

    act = pl.concat(
        [
            pl.scan_parquet(bought).select(
                ["wallet", "timestamp", "event_type", "token_address", "launchpad"]
            ),
            pl.scan_parquet(neg).select(
                ["wallet", "timestamp", "event_type", "token_address", "launchpad"]
            ),
        ]
    ).join(signers.lazy(), on="wallet", how="semi")

    _log("first_seen / lifetime launchpad...")
    first_path = tmp / "first_seen.parquet"
    (
        act.group_by("wallet")
        .agg(
            pl.col("timestamp").min().alias("first_seen_at"),
            pl.col("launchpad")
            .filter(pl.col("launchpad").is_not_null() & (pl.col("launchpad") != ""))
            .n_unique()
            .alias("n_launchpads_ever"),
        )
        .sink_parquet(first_path)
    )
    first_seen = pl.read_parquet(first_path)

    _log("launch events (as-of)...")
    launch_path = tmp / "launches.parquet"
    (
        act.filter(pl.col("event_type") == "launch")
        .select(
            [
                pl.col("wallet").alias("tx_signer"),
                pl.col("timestamp").alias("launch_ts"),
                pl.col("token_address").alias("launched_token"),
                pl.col("launchpad"),
            ]
        )
        .sink_parquet(launch_path)
    )
    launches = pl.read_parquet(launch_path)
    _log(f"launch rows={launches.height:,}")

    launched_before = (
        cold.select(["token_address", "tx_signer", "blockTime"])
        .join(launches, on="tx_signer", how="left")
        .filter(
            pl.col("launch_ts").is_null()
            | (
                (pl.col("launch_ts") < pl.col("blockTime"))
                & (pl.col("launched_token") != pl.col("token_address"))
            )
        )
    )
    launch_agg = launched_before.group_by("token_address").agg(
        pl.col("launch_ts").is_not_null().sum().alias("launches_before"),
        pl.col("launch_ts").max().alias("last_launch_at"),
        pl.col("launch_ts").min().alias("first_launch_at"),
        (
            (pl.col("launch_ts") >= (pl.col("blockTime").first() - 3600))
            & pl.col("launch_ts").is_not_null()
        ).sum().alias("launches_last_1h"),
        (
            (pl.col("launch_ts") >= (pl.col("blockTime").first() - 86400))
            & pl.col("launch_ts").is_not_null()
        ).sum().alias("launches_last_24h"),
        pl.col("launchpad")
        .filter(pl.col("launchpad").is_not_null() & (pl.col("launchpad") != ""))
        .n_unique()
        .alias("n_launchpads_before"),
    )

    _log("partitioning activity by wallet hash...")
    for i in range(N_CHUNKS):
        dest = tmp / f"act_chunk_{i}.parquet"
        if dest.is_file() and dest.stat().st_size > 1000:
            continue
        (
            act.filter((pl.col("wallet").hash() % N_CHUNKS) == i)
            .select(["wallet", "timestamp", "event_type"])
            .sink_parquet(dest)
        )
        _log(f"  wrote chunk {i}")

    parts = []
    deploys_min = cold.select(
        ["token_address", "tx_signer", "blockTime"]
    ).rename({"tx_signer": "wallet"})
    for i in range(N_CHUNKS):
        _log(f"asof counts chunk {i}...")
        ev = pl.read_parquet(tmp / f"act_chunk_{i}.parquet").sort(["wallet", "timestamp"])
        ev = ev.with_columns(
            [
                (pl.int_range(pl.len()).over("wallet") + 1).alias("n_all"),
                pl.col("event_type").eq("buy").cum_sum().over("wallet").alias("n_buy"),
                pl.col("event_type").eq("sell").cum_sum().over("wallet").alias("n_sell"),
            ]
        )
        sub = deploys_min.filter(
            (pl.col("wallet").hash() % N_CHUNKS) == i
        ).sort(["wallet", "blockTime"])
        if sub.height == 0:
            continue
        joined = sub.join_asof(
            ev.select(["wallet", "timestamp", "n_all", "n_buy", "n_sell"]),
            left_on="blockTime",
            right_on="timestamp",
            by="wallet",
            strategy="backward",
        ).with_columns(
            pl.when(pl.col("timestamp").is_not_null() & (pl.col("timestamp") < pl.col("blockTime")))
            .then(pl.col("n_all"))
            .when(pl.col("timestamp") == pl.col("blockTime"))
            .then((pl.col("n_all") - 1).clip(lower_bound=0))
            .otherwise(0)
            .alias("events_before"),
            pl.when(pl.col("timestamp").is_not_null() & (pl.col("timestamp") < pl.col("blockTime")))
            .then(pl.col("n_buy"))
            .otherwise(0)
            .alias("buys_before"),
            pl.when(pl.col("timestamp").is_not_null() & (pl.col("timestamp") < pl.col("blockTime")))
            .then(pl.col("n_sell"))
            .otherwise(0)
            .alias("sells_before"),
        )
        parts.append(
            joined.select(
                ["token_address", "events_before", "buys_before", "sells_before"]
            )
        )
        del ev
    counts = pl.concat(parts) if parts else pl.DataFrame(
        schema={
            "token_address": pl.String,
            "events_before": pl.UInt32,
            "buys_before": pl.UInt32,
            "sells_before": pl.UInt32,
        }
    )

    out = (
        cold.join(launch_agg, on="token_address", how="left")
        .join(counts, on="token_address", how="left")
        .join(
            first_seen.rename({"wallet": "tx_signer"}),
            on="tx_signer",
            how="left",
        )
        .with_columns(
            [
                pl.col("launches_before").fill_null(0),
                pl.col("launches_last_1h").fill_null(0),
                pl.col("launches_last_24h").fill_null(0),
                pl.col("events_before").fill_null(0),
                pl.col("buys_before").fill_null(0),
                pl.col("sells_before").fill_null(0),
                pl.col("n_launchpads_before").fill_null(0),
                pl.col("n_launchpads_ever").fill_null(0),
            ]
        )
        .with_columns(
            [
                (pl.col("launches_before") == 0).cast(pl.Int8).alias("is_first_launch"),
                (pl.col("blockTime") - pl.col("first_seen_at")).alias("age_s"),
                (pl.col("blockTime") - pl.col("last_launch_at")).alias("s_since_last_launch"),
                (pl.col("sells_before") / (pl.col("buys_before") + 1.0)).alias("sell_buy_ratio"),
                (pl.col("buys_before") / (pl.col("events_before") + 1.0)).alias("buy_frac"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("age_s") < 0).then(None).otherwise(pl.col("age_s")).alias("age_s"),
                (pl.col("age_s") < 3600).fill_null(False).cast(pl.Int8).alias("wallet_age_lt_1h"),
                (pl.col("age_s") < 86400).fill_null(False).cast(pl.Int8).alias("wallet_age_lt_1d"),
                (pl.col("launches_last_1h") >= 3).cast(pl.Int8).alias("burst_3_launches_1h"),
                (pl.col("launches_before") >= 10).cast(pl.Int8).alias("serial_10_launches"),
                (pl.col("payer_sol_pre") >= 5_000_000_000).cast(pl.Int8).alias("payer_ge_5sol")
                if "payer_sol_pre" in cold.columns
                else pl.lit(0).alias("payer_ge_5sol"),
            ]
        )
    )
    return out


def main() -> int:
    from src.features.dataset import write_dataset

    cfg = load_config()
    paths = ensure_dirs(cfg)

    labeled_path = paths["processed"] / "labeled_features.parquet"
    if not labeled_path.is_file():
        _log("=== labeled_features ===")
        write_dataset(cfg)

    dest_tbl = paths["processed"] / "cold_hypothesis_table.parquet"
    if dest_tbl.is_file() and "--rebuild" not in sys.argv:
        _log(f"using cached {dest_tbl}")
        return 0

    _log("=== cold table + tx style + factory history ===")
    labeled = pl.read_parquet(labeled_path)
    cold = labeled.filter(pl.col("prior_bought_same_signer") == 0)
    cold = _attach_tx_style(cold, paths)
    cold = build_activity_features(cold, paths)
    cold.write_parquet(dest_tbl)
    _log(f"wrote {dest_tbl} rows={cold.height:,} cols={len(cold.columns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
