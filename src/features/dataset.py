"""Build a leakage-safe labeled table at t_decision (deployment blockTime)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from src.common import ensure_dirs, load_config, project_root

BOT_START = datetime(2026, 3, 12, tzinfo=timezone.utc)


def _epoch(iso: str) -> int:
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def _split_bounds(cfg: dict[str, Any]) -> tuple[int, int]:
    ts = cfg.get("temporal_split") or {}
    train_end = _epoch(str(ts.get("train_end") or "2026-05-29T00:00:00Z"))
    valid_end = _epoch(str(ts.get("valid_end") or "2026-06-12T00:00:00Z"))
    return train_end, valid_end


def _load_positives(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    return (
        df.select(
            [
                pl.col("token_address"),
                pl.col("tx_signer"),
                pl.col("creator_address"),
                pl.col("tx_hash"),
                pl.col("blockTime").cast(pl.Int64),
                pl.col("blockSlot").cast(pl.Int64),
            ]
        )
        .drop_nulls(["token_address", "blockTime"])
        .unique(subset=["token_address"], keep="first")
        .with_columns(pl.lit(1).cast(pl.Int8).alias("label"))
    )


def _load_negatives(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    cols = [
        "token_address",
        "tx_signer",
        "creator_address",
        "tx_hash",
        "blockTime",
        "blockSlot",
    ]
    have = [c for c in cols if c in df.columns]
    out = df.select(have)
    for c in cols:
        if c not in out.columns:
            out = out.with_columns(pl.lit(None).alias(c))
    return (
        out.with_columns(
            [
                pl.col("blockTime").cast(pl.Int64),
                pl.col("blockSlot").cast(pl.Int64),
            ]
        )
        .drop_nulls(["token_address", "blockTime"])
        .unique(subset=["token_address"], keep="first")
        .with_columns(pl.lit(0).cast(pl.Int8).alias("label"))
    )


def build_feature_frame(cfg: dict[str, Any] | None = None) -> pl.DataFrame:
    cfg = cfg or load_config()
    paths = ensure_dirs(cfg)
    pos_idx = paths["positives"] / "bought_deploy_txs_index.parquet"
    neg_path = paths["samples"] / "negative_200k.parquet"
    wallet_path = paths["wallet"] / "5brv79e_activity.parquet"

    pos = _load_positives(pos_idx)
    neg = _load_negatives(neg_path)
    # Drop negatives that are also positives (token overlap)
    pos_tokens = pos.select("token_address")
    neg = neg.join(pos_tokens, on="token_address", how="anti")

    deploys = pl.concat([pos, neg], how="diagonal_relaxed")
    train_end, valid_end = _split_bounds(cfg)
    bot_start = int(BOT_START.timestamp())

    deploys = deploys.with_columns(
        [
            pl.from_epoch(pl.col("blockTime"), time_unit="s").alias("_dt"),
            pl.when(pl.col("blockTime") < train_end)
            .then(pl.lit("train"))
            .when(pl.col("blockTime") < valid_end)
            .then(pl.lit("valid"))
            .otherwise(pl.lit("test"))
            .alias("split"),
        ]
    ).with_columns(
        [
            pl.col("_dt").dt.hour().alias("hour_utc"),
            pl.col("_dt").dt.weekday().alias("dow"),  # 1=Mon
            pl.col("_dt").dt.month().alias("month"),
            ((pl.col("blockTime") - bot_start) / 86400.0).alias("days_since_bot_start"),
            pl.col("token_address").str.ends_with("pump").cast(pl.Int8).alias("token_is_pump"),
            pl.col("token_address").str.len_chars().alias("token_len"),
            pl.col("tx_signer").is_not_null().cast(pl.Int8).alias("has_signer"),
            pl.col("creator_address").is_null().cast(pl.Int8).alias("creator_missing"),
            (pl.col("tx_signer") == pl.col("creator_address"))
            .fill_null(False)
            .cast(pl.Int8)
            .alias("signer_eq_creator"),
        ]
    )

    # Prior bought count by signer (positives only as history — valid at t_decision)
    pos_hist = (
        pos.drop_nulls("tx_signer")
        .select(["tx_signer", "blockTime"])
        .sort(["tx_signer", "blockTime"])
        .with_columns(
            (pl.int_range(pl.len()).over("tx_signer") + 1).alias("n_bought_incl")
        )
    )
    deploys = (
        deploys.sort(["tx_signer", "blockTime"])
        .join_asof(
            pos_hist.sort(["tx_signer", "blockTime"]),
            left_on="blockTime",
            right_on="blockTime",
            by="tx_signer",
            strategy="backward",
        )
        .with_columns(
            pl.col("n_bought_incl").fill_null(0).alias("n_bought_incl")
        )
        .with_columns(
            # asof hits the current positive row; strip it so the count is strictly before t
            pl.when(pl.col("label") == 1)
            .then((pl.col("n_bought_incl") - 1).clip(lower_bound=0))
            .otherwise(pl.col("n_bought_incl"))
            .alias("prior_bought_same_signer")
        )
        .drop("n_bought_incl")
    )

    if wallet_path.is_file():
        w = pl.read_parquet(wallet_path).select(
            [
                pl.col("timestamp").cast(pl.Int64).alias("w_ts"),
                pl.col("event_type"),
                pl.col("token_address").alias("w_token"),
                pl.col("from_address"),
                pl.col("to_address"),
            ]
        )
        # Global bot activity level just before this deploy
        w_sorted = w.sort("w_ts").with_columns(pl.int_range(pl.len()).alias("w_n"))
        deploys = deploys.sort("blockTime").join_asof(
            w_sorted.select(["w_ts", "w_n"]),
            left_on="blockTime",
            right_on="w_ts",
            strategy="backward",
        ).with_columns(pl.col("w_n").fill_null(0).alias("wallet_events_before"))

        # Prior wallet interaction with this signer (from/to)
        counterpart = pl.concat(
            [
                w.select(pl.col("from_address").alias("tx_signer"), pl.col("w_ts")),
                w.select(pl.col("to_address").alias("tx_signer"), pl.col("w_ts")),
            ]
        ).filter(
            pl.col("tx_signer").is_not_null() & (pl.col("tx_signer") != "")
        ).group_by("tx_signer").agg(pl.col("w_ts").alias("w_hits"))
        # explode later is heavy; instead count via join + filter
        hits = (
            deploys.select(["token_address", "tx_signer", "blockTime"])
            .drop_nulls("tx_signer")
            .join(
                pl.concat(
                    [
                        w.select(pl.col("from_address").alias("tx_signer"), "w_ts"),
                        w.select(pl.col("to_address").alias("tx_signer"), "w_ts"),
                    ]
                ).filter(pl.col("tx_signer").is_not_null() & (pl.col("tx_signer") != "")),
                on="tx_signer",
                how="left",
            )
            .filter(pl.col("w_ts").is_not_null() & (pl.col("w_ts") < pl.col("blockTime")))
            .group_by("token_address")
            .len()
            .rename({"len": "wallet_hits_signer_before"})
        )
        deploys = deploys.join(hits, on="token_address", how="left").with_columns(
            pl.col("wallet_hits_signer_before").fill_null(0)
        )
        del counterpart
    else:
        deploys = deploys.with_columns(
            [
                pl.lit(0).alias("wallet_events_before"),
                pl.lit(0).alias("wallet_hits_signer_before"),
            ]
        )

    feature_cols = [
        "hour_utc",
        "dow",
        "month",
        "days_since_bot_start",
        "token_is_pump",
        "token_len",
        "has_signer",
        "creator_missing",
        "signer_eq_creator",
        "prior_bought_same_signer",
        "wallet_events_before",
        "wallet_hits_signer_before",
        "blockSlot",
    ]
    keep = [
        "token_address",
        "tx_signer",
        "creator_address",
        "tx_hash",
        "blockTime",
        "label",
        "split",
        *feature_cols,
    ]
    out = deploys.select([c for c in keep if c in deploys.columns])
    return out


def write_dataset(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    paths = ensure_dirs(cfg)
    df = build_feature_frame(cfg)
    dest = paths["processed"] / "labeled_features.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dest)
    return dest


if __name__ == "__main__":
    p = write_dataset()
    print(f"Wrote {p}")
