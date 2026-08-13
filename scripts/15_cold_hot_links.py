#!/usr/bin/env python3
"""Did 'cold' deployers already share txs or tokens with then-hot deployers?"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl

from src.common import ensure_dirs, load_config, write_json  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def _first_ready(events: pl.DataFrame, ts_col: str) -> pl.DataFrame:
    if events.height == 0:
        return pl.DataFrame(
            schema={"wallet": pl.String, ts_col: pl.Int64},
        )
    return events.group_by("wallet").agg(pl.col("timestamp").min().alias(ts_col))


def main() -> int:
    cfg = load_config()
    paths = ensure_dirs(cfg)
    tmp = paths["processed"] / "_tmp_cold_hot"
    tmp.mkdir(parents=True, exist_ok=True)

    labeled = pl.read_parquet(paths["processed"] / "labeled_features.parquet")
    pos = pl.read_parquet(paths["positives"] / "bought_deploy_txs_index.parquet")

    first_hot = (
        pos.drop_nulls("tx_signer")
        .group_by("tx_signer")
        .agg(pl.col("blockTime").min().alias("became_hot_at"))
    )
    _log(f"signers the sniper ever bought: {first_hot.height:,}")

    bought_path = paths["processed"] / "bought_activity_signers_filtered.parquet"
    neg_path = paths["processed"] / "activity_signers_filtered.parquet"
    if not bought_path.is_file() or not neg_path.is_file():
        _log("missing filtered activity")
        return 1

    cols = ["wallet", "timestamp", "tx_hash", "token_address"]
    _log("loading bought activity (hot-signer histories)...")
    bought = pl.read_parquet(bought_path, columns=cols)
    bought = bought.join(
        first_hot.rename({"tx_signer": "wallet"}),
        on="wallet",
        how="left",
    ).with_columns(
        (
            pl.col("became_hot_at").is_not_null()
            & (pl.col("became_hot_at") < pl.col("timestamp"))
        ).alias("already_hot")
    )
    _log(
        f"bought rows={bought.height:,} already_hot={int(bought['already_hot'].sum()):,}"
    )

    hot_tx = (
        bought.filter(pl.col("already_hot") & pl.col("tx_hash").is_not_null())
        .group_by("tx_hash")
        .agg(pl.col("timestamp").min().alias("hot_tx_at"))
    )
    hot_tok = (
        bought.filter(pl.col("already_hot") & pl.col("token_address").is_not_null())
        .group_by("token_address")
        .agg(pl.col("timestamp").min().alias("hot_token_at"))
    )
    hot_tx.write_parquet(tmp / "hot_tx.parquet")
    hot_tok.write_parquet(tmp / "hot_tok.parquet")
    _log(f"hot tx_hashes={hot_tx.height:,} hot tokens={hot_tok.height:,}")
    del bought

    cold = labeled.filter(pl.col("prior_bought_same_signer") == 0).select(
        ["token_address", "tx_signer", "blockTime", "label", "split"]
    )
    cold_wallets = (
        cold.select(pl.col("tx_signer").alias("wallet")).drop_nulls().unique()
    )
    _log(
        f"cold deploys={cold.height:,} positives={int(cold['label'].sum()):,} "
        f"signers={cold_wallets.height:,}"
    )

    # First activity per cold signer (both files, streaming on the large one).
    _log("first activity timestamps for cold signers...")
    first_act_bought = (
        pl.scan_parquet(bought_path)
        .select(["wallet", "timestamp"])
        .join(cold_wallets.lazy(), on="wallet", how="semi")
        .group_by("wallet")
        .agg(pl.col("timestamp").min().alias("first_act_at"))
        .collect()
    )
    first_act_neg_path = tmp / "first_act_neg.parquet"
    (
        pl.scan_parquet(neg_path)
        .select(["wallet", "timestamp"])
        .join(cold_wallets.lazy(), on="wallet", how="semi")
        .group_by("wallet")
        .agg(pl.col("timestamp").min().alias("first_act_at"))
        .sink_parquet(first_act_neg_path)
    )
    first_act = (
        pl.concat([first_act_bought, pl.read_parquet(first_act_neg_path)])
        .group_by("wallet")
        .agg(pl.col("first_act_at").min())
    )
    _log(f"cold signers with any activity={first_act.height:,}")

    # Shared txs: cold-signer rows whose tx_hash was already used by a then-hot signer.
    _log("shared tx_hash hits in bought activity...")
    shared_tx_bought = (
        pl.scan_parquet(bought_path)
        .select(cols)
        .join(cold_wallets.lazy(), on="wallet", how="semi")
        .join(hot_tx.lazy(), on="tx_hash", how="inner")
        .filter(pl.col("hot_tx_at") <= pl.col("timestamp"))
        .select(["wallet", "timestamp"])
        .collect()
    )
    _log(f"bought shared-tx rows={shared_tx_bought.height:,}")

    _log("shared tx_hash hits in negative activity (streaming)...")
    shared_tx_neg_path = tmp / "shared_tx_neg.parquet"
    (
        pl.scan_parquet(neg_path)
        .select(cols)
        .join(cold_wallets.lazy(), on="wallet", how="semi")
        .join(pl.scan_parquet(tmp / "hot_tx.parquet"), on="tx_hash", how="inner")
        .filter(pl.col("hot_tx_at") <= pl.col("timestamp"))
        .select(["wallet", "timestamp"])
        .sink_parquet(shared_tx_neg_path)
    )
    shared_tx_neg = pl.read_parquet(shared_tx_neg_path)
    _log(f"neg shared-tx rows={shared_tx_neg.height:,}")
    first_shared_tx = _first_ready(
        pl.concat([shared_tx_bought, shared_tx_neg]), "first_shared_tx_at"
    )
    del shared_tx_bought, shared_tx_neg

    # Shared tokens: cold signer touched a token a then-hot signer had already touched.
    _log("shared token hits in bought activity...")
    shared_tok_bought = (
        pl.scan_parquet(bought_path)
        .select(cols)
        .join(cold_wallets.lazy(), on="wallet", how="semi")
        .join(hot_tok.lazy(), on="token_address", how="inner")
        .select(["wallet", "timestamp", "hot_token_at"])
        .with_columns(
            pl.max_horizontal("timestamp", "hot_token_at").alias("timestamp")
        )
        .select(["wallet", "timestamp"])
        .collect()
    )
    _log(f"bought shared-token rows={shared_tok_bought.height:,}")

    _log("shared token hits in negative activity (streaming)...")
    shared_tok_neg_path = tmp / "shared_tok_neg.parquet"
    (
        pl.scan_parquet(neg_path)
        .select(cols)
        .join(cold_wallets.lazy(), on="wallet", how="semi")
        .join(pl.scan_parquet(tmp / "hot_tok.parquet"), on="token_address", how="inner")
        .select(["wallet", "timestamp", "hot_token_at"])
        .with_columns(
            pl.max_horizontal("timestamp", "hot_token_at").alias("timestamp")
        )
        .select(["wallet", "timestamp"])
        .sink_parquet(shared_tok_neg_path)
    )
    shared_tok_neg = pl.read_parquet(shared_tok_neg_path)
    _log(f"neg shared-token rows={shared_tok_neg.height:,}")
    first_shared_tok = _first_ready(
        pl.concat([shared_tok_bought, shared_tok_neg]), "first_shared_token_at"
    )
    del shared_tok_bought, shared_tok_neg, hot_tx, hot_tok

    out = (
        cold.join(
            first_act.rename({"wallet": "tx_signer"}),
            on="tx_signer",
            how="left",
        )
        .join(
            first_shared_tx.rename({"wallet": "tx_signer"}),
            on="tx_signer",
            how="left",
        )
        .join(
            first_shared_tok.rename({"wallet": "tx_signer"}),
            on="tx_signer",
            how="left",
        )
        .with_columns(
            (pl.col("first_act_at").is_not_null() & (pl.col("first_act_at") < pl.col("blockTime"))).alias(
                "has_prior_activity"
            ),
            (
                pl.col("first_shared_tx_at").is_not_null()
                & (pl.col("first_shared_tx_at") < pl.col("blockTime"))
            ).alias("shared_tx"),
            (
                pl.col("first_shared_token_at").is_not_null()
                & (pl.col("first_shared_token_at") < pl.col("blockTime"))
            ).alias("shared_token"),
        )
    )

    def summarize(df: pl.DataFrame) -> dict:
        def rate(mask: pl.Expr, lab: int) -> float | None:
            sub = df.filter(pl.col("label") == lab)
            if sub.height == 0:
                return None
            return float(sub.select(mask.cast(pl.Float64).mean()).item())

        def pos_rate(mask: pl.Expr) -> float | None:
            sub = df.filter(mask)
            if sub.height == 0:
                return None
            return float(sub["label"].mean())

        return {
            "n": df.height,
            "n_pos": int(df["label"].sum()),
            "n_neg": df.height - int(df["label"].sum()),
            "frac_pos_with_prior_activity": rate(pl.col("has_prior_activity"), 1),
            "frac_neg_with_prior_activity": rate(pl.col("has_prior_activity"), 0),
            "frac_pos_shared_tx": rate(pl.col("shared_tx"), 1),
            "frac_neg_shared_tx": rate(pl.col("shared_tx"), 0),
            "frac_pos_shared_token": rate(pl.col("shared_token"), 1),
            "frac_neg_shared_token": rate(pl.col("shared_token"), 0),
            "pos_rate_if_shared_token": pos_rate(pl.col("shared_token")),
            "pos_rate_if_not_shared_token": pos_rate(~pl.col("shared_token")),
            "pos_rate_if_shared_tx": pos_rate(pl.col("shared_tx")),
            "pos_rate_if_not_shared_tx": pos_rate(~pl.col("shared_tx")),
            "n_shared_tx": int(df["shared_tx"].sum()),
            "n_shared_token": int(df["shared_token"].sum()),
        }

    all_s = summarize(out)
    test_s = summarize(out.filter(pl.col("split") == "test"))
    _log(f"ALL {all_s}")
    _log(f"TEST {test_s}")

    dest = paths["metadata"] / "cold_hot_links.json"
    write_json(
        dest,
        {
            "note": (
                "from/to empty; shared_tx = same tx_hash as a then-hot signer; "
                "shared_token = same token_address touched by a then-hot signer; "
                "hot = sniper had already bought that signer before the activity row."
            ),
            "all_cold": all_s,
            "test_cold": test_s,
        },
    )
    _log(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
