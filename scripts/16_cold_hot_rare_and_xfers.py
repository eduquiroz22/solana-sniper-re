#!/usr/bin/env python3
"""Rare shared tokens + distinct-wallet same-tx (proxy for hot↔cold sends)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl

from src.common import ensure_dirs, load_config, write_json  # noqa: E402

RARE_K = (2, 5, 10, 25, 50, 100)
ACT_COLS = ["wallet", "timestamp", "tx_hash", "token_address", "event_type"]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _rate(df: pl.DataFrame, mask: pl.Expr, lab: int) -> float | None:
    sub = df.filter(pl.col("label") == lab)
    if sub.height == 0:
        return None
    return float(sub.select(mask.cast(pl.Float64).mean()).item())


def _pos_rate(df: pl.DataFrame, mask: pl.Expr) -> float | None:
    sub = df.filter(mask)
    if sub.height == 0:
        return None
    return float(sub["label"].mean())


def _summarize_flag(df: pl.DataFrame, flag: str) -> dict:
    return {
        "n": df.height,
        "n_pos": int(df["label"].sum()),
        "n_flag": int(df[flag].sum()),
        "frac_pos": _rate(df, pl.col(flag), 1),
        "frac_neg": _rate(df, pl.col(flag), 0),
        "pos_rate_if_flag": _pos_rate(df, pl.col(flag)),
        "pos_rate_if_not": _pos_rate(df, ~pl.col(flag)),
    }


def main() -> int:
    cfg = load_config()
    paths = ensure_dirs(cfg)
    tmp = paths["processed"] / "_tmp_cold_hot"
    tmp.mkdir(parents=True, exist_ok=True)

    labeled = pl.read_parquet(paths["processed"] / "labeled_features.parquet")
    pos = pl.read_parquet(paths["positives"] / "bought_deploy_txs_index.parquet")
    bought_path = paths["processed"] / "bought_activity_signers_filtered.parquet"
    neg_path = paths["processed"] / "activity_signers_filtered.parquet"
    if not bought_path.is_file() or not neg_path.is_file():
        _log("missing filtered activity")
        return 1

    first_hot = (
        pos.drop_nulls("tx_signer")
        .group_by("tx_signer")
        .agg(pl.col("blockTime").min().alias("became_hot_at"))
    )
    _log(f"signers the sniper ever bought: {first_hot.height:,}")

    cold = labeled.filter(pl.col("prior_bought_same_signer") == 0).select(
        ["token_address", "tx_signer", "blockTime", "label", "split"]
    )
    cold_wallets = (
        cold.select(pl.col("tx_signer").alias("wallet")).drop_nulls().unique()
    )
    _log(
        f"cold deploys={cold.height:,} pos={int(cold['label'].sum()):,} "
        f"signers={cold_wallets.height:,}"
    )

    _log("loading bought activity...")
    bought = pl.read_parquet(bought_path, columns=ACT_COLS).join(
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

    hot_tx_wallets = (
        bought.filter(pl.col("already_hot") & pl.col("tx_hash").is_not_null())
        .select(
            [
                "tx_hash",
                pl.col("wallet").alias("hot_wallet"),
                pl.col("timestamp").alias("hot_ts"),
                pl.col("event_type").alias("hot_event"),
            ]
        )
    )
    hot_tok = (
        bought.filter(pl.col("already_hot") & pl.col("token_address").is_not_null())
        .group_by("token_address")
        .agg(pl.col("timestamp").min().alias("hot_token_at"))
    )
    hot_tx_wallets.write_parquet(tmp / "hot_tx_wallets.parquet")
    hot_tok.write_parquet(tmp / "hot_tok.parquet")
    _log(
        f"hot tx-wallet rows={hot_tx_wallets.height:,} "
        f"unique hot txs={hot_tx_wallets['tx_hash'].n_unique():,} "
        f"hot tokens={hot_tok.height:,}"
    )
    del bought

    # --- token popularity within this deployer set ---
    _log("token popularity (unique wallets per token)...")
    tw_path = tmp / "token_wallet_pairs.parquet"
    (
        pl.concat(
            [
                pl.scan_parquet(bought_path).select(["token_address", "wallet"]),
                pl.scan_parquet(neg_path).select(["token_address", "wallet"]),
            ]
        )
        .filter(
            pl.col("token_address").is_not_null()
            & (pl.col("token_address") != "")
            & pl.col("wallet").is_not_null()
        )
        .group_by(["token_address", "wallet"])
        .len()
        .sink_parquet(tw_path)
    )
    pop = (
        pl.scan_parquet(tw_path)
        .group_by("token_address")
        .agg(
            pl.len().alias("n_wallets"),
            pl.col("len").sum().alias("n_events"),
        )
        .collect()
    )
    pop.write_parquet(tmp / "token_pop.parquet")
    qs = [0.5, 0.75, 0.9, 0.95, 0.99]
    qvals = {f"p{int(q*100)}": float(pop["n_wallets"].quantile(q)) for q in qs}
    _log(
        f"tokens={pop.height:,} n_wallets median={qvals['p50']:.0f} "
        f"p90={qvals['p90']:.0f} p99={qvals['p99']:.0f} "
        f"max={int(pop['n_wallets'].max()):,}"
    )
    k_counts = {
        f"le_{k}": int((pop["n_wallets"] <= k).sum()) for k in RARE_K
    }
    _log(f"rare token counts {k_counts}")

    # cold touches of then-hot tokens, with popularity
    _log("cold×hot token events (streaming)...")
    tok_ev_paths = []
    for tag, src in (("bought", bought_path), ("neg", neg_path)):
        dest = tmp / f"cold_hot_tok_{tag}.parquet"
        (
            pl.scan_parquet(src)
            .select(["wallet", "timestamp", "token_address"])
            .join(cold_wallets.lazy(), on="wallet", how="semi")
            .join(hot_tok.lazy(), on="token_address", how="inner")
            .join(pop.lazy(), on="token_address", how="left")
            .with_columns(
                pl.max_horizontal("timestamp", "hot_token_at").alias("link_at")
            )
            .select(["wallet", "token_address", "link_at", "n_wallets"])
            .sink_parquet(dest)
        )
        tok_ev_paths.append(dest)
        _log(f"  wrote {dest.name}")

    tok_ev = pl.concat([pl.scan_parquet(p) for p in tok_ev_paths])

    def first_rare(k: int) -> pl.DataFrame:
        return (
            tok_ev.filter(pl.col("n_wallets") <= k)
            .group_by("wallet")
            .agg(pl.col("link_at").min().alias("first_rare_at"))
            .collect()
        )

    # --- distinct-wallet same tx (funding / send proxy) ---
    _log("distinct-wallet same-tx hits...")
    hot_hashes = hot_tx_wallets.select("tx_hash").unique()
    xfer_parts = []
    for tag, src in (("bought", bought_path), ("neg", neg_path)):
        dest = tmp / f"cold_same_tx_{tag}.parquet"
        (
            pl.scan_parquet(src)
            .select(["wallet", "timestamp", "tx_hash", "event_type"])
            .join(cold_wallets.lazy(), on="wallet", how="semi")
            .join(hot_hashes.lazy(), on="tx_hash", how="semi")
            .sink_parquet(dest)
        )
        xfer_parts.append(pl.read_parquet(dest))
        _log(f"  {tag} candidate rows={xfer_parts[-1].height:,}")

    cold_hits = pl.concat(xfer_parts)
    pairs = (
        cold_hits.join(hot_tx_wallets, on="tx_hash", how="inner")
        .filter(pl.col("wallet") != pl.col("hot_wallet"))
        .with_columns(
            pl.max_horizontal("timestamp", "hot_ts").alias("pair_at")
        )
    )
    _log(f"distinct-wallet co-tx rows={pairs.height:,}")
    event_pairs = (
        pairs.group_by(["event_type", "hot_event"])
        .len()
        .sort("len", descending=True)
        .head(20)
        .to_dicts()
    )
    _log(f"event-type pairs: {event_pairs[:8]}")
    first_xfer = pairs.group_by("wallet").agg(
        pl.col("pair_at").min().alias("first_xfer_at"),
        pl.len().alias("n_co_tx_rows"),
        pl.col("hot_wallet").n_unique().alias("n_hot_counterparties"),
    )
    _log(f"cold signers with any distinct co-tx={first_xfer.height:,}")

    out = cold.join(
        first_xfer.rename({"wallet": "tx_signer"}),
        on="tx_signer",
        how="left",
    ).with_columns(
        (
            pl.col("first_xfer_at").is_not_null()
            & (pl.col("first_xfer_at") < pl.col("blockTime"))
        ).alias("co_tx_distinct")
    )

    rare_by_k: dict[str, dict] = {}
    for k in RARE_K:
        _log(f"aggregating rare tokens n_wallets<={k}...")
        fr = first_rare(k)
        tagged = out.join(
            fr.rename({"wallet": "tx_signer"}),
            on="tx_signer",
            how="left",
        ).with_columns(
            (
                pl.col("first_rare_at").is_not_null()
                & (pl.col("first_rare_at") < pl.col("blockTime"))
            ).alias("shared_rare")
        )
        rare_by_k[str(k)] = {
            "all": _summarize_flag(tagged, "shared_rare"),
            "train": _summarize_flag(
                tagged.filter(pl.col("split") == "train"), "shared_rare"
            ),
            "valid": _summarize_flag(
                tagged.filter(pl.col("split") == "valid"), "shared_rare"
            ),
            "test": _summarize_flag(
                tagged.filter(pl.col("split") == "test"), "shared_rare"
            ),
        }
        t = rare_by_k[str(k)]["test"]
        _log(
            f"  k={k} test frac_pos={t['frac_pos']} frac_neg={t['frac_neg']} "
            f"pos_rate_yes={t['pos_rate_if_flag']} no={t['pos_rate_if_not']}"
        )

    xfer_s = {
        "all": _summarize_flag(out, "co_tx_distinct"),
        "train": _summarize_flag(out.filter(pl.col("split") == "train"), "co_tx_distinct"),
        "valid": _summarize_flag(out.filter(pl.col("split") == "valid"), "co_tx_distinct"),
        "test": _summarize_flag(out.filter(pl.col("split") == "test"), "co_tx_distinct"),
    }
    _log(f"CO-TX ALL {xfer_s['all']}")
    _log(f"CO-TX TEST {xfer_s['test']}")

    dest = paths["metadata"] / "cold_hot_rare_and_xfers.json"
    write_json(
        dest,
        {
            "note": (
                "from/to are empty and transfer_in/out are ~17 rows, so a send "
                "hot→cold or cold→hot is proxied by distinct wallets in the same "
                "tx_hash, with the hot wallet already bought before that tx. "
                "Rare token = shared token_address whose n_unique wallets in this "
                "deployer activity set is <= K (global count; conservative)."
            ),
            "token_popularity": {
                "n_tokens": pop.height,
                "n_wallets_quantiles": qvals,
                "n_wallets_max": int(pop["n_wallets"].max()),
                "n_tokens_le_k": k_counts,
            },
            "co_tx_distinct_wallets": xfer_s,
            "co_tx_event_type_pairs": event_pairs,
            "shared_rare_token_by_max_wallets": rare_by_k,
        },
    )
    _log(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
