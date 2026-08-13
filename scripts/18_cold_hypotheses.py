#!/usr/bin/env python3
"""Cold-start hypothesis battery + mempool/queue timing + cold-only model."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import polars as pl

from src.common import ensure_dirs, load_config, write_json  # noqa: E402

N_CHUNKS = 8


def _log(msg: str) -> None:
    print(msg, flush=True)


def parse_wallet_buy_index(paths: dict[str, Path]) -> pl.DataFrame:
    src = paths["wallet"] / "5brv79e_activity_txs.jsonl.gz"
    rows = []
    with gzip.open(src, "rt", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            sigs = (o.get("transaction") or {}).get("signatures") or []
            sig = sigs[0] if sigs else None
            rows.append(
                {
                    "buy_tx": sig,
                    "buy_slot": o.get("slot"),
                    "buy_tx_index": o.get("transactionIndex"),
                    "buy_blockTime": o.get("blockTime"),
                }
            )
            if i % 20000 == 0:
                _log(f"  wallet jsonl {i:,}")
    df = pl.DataFrame(rows).drop_nulls("buy_tx").unique(subset=["buy_tx"])
    dest = paths["processed"] / "sniper_buy_tx_index.parquet"
    df.write_parquet(dest)
    _log(f"sniper buy txs indexed={df.height:,}")
    return df


def queue_analysis(paths: dict[str, Path], buy_idx: pl.DataFrame) -> dict:
    pos = pl.read_parquet(paths["positives"] / "bought_deploy_txs_index.parquet")
    labeled = pl.read_parquet(paths["processed"] / "labeled_features.parquet").select(
        ["token_address", "prior_bought_same_signer", "split", "label"]
    )
    pos_tx = pl.read_parquet(paths["processed"] / "pos_tx_features.parquet").select(
        ["tx_hash", "tx_index"]
    )
    wallet = pl.read_parquet(paths["wallet"] / "5brv79e_activity.parquet").select(
        ["timestamp", "event_type", "token_address", "tx_hash"]
    )
    first_buy = (
        wallet.filter(pl.col("event_type") == "buy")
        .sort("timestamp")
        .group_by("token_address")
        .agg(
            pl.col("timestamp").first().alias("buy_ts"),
            pl.col("tx_hash").first().alias("buy_tx"),
        )
    )
    j = (
        pos.join(first_buy, on="token_address", how="left")
        .join(pos_tx, on="tx_hash", how="left")
        .join(buy_idx, on="buy_tx", how="left")
        .join(labeled, on="token_address", how="left")
        .with_columns(
            (pl.col("buy_ts") - pl.col("blockTime")).alias("latency_s"),
            (pl.col("buy_slot") - pl.col("blockSlot")).alias("latency_slots"),
            (pl.col("buy_tx_index") - pl.col("tx_index")).alias("d_index"),
            (pl.col("prior_bought_same_signer") == 0).alias("is_cold"),
        )
    )

    def pack(df: pl.DataFrame, name: str) -> dict:
        n = df.height
        same = df.filter(pl.col("latency_slots") == 0)
        ns = same.height
        d = same["d_index"].drop_nulls()
        return {
            "n": n,
            "frac_same_second": float((df["latency_s"] == 0).mean()),
            "frac_same_slot": float((df["latency_slots"] == 0).mean()),
            "frac_next_slot": float((df["latency_slots"] == 1).mean()),
            "frac_le_2_slots": float((df["latency_slots"] <= 2).mean()),
            "n_same_slot_with_index": ns,
            "same_slot_d_index_median": float(d.median()) if d.len() else None,
            "same_slot_d_index_p90": float(d.quantile(0.9)) if d.len() else None,
            "frac_buy_immediately_after_create": float(((d >= 1) & (d <= 3)).mean())
            if d.len()
            else None,
            "frac_buy_later_in_same_block": float((d > 3).mean()) if d.len() else None,
            "frac_buy_before_create_in_block": float((d < 0).mean()) if d.len() else None,
            "frac_d_index_eq_1": float((d == 1).mean()) if d.len() else None,
        }

    out = {
        "note": (
            "No mempool-arrival timestamp exists. Same slot ≈ the create and the "
            "sniper buy landed in the same ~400ms block. d_index = buy tx position "
            "minus create tx position in that block."
        ),
        "all_positives": pack(j, "all"),
        "hot_positives": pack(j.filter(~pl.col("is_cold")), "hot"),
        "cold_positives": pack(j.filter(pl.col("is_cold")), "cold"),
        "cold_test_positives": pack(
            j.filter(pl.col("is_cold") & (pl.col("split") == "test")), "cold_test"
        ),
    }
    _log(
        "queue cold vs hot same_slot="
        f"{out['cold_positives']['frac_same_slot']:.3f} vs "
        f"{out['hot_positives']['frac_same_slot']:.3f}; "
        f"d_index_med cold={out['cold_positives']['same_slot_d_index_median']} "
        f"hot={out['hot_positives']['same_slot_d_index_median']}"
    )
    return out


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

    # Chunked buy/sell/all counts via asof on hash partitions (one scan each file).
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


def _flag_report(df: pl.DataFrame, flag: str) -> dict | None:
    if flag not in df.columns:
        return None
    col = df[flag]
    if col.null_count() == df.height:
        return None
    # binary-ish
    pos = df.filter(pl.col("label") == 1)
    neg = df.filter(pl.col("label") == 0)

    def rate(sub: pl.DataFrame) -> float | None:
        if sub.height == 0:
            return None
        return float(sub[flag].cast(pl.Float64).mean())

    uniq = set(df[flag].drop_nulls().unique().to_list())
    is_bin = df[flag].dtype in (
        pl.Int8,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt32,
        pl.Boolean,
    ) and uniq <= {0, 1, 0.0, 1.0, True, False}
    yes = df.filter(pl.col(flag) == 1) if is_bin else None
    out: dict = {
        "frac_pos": rate(pos),
        "frac_neg": rate(neg),
        "median_pos": float(pos[flag].median()) if pos.height else None,
        "median_neg": float(neg[flag].median()) if neg.height else None,
    }
    if yes is not None and yes.height > 0:
        no = df.filter(pl.col(flag) == 0)
        out["pos_rate_if"] = float(yes["label"].mean())
        out["pos_rate_if_not"] = float(no["label"].mean()) if no.height else None
        if out["pos_rate_if_not"] not in (None, 0):
            out["lift"] = out["pos_rate_if"] / out["pos_rate_if_not"]
        elif out["pos_rate_if"]:
            out["lift"] = None
    return out


def hypothesis_battery(df: pl.DataFrame) -> dict:
    flags = [
        "is_first_launch",
        "wallet_age_lt_1h",
        "wallet_age_lt_1d",
        "burst_3_launches_1h",
        "serial_10_launches",
        "has_buy_same_tx",
        "has_any_tip",
        "has_service_tip",
        "has_jito_tip",
        "payer_ge_5sol",
        "token_is_pump",
        "has_create_v2",
    ]
    cont = [
        "launches_before",
        "launches_last_1h",
        "launches_last_24h",
        "events_before",
        "buys_before",
        "sells_before",
        "age_s",
        "s_since_last_launch",
        "sell_buy_ratio",
        "payer_sol_pre",
        "sol_spent_lamports",
        "cu_price_micro",
        "fee_lamports",
        "tx_index",
        "n_ix",
        "hour_utc",
    ]
    by_split = {}
    for name, sub in (
        ("all", df),
        ("train", df.filter(pl.col("split") == "train")),
        ("valid", df.filter(pl.col("split") == "valid")),
        ("test", df.filter(pl.col("split") == "test")),
    ):
        rec = {"n": sub.height, "n_pos": int(sub["label"].sum()), "flags": {}, "continuous": {}}
        for f in flags:
            r = _flag_report(sub, f)
            if r:
                rec["flags"][f] = r
        for c in cont:
            r = _flag_report(sub, c)
            if r:
                rec["continuous"][c] = {
                    "median_pos": r["median_pos"],
                    "median_neg": r["median_neg"],
                }
        by_split[name] = rec
        _log(
            f"{name} first_launch frac_pos={rec['flags'].get('is_first_launch', {}).get('frac_pos')} "
            f"frac_neg={rec['flags'].get('is_first_launch', {}).get('frac_neg')}"
        )
    return by_split


def train_cold_model(df: pl.DataFrame) -> dict:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score, roc_auc_score

    feats = [
        c
        for c in [
            "hour_utc",
            "dow",
            "token_is_pump",
            "token_len",
            "days_since_bot_start",
            "tx_index",
            "cu",
            "fee_lamports",
            "n_accounts",
            "n_signers",
            "n_ix",
            "n_inner_ix",
            "cu_limit",
            "cu_price_micro",
            "n_pump_ix",
            "n_sol_transfers",
            "tip_lamports",
            "has_jito_tip",
            "has_service_tip",
            "has_any_tip",
            "has_buy_same_tx",
            "payer_sol_pre",
            "sol_spent_lamports",
            "launches_before",
            "launches_last_1h",
            "launches_last_24h",
            "events_before",
            "buys_before",
            "sells_before",
            "age_s",
            "s_since_last_launch",
            "sell_buy_ratio",
            "buy_frac",
            "is_first_launch",
            "wallet_age_lt_1h",
            "burst_3_launches_1h",
            "serial_10_launches",
            "n_launchpads_before",
        ]
        if c in df.columns
    ]

    def xy(split: str):
        sub = df.filter(pl.col("split") == split)
        return sub.select(feats).to_pandas(), sub["label"].to_numpy()

    x_tr, y_tr = xy("train")
    w = np.where(y_tr == 1, (len(y_tr) - y_tr.sum()) / max(int(y_tr.sum()), 1), 1.0)
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        l2_regularization=0.15,
        min_samples_leaf=50,
        random_state=42,
    )
    clf.fit(x_tr, y_tr, sample_weight=w)
    out: dict = {"features": feats, "splits": {}}
    for split in ("valid", "test"):
        x, y = xy(split)
        s = clf.predict_proba(x)[:, 1]
        order = np.argsort(-s)
        p100 = float(y[order][:100].mean()) if len(y) >= 100 else None
        out["splits"][split] = {
            "roc_auc": float(roc_auc_score(y, s)),
            "pr_auc": float(average_precision_score(y, s)),
            "precision_at_100": p100,
            "base_rate": float(y.mean()),
        }
        _log(f"cold-only model {split} {out['splits'][split]}")

    x_te, y_te = xy("test")
    perm = permutation_importance(
        clf, x_te, y_te, n_repeats=4, random_state=42, scoring="roc_auc", n_jobs=2
    )
    imp = sorted(
        zip(feats, perm.importances_mean.tolist()),
        key=lambda z: -z[1],
    )[:15]
    out["permutation_importance_test_roc"] = [{"feature": a, "delta_roc": b} for a, b in imp]
    _log(f"top importance {imp[:8]}")
    return out


def main() -> int:
    cfg = load_config()
    paths = ensure_dirs(cfg)

    dest_tbl = paths["processed"] / "cold_hypothesis_table.parquet"
    buy_idx_path = paths["processed"] / "sniper_buy_tx_index.parquet"

    if dest_tbl.is_file() and buy_idx_path.is_file() and "--rebuild" not in sys.argv:
        _log("=== using cached tables ===")
        buy_idx = pl.read_parquet(buy_idx_path)
        queue = queue_analysis(paths, buy_idx)
        cold = pl.read_parquet(dest_tbl)
    else:
        _log("=== 1. sniper buy tx index (queue) ===")
        buy_idx = parse_wallet_buy_index(paths)
        queue = queue_analysis(paths, buy_idx)

        _log("=== 2. cold table + tx style ===")
        labeled = pl.read_parquet(paths["processed"] / "labeled_features.parquet")
        cold = labeled.filter(pl.col("prior_bought_same_signer") == 0)
        cold = _attach_tx_style(cold, paths)
        if "payer_sol_pre" in cold.columns:
            cold = cold.with_columns(
                (pl.col("payer_sol_pre") >= 5_000_000_000).cast(pl.Int8).alias("payer_ge_5sol")
            )

        _log("=== 3. deployer history features ===")
        cold = build_activity_features(cold, paths)
        cold.write_parquet(dest_tbl)
        _log(f"wrote {dest_tbl} rows={cold.height:,} cols={len(cold.columns)}")

    _log("=== 4. hypothesis battery ===")
    battery = hypothesis_battery(cold)

    _log("=== 5. cold-only model ===")
    model = train_cold_model(cold)

    dest = paths["metadata"] / "cold_hypotheses.json"
    write_json(
        dest,
        {
            "queue_or_mempool_window": queue,
            "battery_univariate": battery,
            "cold_only_model": model,
            "plain": {
                "queue": (
                    "No hay reloj de 'cuándo entró a la cola'. Solo vemos si la "
                    "compra del sniper cayó en el mismo bloque (~0.4s) y a cuántas "
                    "transacciones de distancia del create."
                ),
                "cold_vs_hot_speed": (
                    "Si los cold fueran más lentos, eso apoyaría 'necesitó tiempo "
                    "para pensar'. Si salen igual de instantáneos, no."
                ),
            },
        },
    )
    _log(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
