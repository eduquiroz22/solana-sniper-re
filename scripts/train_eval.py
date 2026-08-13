#!/usr/bin/env python3
"""Train the official HGB model, dump metrics, replica P&L on the test split."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.common import ensure_dirs, load_config, write_json  # noqa: E402

FEATS = [
    "hour_utc",
    "dow",
    "token_is_pump",
    "token_len",
    "days_since_bot_start",
    "prior_bought_same_signer",
    "tx_index",
    "cu",
    "fee_lamports",
    "n_accounts",
    "n_ix",
    "n_inner_ix",
    "cu_limit",
    "cu_price_micro",
    "n_pump_ix",
    "tip_lamports",
    "has_service_tip",
    "has_buy_same_tx",
    "payer_sol_pre",
    "sol_spent_lamports",
    "launches_before",
    "launches_last_1h",
    "launches_last_24h",
    "events_before",
    "s_since_last_launch",
    "age_s",
    "is_first_launch",
    "burst_3_launches_1h",
    "serial_10_launches",
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _f(x) -> float | None:
    if x is None:
        return None
    try:
        if x != x:  # nan
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def bot_behavior(paths: dict[str, Path], pos: pl.DataFrame) -> dict:
    w = pl.read_parquet(paths["wallet"] / "5brv79e_activity.parquet")
    w = w.with_columns(
        [
            pl.col("quote_amount").cast(pl.Float64, strict=False).alias("qty"),
            pl.col("cost_usd").cast(pl.Float64, strict=False).alias("usd"),
            pl.col("gas_native").cast(pl.Float64, strict=False).alias("gas"),
            pl.col("tip_fee").cast(pl.Float64, strict=False).alias("tip"),
            pl.col("dex_native").cast(pl.Float64, strict=False).alias("dex"),
        ]
    )
    solish = pl.col("quote_token_symbol").is_in(["SOL", "WSOL"])
    buys = w.filter(pl.col("event_type") == "buy")
    sells = w.filter(pl.col("event_type") == "sell")
    widx = pl.read_parquet(paths["wallet"] / "5brv79e_activity_txs_index.parquet")
    buy_tx = (
        buys.sort("timestamp")
        .group_by("token_address")
        .agg(pl.col("tx_hash").first().alias("buy_tx"), pl.col("timestamp").first().alias("buy_ts"))
    )
    lat = (
        pos.join(buy_tx, on="token_address", how="left")
        .join(
            widx.rename({"tx_hash": "buy_tx", "blockSlot": "buy_slot", "blockTime": "buy_bt"}),
            on="buy_tx",
            how="left",
        )
        .with_columns(
            (pl.col("buy_ts") - pl.col("blockTime")).alias("latency_s"),
            (pl.col("buy_slot") - pl.col("blockSlot")).alias("latency_slots"),
        )
    )

    # SOL/WSOL cashflow per token. Gross = quote in/out. Net = gross − gas − tip − DEX.
    flow = (
        w.filter(solish)
        .with_columns(
            [
                pl.col("gas").fill_null(0),
                pl.col("tip").fill_null(0),
                pl.col("dex").fill_null(0),
            ]
        )
        .with_columns(
            pl.when(pl.col("event_type") == "buy")
            .then(-pl.col("qty"))
            .when(pl.col("event_type") == "sell")
            .then(pl.col("qty"))
            .otherwise(0.0)
            .alias("delta"),
            (pl.col("gas") + pl.col("tip") + pl.col("dex")).alias("fees"),
        )
        .group_by("token_address")
        .agg(
            pl.col("delta").sum().alias("gross_sol"),
            pl.col("fees").sum().alias("fees_sol"),
            pl.col("qty").filter(pl.col("event_type") == "buy").sum().alias("spent_sol"),
            pl.col("qty").filter(pl.col("event_type") == "sell").sum().alias("got_sol"),
            pl.col("timestamp").min().alias("first_ts"),
            pl.col("timestamp").filter(pl.col("event_type") == "buy").min().alias("t_buy"),
            pl.col("timestamp").filter(pl.col("event_type") == "sell").min().alias("t_first_sell"),
            pl.col("timestamp").filter(pl.col("event_type") == "sell").max().alias("t_last_sell"),
            pl.col("timestamp").filter(pl.col("event_type") == "sell").len().alias("n_sells"),
        )
        .with_columns(
            (pl.col("gross_sol") - pl.col("fees_sol")).alias("net_sol"),
            (pl.col("t_first_sell") - pl.col("t_buy")).alias("hold_first_s"),
        )
    )
    complete = flow.filter(pl.col("n_sells") > 0)
    hit_net = complete.filter(pl.col("net_sol") > 0)
    eq = complete.sort("first_ts").with_columns(pl.col("net_sol").cum_sum().alias("equity"))
    drawdown = _f((eq["equity"] - eq["equity"].cum_max()).min()) if complete.height else None
    sol_buys = buys.filter(solish)
    slots = lat["latency_slots"]
    same = float((slots == 0).mean())
    nxt = float((slots == 1).mean())
    out = {
        "n_wallet_rows": w.height,
        "n_buys": buys.height,
        "n_sells": sells.height,
        "n_buy_tokens": buys["token_address"].n_unique(),
        "latency": {
            "n": lat.height,
            "frac_same_second": float((lat["latency_s"] == 0).mean()),
            "frac_same_slot": same,
            "frac_next_slot": nxt,
            "frac_later": float((slots > 1).mean()),
            "median_s": _f(lat["latency_s"].median()),
        },
        "entry_sol_like": {
            "n": sol_buys.height,
            "median": _f(sol_buys["qty"].median()),
            "p90": _f(sol_buys["qty"].quantile(0.9)),
            "mean": _f(sol_buys["qty"].mean()),
        },
        "hold_seconds": {
            "median_to_first_sell": _f(complete["hold_first_s"].median()),
            "p90_to_first_sell": _f(complete["hold_first_s"].quantile(0.9)),
        },
        "pnl_sol_like_complete": {
            "n_positions": complete.height,
            "gross_sol": _f(complete["gross_sol"].sum()),
            "fees_sol": _f(complete["fees_sol"].sum()),
            "net_sol": _f(complete["net_sol"].sum()),
            "hit_rate_gross": float((complete["gross_sol"] > 0).mean()) if complete.height else None,
            "hit_rate": float((complete["net_sol"] > 0).mean()) if complete.height else None,
            "median_sol": _f(complete["net_sol"].median()),
            "mean_win_sol": _f(hit_net["net_sol"].mean()) if hit_net.height else None,
            "mean_loss_sol": _f(complete.filter(pl.col("net_sol") <= 0)["net_sol"].mean()),
            "max_drawdown_sol": drawdown,
            "note": (
                "SOL+WSOL quote only. Gross = buy cash-out + sell cash-in. "
                "Net = gross − gas_native − tip_fee − dex_native. "
                "hit_rate is after fees (the 55.6% number)."
            ),
        },
    }
    flow.write_parquet(paths["processed"] / "bot_token_pnl.parquet")
    flow.select(
        ["token_address", "gross_sol", "fees_sol", "first_ts", "n_sells", "net_sol"]
    ).write_parquet(paths["processed"] / "bot_token_pnl_net.parquet")
    _log(
        f"bot pnl gross={out['pnl_sol_like_complete']['gross_sol']:.1f} "
        f"net={out['pnl_sol_like_complete']['net_sol']:.1f} SOL "
        f"hit_net={out['pnl_sol_like_complete']['hit_rate']:.3f} "
        f"dd={drawdown:.1f} same_slot={same:.3f}"
    )
    return out


def attach_frame(paths: dict[str, Path]) -> pl.DataFrame:
    labeled = pl.read_parquet(paths["processed"] / "labeled_features.parquet")
    extra = [
        "tx_index",
        "cu",
        "fee_lamports",
        "n_accounts",
        "n_ix",
        "n_inner_ix",
        "cu_limit",
        "cu_price_micro",
        "n_pump_ix",
        "tip_lamports",
        "has_service_tip",
        "has_buy_same_tx",
        "payer_sol_pre",
        "payer_sol_post",
    ]
    pos_tx = pl.read_parquet(paths["processed"] / "pos_tx_features.parquet")
    neg_tx = pl.read_parquet(paths["processed"] / "neg_tx_features.parquet")
    cols = ["tx_hash"] + [c for c in extra if c in pos_tx.columns]
    style = pl.concat([pos_tx.select(cols), neg_tx.select(cols)]).unique("tx_hash")
    df = labeled.join(style, on="tx_hash", how="left")
    if "payer_sol_pre" in df.columns:
        df = df.with_columns(
            (pl.col("payer_sol_pre") - pl.col("payer_sol_post")).alias("sol_spent_lamports")
        )
    cold = pl.read_parquet(paths["processed"] / "cold_hypothesis_table.parquet")
    cold_cols = [
        c
        for c in [
            "token_address",
            "launches_before",
            "launches_last_1h",
            "launches_last_24h",
            "events_before",
            "s_since_last_launch",
            "age_s",
            "is_first_launch",
            "burst_3_launches_1h",
            "serial_10_launches",
        ]
        if c in cold.columns
    ]
    # avoid duplicate style cols already on labeled
    take = [c for c in cold_cols if c == "token_address" or c not in df.columns]
    df = df.join(cold.select(take), on="token_address", how="left")
    return df


def metrics_at(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    pred = (s >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    order = np.argsort(-s)
    p100 = float(y[order][:100].mean()) if len(y) >= 100 else None
    return {
        "threshold": float(thr),
        "n_selected": int(pred.sum()),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "precision_at_100": p100,
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
    }


def train_and_eval(df: pl.DataFrame) -> tuple[dict, pl.DataFrame]:
    use = [c for c in FEATS if c in df.columns]

    def xy(split: str):
        sub = df.filter(pl.col("split") == split)
        return sub.select(use).to_pandas(), sub["label"].to_numpy(), sub

    x_tr, y_tr, _ = xy("train")
    # No class weight: a 15× positive weight inflated scores and hurt valid PR/F1.
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.07,
        max_iter=350,
        l2_regularization=0.12,
        min_samples_leaf=40,
        random_state=42,
    )
    clf.fit(x_tr, y_tr)

    # threshold on valid (max F1)
    x_va, y_va, _ = xy("valid")
    s_va = clf.predict_proba(x_va)[:, 1]
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.9, 86):
        f1 = f1_score(y_va, (s_va >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)

    out = {"features": use, "valid_best_f1_threshold": best_thr, "splits": {}}
    scored_parts = []
    for split in ("train", "valid", "test"):
        x, y, sub = xy(split)
        s = clf.predict_proba(x)[:, 1]
        rec = {
            "n": int(len(y)),
            "n_pos": int(y.sum()),
            "prevalence": float(y.mean()),
            "roc_auc": float(roc_auc_score(y, s)),
            "pr_auc": float(average_precision_score(y, s)),
            "at_valid_f1_threshold": metrics_at(y, s, best_thr),
        }
        cold = sub["prior_bought_same_signer"].to_numpy() == 0
        if cold.any() and y[cold].sum() > 0 and (1 - y[cold]).sum() > 0:
            rec["cold_roc"] = float(roc_auc_score(y[cold], s[cold]))
            rec["cold_pr"] = float(average_precision_score(y[cold], s[cold]))
        out["splits"][split] = rec
        _log(f"{split} ROC={rec['roc_auc']:.3f} PR={rec['pr_auc']:.3f} {rec['at_valid_f1_threshold']}")
        scored_parts.append(sub.with_columns(pl.Series("score", s)))

    scored = pl.concat(scored_parts)
    return out, scored


def replica_backtest(test: pl.DataFrame, pnl: pl.DataFrame, thr: float) -> dict:
    t = test.with_columns((pl.col("score") >= thr).alias("replica_buy"))
    bot = t.filter(pl.col("label") == 1)
    rep = t.filter(pl.col("replica_buy"))
    overlap = t.filter(pl.col("replica_buy") & (pl.col("label") == 1))
    cols = [c for c in ["token_address", "gross_sol", "net_sol", "n_sells"] if c in pnl.columns]
    joined = t.join(pnl.select(cols), on="token_address", how="left")
    bot_net = joined.filter(pl.col("label") == 1)["net_sol"]
    cap_net = joined.filter(pl.col("replica_buy") & (pl.col("label") == 1))["net_sol"]
    bot_gross = joined.filter(pl.col("label") == 1)["gross_sol"] if "gross_sol" in joined.columns else bot_net
    cap_gross = (
        joined.filter(pl.col("replica_buy") & (pl.col("label") == 1))["gross_sol"]
        if "gross_sol" in joined.columns
        else cap_net
    )
    fp = joined.filter(pl.col("replica_buy") & (pl.col("label") == 0))
    return {
        "test_n": t.height,
        "bot_buys": bot.height,
        "replica_buys": rep.height,
        "overlap": overlap.height,
        "recall_of_bot": float(overlap.height / bot.height) if bot.height else None,
        "precision_of_replica": float(overlap.height / rep.height) if rep.height else None,
        "bot_realized_sol_on_test_tokens_with_pnl": _f(bot_gross.drop_nulls().sum()),
        "n_bot_with_sol_pnl": int(bot_net.drop_nulls().len()),
        "replica_captured_bot_sol": _f(cap_gross.drop_nulls().sum()),
        "bot_net_sol_test_after_fees": _f(bot_net.drop_nulls().sum()),
        "replica_captured_net_sol_after_fees": _f(cap_net.drop_nulls().sum()),
        "n_replica_fp_no_price": fp.height,
        "note": (
            "P&L only for tokens the sniper actually traded (SOL/WSOL quotes). "
            "Net = quote cashflow − gas − tip − DEX. "
            "False positives have no price path in local data (mcap_candles 2.8GiB not used). "
            "Sampled negatives inflate precision vs the full ~5M universe."
        ),
    }


def ablation_roc(df: pl.DataFrame) -> dict:
    """Same HGB as the official model, no class weight. Test ROC only."""
    groups = {
        "solo_hot": ["prior_bought_same_signer"],
        "solo_fabrica": [
            "launches_before",
            "launches_last_1h",
            "launches_last_24h",
            "events_before",
            "s_since_last_launch",
            "age_s",
            "is_first_launch",
            "burst_3_launches_1h",
            "serial_10_launches",
        ],
        "solo_tx": [
            "token_is_pump",
            "tx_index",
            "cu",
            "fee_lamports",
            "n_accounts",
            "n_ix",
            "n_inner_ix",
            "cu_limit",
            "cu_price_micro",
            "n_pump_ix",
            "tip_lamports",
            "has_service_tip",
            "has_buy_same_tx",
            "payer_sol_pre",
            "sol_spent_lamports",
        ],
        "hot_mas_tx": ["prior_bought_same_signer"],
        "todo": list(FEATS),
    }
    groups["hot_mas_tx"] = groups["solo_hot"] + groups["solo_tx"]

    def xy(split: str, feats: list[str]):
        sub = df.filter(pl.col("split") == split)
        use = [c for c in feats if c in sub.columns]
        return sub.select(use).to_pandas(), sub["label"].to_numpy(), use

    out = {}
    for name, feats in groups.items():
        x_tr, y_tr, use = xy("train", feats)
        clf = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.07,
            max_iter=350,
            l2_regularization=0.12,
            min_samples_leaf=40,
            random_state=42,
        )
        clf.fit(x_tr, y_tr)
        x_te, y_te, _ = xy("test", use)
        s = clf.predict_proba(x_te)[:, 1]
        rec = {
            "n_features": len(use),
            "roc_auc": float(roc_auc_score(y_te, s)),
            "pr_auc": float(average_precision_score(y_te, s)),
        }
        out[name] = rec
        _log(f"ablation {name} ROC={rec['roc_auc']:.3f} PR={rec['pr_auc']:.3f}")
    return out


def june_rule_rates(paths: dict[str, Path]) -> dict:
    labeled = pl.read_parquet(paths["processed"] / "labeled_features.parquet")
    te = labeled.filter(pl.col("split") == "test")
    hot = te.filter(pl.col("prior_bought_same_signer") > 0)
    cold_l = te.filter(pl.col("prior_bought_same_signer") == 0)
    rates = {
        "test_hot_buy_rate": float(hot["label"].mean()) if hot.height else None,
        "test_cold_buy_rate": float(cold_l["label"].mean()) if cold_l.height else None,
        "n_test": te.height,
        "n_test_hot": hot.height,
        "n_test_cold": cold_l.height,
    }
    cold_path = paths["processed"] / "cold_hypothesis_table.parquet"
    if cold_path.is_file():
        cold = pl.read_parquet(cold_path).filter(pl.col("split") == "test")
        if "burst_3_launches_1h" in cold.columns:
            burst = cold.filter(pl.col("burst_3_launches_1h") == 1)
            quiet = cold.filter(pl.col("burst_3_launches_1h") == 0)
            rates["test_cold_burst_buy_rate"] = float(burst["label"].mean()) if burst.height else None
            rates["test_cold_no_burst_buy_rate"] = float(quiet["label"].mean()) if quiet.height else None
        if "s_since_last_launch" in cold.columns:
            rates["test_cold_hours_since_last_launch_med_bought"] = _f(
                cold.filter(pl.col("label") == 1)["s_since_last_launch"].median()
            )
            rates["test_cold_hours_since_last_launch_med_skipped"] = _f(
                cold.filter(pl.col("label") == 0)["s_since_last_launch"].median()
            )
            if rates["test_cold_hours_since_last_launch_med_bought"] is not None:
                rates["test_cold_hours_since_last_launch_med_bought"] /= 3600.0
            if rates["test_cold_hours_since_last_launch_med_skipped"] is not None:
                rates["test_cold_hours_since_last_launch_med_skipped"] /= 3600.0
        if "sol_spent_lamports" in cold.columns:
            rates["test_cold_create_sol_med_bought"] = _f(
                cold.filter(pl.col("label") == 1)["sol_spent_lamports"].median()
            )
            rates["test_cold_create_sol_med_skipped"] = _f(
                cold.filter(pl.col("label") == 0)["sol_spent_lamports"].median()
            )
            if rates["test_cold_create_sol_med_bought"] is not None:
                rates["test_cold_create_sol_med_bought"] /= 1e9
            if rates["test_cold_create_sol_med_skipped"] is not None:
                rates["test_cold_create_sol_med_skipped"] /= 1e9
    return rates


def main() -> int:
    cfg = load_config()
    paths = ensure_dirs(cfg)
    pos = pl.read_parquet(paths["positives"] / "bought_deploy_txs_index.parquet")

    _log("=== Part 1 bot behavior / P&L ===")
    behavior = bot_behavior(paths, pos)

    _log("=== attach features and train ===")
    df = attach_frame(paths)
    model, scored = train_and_eval(df)
    scored.write_parquet(paths["processed"] / "scored_deploys.parquet")

    _log("=== replica vs bot on test ===")
    pnl = pl.read_parquet(paths["processed"] / "bot_token_pnl_net.parquet")
    test = scored.filter(pl.col("split") == "test")
    thr = model["valid_best_f1_threshold"]
    replica = replica_backtest(test, pnl, thr)
    _log(f"replica {replica}")

    _log("=== ablations (no class weight) ===")
    ablations = ablation_roc(df)
    rates = june_rule_rates(paths)

    # also top-100 / top-500 operating points on test
    s = test["score"].to_numpy()
    y = test["label"].to_numpy()
    extra_ops = {}
    for k in (50, 100, 200, 500):
        extra_ops[f"top_{k}"] = metrics_at(y, s, float(np.sort(s)[-k]) if len(s) >= k else 0)

    dest = paths["metadata"] / "kaggle_train_backtest.json"
    write_json(
        dest,
        {
            "kaggle": {
                "competition": "solana-sniper-bot-reverse-engineering",
                "deadline_utc": "2026-08-14 21:00",
                "submission": "writeup <=3000 words + public notebook + public repo",
                "rubric": {
                    "part1_behavior": "0-20",
                    "part2_features_rules": "0-20",
                    "part2_classification": "0-15 (PR-AUC, P/R/F1; time split)",
                    "part3_backtest": "0-20",
                    "part3_vs_bot": "0-15",
                    "reproducibility": "0-10",
                },
                "competitor_notebook": {
                    "url": "https://www.kaggle.com/code/thtennant/solana-sniper-re-sealed-evidence",
                    "repo": "https://github.com/teddytennant/solana-sniper-reverse-engineering",
                    "takeaway": (
                        "Confirms same-slot sniper, prior-buy recency, quiet-since-last-deploy. "
                        "June AP 0.223 on FULL 852k deploys (not a 200k sample). "
                        "Replica +1 slot with 3 seats lost ~5 SOL; same-slot slice was optimistic."
                    ),
                },
            },
            "behavior": behavior,
            "model": model,
            "weighting": "none (15x positive weight inflated scores and hurt valid PR/F1)",
            "june_rule_rates": rates,
            "ablations_test": ablations,
            "test_extra_operating_points": extra_ops,
            "replica_vs_bot_test": replica,
        },
    )
    _log(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
