#!/usr/bin/env python3
"""Ablations, reglas simples, P&L hot/cold, drawdown, calibración."""

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

HOT = ["prior_bought_same_signer"]
TX = [
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
]
FACTORY = [
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
TIME = ["hour_utc", "dow", "token_len", "days_since_bot_start"]
ALL = TIME + HOT + TX + FACTORY


def _log(msg: str) -> None:
    print(msg, flush=True)


def _f(x):
    try:
        if x is None or x != x:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def metrics_at(y, s, thr: float) -> dict:
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
        "roc_auc": float(roc_auc_score(y, s)) if y.min() != y.max() else None,
        "pr_auc": float(average_precision_score(y, s)) if y.min() != y.max() else None,
    }


def fit_hgb(df: pl.DataFrame, feats: list[str]) -> dict:
    use = [c for c in feats if c in df.columns]
    tr = df.filter(pl.col("split") == "train")
    va = df.filter(pl.col("split") == "valid")
    te = df.filter(pl.col("split") == "test")
    x_tr = tr.select(use).to_pandas()
    y_tr = tr["label"].to_numpy()
    w = np.where(y_tr == 1, (len(y_tr) - y_tr.sum()) / max(int(y_tr.sum()), 1), 1.0)
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.07,
        max_iter=200,
        l2_regularization=0.12,
        min_samples_leaf=40,
        random_state=42,
    )
    clf.fit(x_tr, y_tr, sample_weight=w)
    s_va = clf.predict_proba(va.select(use).to_pandas())[:, 1]
    y_va = va["label"].to_numpy()
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.linspace(0.05, 0.9, 86):
        f1 = f1_score(y_va, (s_va >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    s_te = clf.predict_proba(te.select(use).to_pandas())[:, 1]
    y_te = te["label"].to_numpy()
    rec = metrics_at(y_te, s_te, best_thr)
    rec["valid_f1"] = float(best_f1)
    rec["n_features"] = len(use)
    cold = te["prior_bought_same_signer"].to_numpy() == 0
    if cold.any() and y_te[cold].sum() > 0:
        rec["cold_roc"] = float(roc_auc_score(y_te[cold], s_te[cold]))
        rec["cold_pr"] = float(average_precision_score(y_te[cold], s_te[cold]))
    return rec


def rule_score(df: pl.DataFrame, name: str, pred: pl.Expr, score: pl.Expr | None = None) -> dict:
    te = df.filter(pl.col("split") == "test").with_columns(
        pred.fill_null(False).cast(pl.Int8).alias("pred"),
        (score if score is not None else pred.fill_null(False).cast(pl.Float64)).alias("s"),
    )
    y = te["label"].to_numpy()
    s = np.nan_to_num(te["s"].to_numpy(), nan=0.0)
    rec = metrics_at(y, s, 0.5)
    rec["name"] = name
    return rec


def token_pnl(paths) -> pl.DataFrame:
    w = pl.read_parquet(paths["wallet"] / "5brv79e_activity.parquet").with_columns(
        [
            pl.col("quote_amount").cast(pl.Float64, strict=False).alias("qty"),
            pl.col("gas_native").cast(pl.Float64, strict=False).fill_null(0).alias("gas"),
            pl.col("tip_fee").cast(pl.Float64, strict=False).fill_null(0).alias("tip"),
            pl.col("dex_native").cast(pl.Float64, strict=False).fill_null(0).alias("dex"),
        ]
    )
    sol = w.filter(pl.col("quote_token_symbol").is_in(["SOL", "WSOL"]))
    pnl = (
        sol.with_columns(
            pl.when(pl.col("event_type") == "buy")
            .then(-pl.col("qty"))
            .when(pl.col("event_type") == "sell")
            .then(pl.col("qty"))
            .otherwise(0)
            .alias("delta"),
            (pl.col("gas") + pl.col("tip") + pl.col("dex")).alias("fees"),
            pl.col("timestamp").alias("ts"),
        )
        .group_by("token_address")
        .agg(
            pl.col("delta").sum().alias("gross_sol"),
            pl.col("fees").sum().alias("fees_sol"),
            pl.col("ts").min().alias("first_ts"),
            pl.col("qty").filter(pl.col("event_type") == "sell").len().alias("n_sells"),
        )
        .with_columns((pl.col("gross_sol") - pl.col("fees_sol")).alias("net_sol"))
    )
    pnl.write_parquet(paths["processed"] / "bot_token_pnl_net.parquet")
    return pnl


def main() -> int:
    cfg = load_config()
    paths = ensure_dirs(cfg)
    scored = pl.read_parquet(paths["processed"] / "scored_deploys.parquet")
    cold = pl.read_parquet(paths["processed"] / "cold_hypothesis_table.parquet")
    extra = [
        c
        for c in FACTORY + ["sol_spent_lamports", "has_buy_same_tx", "has_service_tip", "payer_sol_pre"]
        if c in cold.columns and c not in scored.columns
    ]
    df = scored.join(cold.select(["token_address"] + extra), on="token_address", how="left") if extra else scored

    _log("=== ablaciones HGB ===")
    ablations = {
        "solo_hot": fit_hgb(df, HOT),
        "hot_mas_tiempo": fit_hgb(df, HOT + TIME),
        "hot_mas_tx": fit_hgb(df, HOT + TX),
        "hot_mas_fabrica": fit_hgb(df, HOT + FACTORY),
        "solo_fabrica": fit_hgb(df, FACTORY),
        "solo_tx": fit_hgb(df, TX),
        "todo": fit_hgb(df, ALL),
    }
    for k, v in ablations.items():
        _log(f"  {k}: ROC={v['roc_auc']:.3f} PR={v['pr_auc']:.3f} F1={v['f1']:.3f} cold_roc={v.get('cold_roc')}")

    _log("=== reglas simples en test ===")
    lam = 1_000_000_000
    rules = [
        rule_score(df, "si_hot_compra", pl.col("prior_bought_same_signer") > 0),
        rule_score(df, "si_no_burst3", pl.col("burst_3_launches_1h") == 0),
        rule_score(df, "si_callado_1h", pl.col("s_since_last_launch") >= 3600),
        rule_score(df, "si_create_ge_05sol", pl.col("sol_spent_lamports") >= 0.5 * lam),
        rule_score(
            df,
            "hot_o_anti_fabrica",
            (pl.col("prior_bought_same_signer") > 0)
            | (
                (pl.col("burst_3_launches_1h") == 0)
                & (pl.col("s_since_last_launch") >= 3600)
                & (pl.col("sol_spent_lamports") >= 0.5 * lam)
            ),
        ),
        rule_score(
            df,
            "anti_fabrica_estricta",
            (pl.col("burst_3_launches_1h") == 0)
            & (pl.col("s_since_last_launch") >= 3600)
            & (pl.col("sol_spent_lamports") >= 0.5 * lam)
            & (pl.col("token_is_pump") == 1),
        ),
    ]
    for r in rules:
        _log(f"  {r['name']}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} n={r['n_selected']}")

    _log("=== P&L bot hot vs cold + drawdown ===")
    pnl = token_pnl(paths)
    lab = pl.read_parquet(paths["processed"] / "labeled_features.parquet").select(
        ["token_address", "prior_bought_same_signer", "split", "label", "blockTime"]
    )
    bot = lab.filter(pl.col("label") == 1).join(pnl, on="token_address", how="left")
    closed = bot.filter(pl.col("n_sells") > 0)

    def pnl_slice(name, sub):
        c = sub.filter(pl.col("n_sells") > 0)
        return {
            "name": name,
            "n": c.height,
            "net_sol": _f(c["net_sol"].sum()),
            "hit_rate": _f((c["net_sol"] > 0).mean()),
            "median_net": _f(c["net_sol"].median()),
            "mean_net": _f(c["net_sol"].mean()),
        }

    pnl_groups = [
        pnl_slice("all_bot", bot),
        pnl_slice("hot", bot.filter(pl.col("prior_bought_same_signer") > 0)),
        pnl_slice("cold", bot.filter(pl.col("prior_bought_same_signer") == 0)),
        pnl_slice("test", bot.filter(pl.col("split") == "test")),
        pnl_slice("test_hot", bot.filter((pl.col("split") == "test") & (pl.col("prior_bought_same_signer") > 0))),
        pnl_slice("test_cold", bot.filter((pl.col("split") == "test") & (pl.col("prior_bought_same_signer") == 0))),
    ]
    for g in pnl_groups:
        _log(f"  {g}")

    eq = closed.sort("first_ts").with_columns(pl.col("net_sol").cum_sum().alias("equity"))
    peak = eq.with_columns(pl.col("equity").cum_max().alias("peak"))
    dd = peak.with_columns((pl.col("equity") - pl.col("peak")).alias("dd"))
    drawdown = {
        "n_closed": closed.height,
        "final_equity_sol": _f(eq["equity"][-1]),
        "max_drawdown_sol": _f(dd["dd"].min()),
        "max_equity_sol": _f(eq["equity"].max()),
        "roi_vs_median_entry": None,
    }
    _log(f"  drawdown {drawdown}")

    _log("=== calibración del score en test ===")
    te = df.filter(pl.col("split") == "test")
    bins = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.01]
    cal = []
    for a, b in zip(bins[:-1], bins[1:]):
        sub = te.filter((pl.col("score") >= a) & (pl.col("score") < b))
        if sub.height == 0:
            continue
        cal.append(
            {
                "bin": f"[{a}, {b})",
                "n": sub.height,
                "mean_score": _f(sub["score"].mean()),
                "frac_real_snipe": _f(sub["label"].mean()),
            }
        )
    for row in cal:
        _log(f"  {row}")

    _log("=== P&L capturado por umbral (solo TP con precio) ===")
    te_pnl = te.join(pnl.select(["token_address", "net_sol", "n_sells"]), on="token_address", how="left")
    captured = []
    for thr in (0.5, 0.75, 0.9, 0.95):
        sub = te_pnl.filter(pl.col("score") >= thr)
        tp = sub.filter(pl.col("label") == 1)
        captured.append(
            {
                "threshold": thr,
                "n_buys": sub.height,
                "n_tp": tp.height,
                "precision": _f((sub["label"] == 1).mean()),
                "captured_net_sol": _f(tp.filter(pl.col("n_sells") > 0)["net_sol"].sum()),
                "n_fp": sub.filter(pl.col("label") == 0).height,
            }
        )
    for row in captured:
        _log(f"  {row}")

    dest = paths["metadata"] / "extra_hypotheses.json"
    write_json(
        dest,
        {
            "ablations_test": ablations,
            "rules_test": rules,
            "pnl_groups": pnl_groups,
            "drawdown": drawdown,
            "calibration_test": cal,
            "captured_by_threshold": captured,
        },
    )
    _log(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
