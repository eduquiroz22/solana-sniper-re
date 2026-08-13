#!/usr/bin/env python3
"""Variantes de entrenamiento + calibración. Se elige en VALID, se mira TEST al final.

No toca el examen para elegir. Objetivo: mejor PR-AUC/F1 en valid
(lo que puntúa Kaggle), scores menos inflados.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
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


def metrics_at(y, s, thr: float) -> dict:
    pred = (s >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    order = np.argsort(-s)
    return {
        "threshold": float(thr),
        "n_selected": int(pred.sum()),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "precision_at_100": float(y[order][:100].mean()) if len(y) >= 100 else None,
        "tp": int(((pred == 1) & (y == 1)).sum()),
        "fp": int(((pred == 1) & (y == 0)).sum()),
        "fn": int(((pred == 0) & (y == 1)).sum()),
        "tn": int(((pred == 0) & (y == 0)).sum()),
    }


def best_f1_thr(y, s) -> tuple[float, float]:
    best_thr, best = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y, (s >= thr).astype(int), zero_division=0)
        if f1 > best:
            best, best_thr = f1, float(thr)
    return best_thr, float(best)


def calib_bins(y, s) -> list[dict]:
    out = []
    for a, b in [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.01)]:
        m = (s >= a) & (s < b)
        if not m.any():
            continue
        out.append(
            {
                "bin": f"[{a}, {b})",
                "n": int(m.sum()),
                "mean_score": float(s[m].mean()),
                "frac_real": float(y[m].mean()),
            }
        )
    return out


def fit_hgb(x, y, weight: str):
    if weight == "none":
        w = None
    elif weight == "full":
        w = np.where(y == 1, (len(y) - y.sum()) / max(int(y.sum()), 1), 1.0)
    elif weight == "sqrt":
        ratio = (len(y) - y.sum()) / max(int(y.sum()), 1)
        w = np.where(y == 1, np.sqrt(ratio), 1.0)
    else:
        raise ValueError(weight)
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.07,
        max_iter=350,
        l2_regularization=0.12,
        min_samples_leaf=40,
        random_state=42,
    )
    if w is None:
        clf.fit(x, y)
    else:
        clf.fit(x, y, sample_weight=w)
    return clf


def pack(name, y, s, thr) -> dict:
    rec = {
        "name": name,
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        **metrics_at(y, s, thr),
        "calibration": calib_bins(y, s),
    }
    return rec


def main() -> int:
    cfg = load_config()
    paths = ensure_dirs(cfg)
    df = pl.read_parquet(paths["processed"] / "scored_deploys.parquet")
    use = [c for c in FEATS if c in df.columns]
    tr = df.filter(pl.col("split") == "train")
    va = df.filter(pl.col("split") == "valid")
    te = df.filter(pl.col("split") == "test")
    x_tr, y_tr = tr.select(use).to_pandas(), tr["label"].to_numpy()
    x_va, y_va = va.select(use).to_pandas(), va["label"].to_numpy()
    x_te, y_te = te.select(use).to_pandas(), te["label"].to_numpy()

    variants = []
    best_valid_pr = -1.0
    winner = None
    winner_scores = None

    for weight in ("full", "sqrt", "none"):
        _log(f"=== HGB weight={weight} ===")
        clf = fit_hgb(x_tr, y_tr, weight)
        raw_va = clf.predict_proba(x_va)[:, 1]
        raw_te = clf.predict_proba(x_te)[:, 1]

        # isotonic: fit on valid scores vs labels, apply to test (and valid for display)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(raw_va, y_va)
        cal_va = iso.predict(raw_va)
        cal_te = iso.predict(raw_te)

        for tag, s_va, s_te in (
            (f"hgb_{weight}", raw_va, raw_te),
            (f"hgb_{weight}_isotonic", cal_va, cal_te),
        ):
            thr, _ = best_f1_thr(y_va, s_va)
            vrec = pack(tag + "/valid", y_va, s_va, thr)
            trec = pack(tag + "/test", y_te, s_te, thr)
            _log(
                f"  {tag} valid PR={vrec['pr_auc']:.3f} F1={vrec['f1']:.3f} thr={thr:.2f} "
                f"| test PR={trec['pr_auc']:.3f} F1={trec['f1']:.3f} P={trec['precision']:.3f} "
                f"R={trec['recall']:.3f} P@100={trec['precision_at_100']}"
            )
            variants.append({"valid": vrec, "test": trec})
            # choose on VALID pr_auc, tie-break F1
            score = (vrec["pr_auc"], vrec["f1"])
            if score[0] > best_valid_pr or (
                abs(score[0] - best_valid_pr) < 1e-6 and vrec["f1"] > (winner["valid"]["f1"] if winner else -1)
            ):
                best_valid_pr = vrec["pr_auc"]
                winner = {"name": tag, "threshold": thr, "valid": vrec, "test": trec}
                winner_scores = (clf, iso if "isotonic" in tag else None, s_te, tag)

    # also sklearn CalibratedClassifierCV (sigmoid) on a clone — uses cv on train
    _log("=== CalibratedClassifierCV sigmoid (cv=3, no extra weight) ===")
    base = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.07,
        max_iter=250,
        l2_regularization=0.12,
        min_samples_leaf=40,
        random_state=42,
    )
    cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    cal.fit(x_tr, y_tr)
    s_va = cal.predict_proba(x_va)[:, 1]
    s_te = cal.predict_proba(x_te)[:, 1]
    thr, _ = best_f1_thr(y_va, s_va)
    vrec = pack("sigmoid_cv/valid", y_va, s_va, thr)
    trec = pack("sigmoid_cv/test", y_te, s_te, thr)
    _log(
        f"  sigmoid_cv valid PR={vrec['pr_auc']:.3f} F1={vrec['f1']:.3f} "
        f"| test PR={trec['pr_auc']:.3f} F1={trec['f1']:.3f} P={trec['precision']:.3f} R={trec['recall']:.3f}"
    )
    variants.append({"valid": vrec, "test": trec})
    if vrec["pr_auc"] > best_valid_pr:
        best_valid_pr = vrec["pr_auc"]
        winner = {"name": "sigmoid_cv", "threshold": thr, "valid": vrec, "test": trec}
        winner_scores = (cal, None, s_te, "sigmoid_cv")

    _log(f"\nWINNER (según valid PR-AUC): {winner['name']}")
    dest = paths["metadata"] / "win_calibrate.json"
    write_json(
        dest,
        {
            "note": (
                "Elegido en validación. Test no se usó para elegir. "
                "Isotónica se ajusta en valid y se aplica a test (sin re-mirar labels de test)."
            ),
            "winner": winner,
            "variants": variants,
        },
    )
    _log(f"Wrote {dest}")

    # persist winning test scores alongside old score for the sample csv
    tag = winner["name"]
    te2 = te.with_columns(pl.Series("score_win", winner_scores[2]))
    te2.select(
        [
            c
            for c in te2.columns
            if c
            in {
                "token_address",
                "tx_signer",
                "blockTime",
                "label",
                "score",
                "score_win",
                "prior_bought_same_signer",
            }
        ]
    ).write_parquet(paths["processed"] / "test_scores_win.parquet")
    _log(f"winner tag={tag} test PR={winner['test']['pr_auc']:.3f} F1={winner['test']['f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
