#!/usr/bin/env python3
"""Train a leakage-safe baseline and write metrics + a short report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import ensure_dirs, load_config, write_json  # noqa: E402
from src.features.dataset import write_dataset  # noqa: E402

FEATURE_COLS = [
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
]


def _metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    import numpy as np
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        classification_report,
        confusion_matrix,
        precision_recall_curve,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_hat = (y_prob >= threshold).astype(int)
    out: dict = {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "pos_rate": float(y_true.mean()) if len(y_true) else None,
        "threshold": threshold,
    }
    if y_true.min() != y_true.max():
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = None
        out["pr_auc"] = None
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    cm = confusion_matrix(y_true, y_hat, labels=[0, 1])
    out["tn"], out["fp"], out["fn"], out["tp"] = (int(x) for x in cm.ravel())
    prec = out["tp"] / (out["tp"] + out["fp"]) if (out["tp"] + out["fp"]) else 0.0
    rec = out["tp"] / (out["tp"] + out["fn"]) if (out["tp"] + out["fn"]) else 0.0
    out["precision"] = float(prec)
    out["recall"] = float(rec)
    out["f1"] = float(2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    # precision at top-k (k = number of positives in split, or 100)
    k = int(min(100, len(y_prob)))
    order = np.argsort(-y_prob)[:k]
    out["precision_at_100"] = float(y_true[order].mean()) if k else None
    k2 = int(y_true.sum()) or 1
    k2 = min(k2, len(y_prob))
    order2 = np.argsort(-y_prob)[:k2]
    out["precision_at_n_pos"] = float(y_true[order2].mean())
    out["report"] = classification_report(y_true, y_hat, digits=3, zero_division=0)
    prec_c, rec_c, thr = precision_recall_curve(y_true, y_prob)
    # threshold targeting ~20% recall (catch some snipes without flooding)
    target_rec = 0.20
    idx = int(np.argmin(np.abs(rec_c[:-1] - target_rec))) if len(thr) else 0
    if len(thr):
        t2 = float(thr[idx])
        y2 = (y_prob >= t2).astype(int)
        cm2 = confusion_matrix(y_true, y2, labels=[0, 1])
        tn, fp, fn, tp = (int(x) for x in cm2.ravel())
        out["at_recall_20"] = {
            "threshold": t2,
            "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
            "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
            "tp": tp,
            "fp": fp,
        }
    return out


def _write_report(paths, cfg, split_counts, metrics, importances, notes: list[str]) -> Path:
    dest = paths["metadata"] / "PHASE2_REPORT.md"
    m_va = metrics["valid"]
    m_te = metrics["test"]
    imp_lines = "\n".join(
        f"| `{name}` | {gain:.4f} |" for name, gain in importances[:12]
    )
    body = f"""# Phase 2 report — baseline sniper classifier

Leakage-safe tabular model at `t_decision` = deployment `blockTime`.
No post-deploy trades, mcap, or `bought_deployers_activity` (that table is defined by the label).

## Data

| Split | Rows | Positives | Pos rate |
|-------|------|-----------|----------|
| train | {split_counts['train']['n']} | {split_counts['train']['n_pos']} | {split_counts['train']['pos_rate']:.3%} |
| valid | {split_counts['valid']['n']} | {split_counts['valid']['n_pos']} | {split_counts['valid']['pos_rate']:.3%} |
| test  | {split_counts['test']['n']} | {split_counts['test']['n_pos']} | {split_counts['test']['pos_rate']:.3%} |

Cuts: train < `{cfg['temporal_split']['train_end']}` ≤ valid < `{cfg['temporal_split']['valid_end']}` ≤ test.
Positives: full `bought_deploy_txs_index` (15,927). Negatives: `negative_200k.parquet` (~197k after overlap drop).

## Test results (held-out 2026-06-12 → 2026-06-30)

| Metric | Valid | Test |
|--------|-------|------|
| ROC-AUC | {m_va.get('roc_auc')} | {m_te.get('roc_auc')} |
| PR-AUC (vs pos rate) | {m_va.get('pr_auc')} (base {m_va.get('pos_rate'):.3f}) | {m_te.get('pr_auc')} (base {m_te.get('pos_rate'):.3f}) |
| Precision@100 | {m_va.get('precision_at_100')} | {m_te.get('precision_at_100')} |
| Precision@#pos | {m_va.get('precision_at_n_pos')} | {m_te.get('precision_at_n_pos')} |
| Precision / Recall @0.5 | {m_va.get('precision'):.3f} / {m_va.get('recall'):.3f} | {m_te.get('precision'):.3f} / {m_te.get('recall'):.3f} |

Test @ ~20% recall: precision={m_te.get('at_recall_20', {}).get('precision')}, "
        f"fp={m_te.get('at_recall_20', {}).get('fp')}, tp={m_te.get('at_recall_20', {}).get('tp')}.

## What the model is using

| Feature | Importance |
|---------|------------|
{imp_lines}

## Caveats

- Negatives are a **sample** of ~5M non-buys, not the full universe.
- `creator_address` is almost always null on positives; identity is `tx_signer`.
- No deployer activity history for non-bought tokens (23 GiB file not downloaded).
- This is a **baseline**, not a competition-winning stack.
{"".join(f"- {n}" + chr(10) for n in notes)}

## Files

- `data/processed/labeled_features.parquet`
- `models/baseline_hgb.joblib`
- `data/metadata/baseline_metrics.json`
"""
    dest.write_text(body, encoding="utf-8")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild feature parquet")
    args = parser.parse_args()

    cfg = load_config()
    paths = ensure_dirs(cfg)
    feat_path = paths["processed"] / "labeled_features.parquet"
    if args.rebuild or not feat_path.is_file():
        print("Building feature table...")
        feat_path = write_dataset(cfg)
    print(f"Features: {feat_path}")

    import numpy as np
    import polars as pl
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance

    df = pl.read_parquet(feat_path)
    print(df.group_by("split", "label").len().sort(["split", "label"]))

    split_counts = {}
    for sp in ("train", "valid", "test"):
        sub = df.filter(pl.col("split") == sp)
        split_counts[sp] = {
            "n": sub.height,
            "n_pos": int(sub.select(pl.col("label").sum()).item()),
            "pos_rate": float(sub.select(pl.col("label").mean()).item() or 0),
        }

    use_cols = [c for c in FEATURE_COLS if c in df.columns]

    def xy(split: str):
        sub = df.filter(pl.col("split") == split)
        x = sub.select(use_cols).to_pandas()
        y = sub.select("label").to_pandas()["label"].to_numpy()
        return x, y

    x_tr, y_tr = xy("train")
    x_va, y_va = xy("valid")
    x_te, y_te = xy("test")
    print(f"train={len(y_tr)} valid={len(y_va)} test={len(y_te)} features={use_cols}")

    # Imbalance: weight positives
    n_pos = max(int(y_tr.sum()), 1)
    n_neg = max(int((1 - y_tr).sum()), 1)
    w_pos = n_neg / n_pos
    sample_w = np.where(y_tr == 1, w_pos, 1.0)

    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=250,
        l2_regularization=0.1,
        min_samples_leaf=40,
        random_state=42,
    )
    print("Fitting HistGradientBoosting...")
    clf.fit(x_tr, y_tr, sample_weight=sample_w)

    metrics = {
        "train": _metrics(y_tr, clf.predict_proba(x_tr)[:, 1]),
        "valid": _metrics(y_va, clf.predict_proba(x_va)[:, 1]),
        "test": _metrics(y_te, clf.predict_proba(x_te)[:, 1]),
    }
    for sp, m in metrics.items():
        print(f"\n=== {sp} ===")
        print(
            f"ROC-AUC={m.get('roc_auc')} PR-AUC={m.get('pr_auc')} "
            f"P@100={m.get('precision_at_100')} P={m.get('precision'):.3f} R={m.get('recall'):.3f}"
        )
        print(m["report"])

    # Permutation importance on a valid subsample (fast)
    rng = np.random.default_rng(42)
    n_perm = min(8000, len(y_va))
    idx = rng.choice(len(y_va), size=n_perm, replace=False)
    perm = permutation_importance(
        clf,
        x_va.iloc[idx],
        y_va[idx],
        n_repeats=5,
        random_state=42,
        scoring="roc_auc",
        n_jobs=6,
    )
    importances = sorted(
        zip(use_cols, perm.importances_mean.tolist()),
        key=lambda t: -t[1],
    )
    print("\nPermutation importance (valid, ROC-AUC):")
    for name, gain in importances:
        print(f"  {name:28s} {gain:.4f}")

    import joblib

    model_path = ROOT / "models" / "baseline_hgb.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": use_cols}, model_path)
    print(f"Wrote {model_path}")

    notes = [
        f"Positive class weight on train: {w_pos:.2f}",
        "HistGradientBoosting, max_depth=6, max_iter=250.",
    ]
    metrics_path = paths["metadata"] / "baseline_metrics.json"
    write_json(
        metrics_path,
        {
            "features": use_cols,
            "split_counts": split_counts,
            "metrics": {
                k: {kk: vv for kk, vv in v.items() if kk != "report"}
                for k, v in metrics.items()
            },
            "importances": [{"feature": n, "roc_auc_drop": g} for n, g in importances],
            "model": str(model_path),
        },
    )
    report = _write_report(paths, cfg, split_counts, metrics, importances, notes)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
