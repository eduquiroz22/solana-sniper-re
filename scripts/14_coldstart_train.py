#!/usr/bin/env python3
"""Add pre-t_decision deployer activity features and retrain. CPU only."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import ensure_dirs, load_config, write_json  # noqa: E402


def main() -> int:
    import numpy as np
    import polars as pl
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    cfg = load_config()
    paths = ensure_dirs(cfg)
    labeled = pl.read_parquet(paths["processed"] / "labeled_features.parquet")
    parts = []
    for name in (
        "activity_signers_filtered.parquet",
        "bought_activity_signers_filtered.parquet",
    ):
        p = paths["processed"] / name
        if p.is_file():
            parts.append(pl.scan_parquet(p).select(["wallet", "timestamp", "event_type"]))
    if not parts:
        print("No filtered activity yet")
        return 1

    act = pl.concat(parts).unique().sort(["wallet", "timestamp"]).collect()
    act = act.with_columns(
        (pl.int_range(pl.len()).over("wallet") + 1).alias("n_evt_incl")
    )
    print(f"activity rows={act.height:,}")

    deploys = labeled.sort(["tx_signer", "blockTime"]).join_asof(
        act.select(
            [
                pl.col("wallet").alias("tx_signer"),
                pl.col("timestamp"),
                pl.col("n_evt_incl"),
            ]
        ).sort(["tx_signer", "timestamp"]),
        left_on="blockTime",
        right_on="timestamp",
        by="tx_signer",
        strategy="backward",
    ).with_columns(
        pl.when(pl.col("timestamp").is_not_null() & (pl.col("timestamp") < pl.col("blockTime")))
        .then(pl.col("n_evt_incl"))
        .when(pl.col("timestamp") == pl.col("blockTime"))
        .then((pl.col("n_evt_incl") - 1).clip(lower_bound=0))
        .otherwise(0)
        .alias("deployer_events_before")
    )

    feat_cols = [
        c
        for c in [
            "hour_utc",
            "dow",
            "token_is_pump",
            "prior_bought_same_signer",
            "deployer_events_before",
            "days_since_bot_start",
        ]
        if c in deploys.columns
    ]

    def xy(split: str):
        sub = deploys.filter(pl.col("split") == split)
        return sub.select(feat_cols).to_pandas(), sub["label"].to_numpy()

    x_tr, y_tr = xy("train")
    w = np.where(y_tr == 1, (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1), 1.0)
    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=250,
        l2_regularization=0.1,
        min_samples_leaf=40,
        random_state=42,
    )
    clf.fit(x_tr, y_tr, sample_weight=w)

    out = {"features": feat_cols, "splits": {}}
    for sp in ("valid", "test"):
        x, y = xy(sp)
        p = clf.predict_proba(x)[:, 1]
        sub = deploys.filter(pl.col("split") == sp)
        cold = sub["prior_bought_same_signer"].to_numpy() == 0
        roc = float(roc_auc_score(y, p))
        pr = float(average_precision_score(y, p))
        roc_c = float(roc_auc_score(y[cold], p[cold])) if cold.any() and y[cold].min() != y[cold].max() else None
        out["splits"][sp] = {
            "roc_auc": roc,
            "pr_auc": pr,
            "cold_roc": roc_c,
            "mean_deployer_events": float(sub["deployer_events_before"].mean()),
        }
        print(f"{sp} ROC={roc:.3f} PR={pr:.3f} cold_ROC={roc_c}")

    import joblib

    joblib.dump({"model": clf, "features": feat_cols}, ROOT / "models" / "coldstart_hgb.joblib")
    write_json(paths["metadata"] / "coldstart_metrics.json", out)

    report = paths["metadata"] / "INFORME_FINAL.md"
    extra = (
        "\n\n## 11. Update cold-start (activity pre-t_decision)\n\n"
        f"Test ROC={out['splits']['test']['roc_auc']:.3f}  "
        f"PR={out['splits']['test']['pr_auc']:.3f}  "
        f"cold ROC={out['splits']['test']['cold_roc']}\n"
        "Features include `deployer_events_before` from filtered not_bought + bought activity, "
        "only events with timestamp < deploy blockTime.\n"
    )
    if report.is_file():
        text = report.read_text(encoding="utf-8")
        if "## 11. Update cold-start" not in text:
            report.write_text(text.rstrip() + extra, encoding="utf-8")
    print(f"Wrote {paths['metadata'] / 'coldstart_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
