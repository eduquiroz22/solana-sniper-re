#!/usr/bin/env python3
"""Buy latency + richer deploy-tx style (tips, CU price, create+buy) vs label."""

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
from src.features.tx_extract import extract_tx_features  # noqa: E402

NEW_FEATS = [
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
    "has_token_2022",
    "payer_sol_pre",
    "payer_sol_post",
]


def _log(msg: str) -> None:
    print(msg, flush=True)


def buy_latency(paths: dict[str, Path]) -> dict:
    pos = pl.read_parquet(paths["positives"] / "bought_deploy_txs_index.parquet")
    wallet = pl.read_parquet(paths["wallet"] / "5brv79e_activity.parquet").select(
        ["timestamp", "event_type", "token_address", "tx_hash"]
    )
    widx = pl.read_parquet(paths["wallet"] / "5brv79e_activity_txs_index.parquet").select(
        ["tx_hash", "blockTime", "blockSlot"]
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
        .join(
            widx.rename(
                {
                    "tx_hash": "buy_tx",
                    "blockTime": "buy_blockTime",
                    "blockSlot": "buy_slot",
                }
            ),
            on="buy_tx",
            how="left",
        )
        .with_columns(
            (pl.col("buy_ts") - pl.col("blockTime")).alias("latency_s"),
            (pl.col("buy_slot") - pl.col("blockSlot")).alias("latency_slots"),
        )
    )
    n = j.height
    dt = j["latency_s"].drop_nulls()
    ds = j["latency_slots"].drop_nulls()

    def frac(mask: pl.Series) -> float:
        return float(mask.sum() / n) if n else 0.0

    buckets_s = {
        "same_second": frac(j["latency_s"] == 0),
        "le_1s": frac(j["latency_s"] <= 1),
        "le_5s": frac(j["latency_s"] <= 5),
        "le_30s": frac(j["latency_s"] <= 30),
        "gt_30s": frac(j["latency_s"] > 30),
    }
    buckets_slots = {
        "same_slot": frac(j["latency_slots"] == 0),
        "next_slot": frac(j["latency_slots"] == 1),
        "le_2_slots": frac(j["latency_slots"] <= 2),
        "le_5_slots": frac(j["latency_slots"] <= 5),
    }
    out = {
        "n_positive_deploys": n,
        "n_matched_buys": int(j["buy_ts"].is_not_null().sum()),
        "latency_seconds": {
            "median": float(dt.median()) if dt.len() else None,
            "p90": float(dt.quantile(0.90)) if dt.len() else None,
            "p99": float(dt.quantile(0.99)) if dt.len() else None,
            "mean": float(dt.mean()) if dt.len() else None,
            "max": int(dt.max()) if dt.len() else None,
            "fractions": buckets_s,
        },
        "latency_slots": {
            "median": float(ds.median()) if ds.len() else None,
            "p90": float(ds.quantile(0.90)) if ds.len() else None,
            "mean": float(ds.mean()) if ds.len() else None,
            "max": int(ds.max()) if ds.len() else None,
            "fractions": buckets_slots,
            "note": "Solana slot is ~400 ms; same slot = same block as the create tx.",
        },
        "sniper_is_create_signer": 0,
        "interpretation": (
            "Most buys land in the same block or the next one, as a separate tx "
            "(sniper is not a signer on the create). That is mempool/bundle speed, "
            "not a multi-second research window."
        ),
    }
    _log(
        f"latency: same_s={buckets_s['same_second']:.3f} "
        f"<=1s={buckets_s['le_1s']:.3f} same_slot={buckets_slots['same_slot']:.3f} "
        f"<=2slots={buckets_slots['le_2_slots']:.3f}"
    )
    return out


def parse_positives(paths: dict[str, Path]) -> Path:
    src = paths["positives"] / "bought_deploy_txs.jsonl.gz"
    dest = paths["processed"] / "pos_tx_features.parquet"
    rows = []
    with gzip.open(src, "rt", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(extract_tx_features(obj, i))
            if i % 2000 == 0:
                _log(f"  positives parsed {i:,}")
    df = pl.DataFrame(rows)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dest)
    _log(f"Wrote {dest} rows={df.height}")
    return dest


def _summarize_split(df: pl.DataFrame) -> dict:
    n = df.height
    n_pos = int(df["label"].sum()) if n else 0

    def mean_lab(col: str, lab: int) -> float | None:
        sub = df.filter(pl.col("label") == lab)
        if sub.height == 0 or col not in sub.columns:
            return None
        return float(sub[col].cast(pl.Float64).mean())

    def pos_rate(mask: pl.Expr) -> float | None:
        sub = df.filter(mask)
        if sub.height == 0:
            return None
        return float(sub["label"].mean())

    flags = [
        "has_any_tip",
        "has_jito_tip",
        "has_service_tip",
        "has_buy_same_tx",
        "has_create_v2",
        "has_token_2022",
    ]
    out: dict = {"n": n, "n_pos": n_pos, "flags": {}}
    for f in flags:
        if f not in df.columns:
            continue
        out["flags"][f] = {
            "frac_pos": mean_lab(f, 1),
            "frac_neg": mean_lab(f, 0),
            "pos_rate_if": pos_rate(pl.col(f) == 1),
            "pos_rate_if_not": pos_rate(pl.col(f) == 0),
        }
    for col in ("cu_price_micro", "tip_lamports", "payer_sol_pre", "fee_lamports", "tx_index"):
        if col not in df.columns:
            continue
        out[f"median_{col}_pos"] = (
            float(df.filter(pl.col("label") == 1)[col].median()) if n_pos else None
        )
        out[f"median_{col}_neg"] = (
            float(df.filter(pl.col("label") == 0)[col].median()) if n - n_pos else None
        )
    return out


def _attach_tx(labeled: pl.DataFrame, paths: dict[str, Path]) -> pl.DataFrame:
    pos_tx = pl.read_parquet(paths["processed"] / "pos_tx_features.parquet")
    neg_tx = pl.read_parquet(paths["processed"] / "neg_tx_features.parquet")
    extra = [
        c
        for c in NEW_FEATS
        + [
            "tx_index",
            "cu",
            "fee_lamports",
            "n_accounts",
            "n_ix",
            "n_inner_ix",
            "has_pump_program",
            "has_compute_budget",
        ]
        if c in pos_tx.columns and c in neg_tx.columns and c not in labeled.columns
    ]
    style = pl.concat(
        [
            pos_tx.select(["tx_hash", *extra]),
            neg_tx.select(["tx_hash", *extra]),
        ]
    ).unique(subset=["tx_hash"], keep="first")
    return labeled.join(style, on="tx_hash", how="left")


def evaluate(paths: dict[str, Path]) -> dict:
    labeled = _attach_tx(
        pl.read_parquet(paths["processed"] / "labeled_features.parquet"),
        paths,
    )

    cold = labeled.filter(pl.col("prior_bought_same_signer") == 0)
    report = {
        "all": _summarize_split(labeled),
        "test": _summarize_split(labeled.filter(pl.col("split") == "test")),
        "test_cold": _summarize_split(
            labeled.filter((pl.col("split") == "test") & (pl.col("prior_bought_same_signer") == 0))
        ),
        "train_cold": _summarize_split(
            labeled.filter((pl.col("split") == "train") & (pl.col("prior_bought_same_signer") == 0))
        ),
        "n_cold": cold.height,
        "coverage_has_buy": float(labeled["has_buy_same_tx"].is_not_null().mean())
        if "has_buy_same_tx" in labeled.columns
        else None,
    }
    return report


def maybe_train(paths: dict[str, Path]) -> dict | None:
    try:
        import numpy as np
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        return None

    labeled = _attach_tx(
        pl.read_parquet(paths["processed"] / "labeled_features.parquet"),
        paths,
    )
    extra = [c for c in NEW_FEATS if c in labeled.columns]

    base = [
        "hour_utc",
        "dow",
        "token_is_pump",
        "prior_bought_same_signer",
        "tx_index",
        "cu",
        "fee_lamports",
        "n_accounts",
        "n_ix",
        "n_inner_ix",
        "has_pump_program",
        "has_compute_budget",
    ]
    use = [c for c in base + extra if c in labeled.columns]

    def xy(split: str):
        sub = labeled.filter(pl.col("split") == split)
        return sub.select(use).to_pandas(), sub["label"].to_numpy(), sub

    x_tr, y_tr, _ = xy("train")
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
    out = {"features": use, "splits": {}}
    for split in ("valid", "test"):
        x, y, sub = xy(split)
        s = clf.score if False else clf.predict_proba(x)[:, 1]
        cold = sub["prior_bought_same_signer"].to_numpy() == 0
        out["splits"][split] = {
            "roc_auc": float(roc_auc_score(y, s)),
            "pr_auc": float(average_precision_score(y, s)),
            "cold_roc": float(roc_auc_score(y[cold], s[cold])) if cold.any() else None,
        }
        _log(f"model {split} {out['splits'][split]}")
    return out


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip re-download/parse; use existing pos/neg_tx_features parquet",
    )
    args = parser.parse_args()

    cfg = load_config()
    paths = ensure_dirs(cfg)

    _log("=== buy latency ===")
    lat = buy_latency(paths)

    if not args.eval_only:
        _log("=== reparse positives ===")
        parse_positives(paths)

        _log("=== reparse negatives from TAR ===")
        import subprocess

        r = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "10_enrich_deploys.py"),
                "--source",
                "negatives",
                "--workers",
                str(int(cfg.get("n_workers") or 6)),
            ],
            cwd=str(ROOT),
            check=False,
        )
        if r.returncode != 0:
            _log(f"negative enrich failed rc={r.returncode}")
            return r.returncode or 1

    _log("=== evaluate style features ===")
    style = evaluate(paths)
    _log(f"TEST flags {style['test']['flags']}")
    _log(f"TEST COLD flags {style['test_cold']['flags']}")

    _log("=== retrain with new fields ===")
    model = maybe_train(paths)

    dest = paths["metadata"] / "deploy_style_and_latency.json"
    write_json(
        dest,
        {
            "latency": lat,
            "style": style,
            "model_with_new_fields": model,
        },
    )
    _log(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
