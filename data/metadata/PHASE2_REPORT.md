# Phase 2 report — baseline sniper classifier

Frozen lab notes. The official model is `scripts/train_eval.py` (no class weight; see `kaggle_train_backtest.json`).

Leakage-safe tabular model at `t_decision` = deployment `blockTime`.
No post-deploy trades, mcap, or `bought_deployers_activity` (that table is defined by the label).

## Data

| Split | Rows | Positives | Pos rate |
|-------|------|-----------|----------|
| train | 173036 | 11097 | 6.41% |
| valid | 17286 | 2385 | 13.80% |
| test  | 22132 | 2445 | 11.05% |

Cuts: train < `2026-05-29` ≤ valid < `2026-06-12` ≤ test.
Positives: full `bought_deploy_txs_index` (15,927). Negatives: `negative_200k.parquet` (196,878; overlap with positives dropped).

## Test results (held-out 2026-06-12 → 2026-06-30)

| Metric | Valid | Test |
|--------|-------|------|
| ROC-AUC | 0.840 | 0.819 |
| PR-AUC (vs pos rate) | 0.550 (base 0.138) | 0.489 (base 0.110) |
| Precision@100 | 0.80 | 0.86 |
| Precision@#pos | 0.593 | 0.517 |
| Precision / Recall @0.5 | 0.295 / 0.806 | 0.219 / 0.784 |

On test, at ~20% recall the precision is **0.67** (488 tp / 240 fp). Random would be ~0.11.

**Reading:** the model ranks well (AUC 0.82, 86 of the top-100 test scores are real snipes). At a default 0.5 threshold it is too aggressive (lots of false positives). Use a high score cutoff if you want precision.

## What the model is using

Permutation importance on valid (ROC-AUC drop):

| Feature | Importance |
|---------|------------|
| `prior_bought_same_signer` | **0.253** |
| `token_is_pump` | 0.021 |
| `dow` | 0.013 |
| `hour_utc` | 0.012 |
| everything else | ~0 |

The sniper **repeats deployers it already bought**. That single feature is most of the signal. Pump.fun mint suffix and time-of-day/week add a little. Wallet activity counters added nothing (from/to on the activity table are mostly empty).

## Caveats

- Negatives are a **sample** of ~5.06M non-buys, not the full universe. Precision in production against all deploys would be lower.
- `creator_address` is almost always null on positives; identity is `tx_signer`.
- No deployer activity history for non-bought tokens (23 GiB file not downloaded on purpose).
- Train AUC 0.95 vs test 0.82 → some overfit; still generalizes.
- This is a **baseline**, not a prize stack. Next gains: parse deploy JSONL (programs/ix), same-block positioning, more honest negative activity.

## Files

- `data/processed/labeled_features.parquet`
- `models/baseline_hgb.joblib`
- `data/metadata/baseline_metrics.json`
- `scripts/09_train_baseline.py`
