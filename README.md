# Same-block sniper, reverse-engineered

Kaggle hackathon: [Solana Sniper Bot Reverse-Engineering](https://www.kaggle.com/competitions/solana-sniper-bot-reverse-engineering)

**Start here:** [`notebooks/solana-sniper-reverse-engineering.ipynb`](notebooks/solana-sniper-reverse-engineering.ipynb)  
(open it on GitHub or Kaggle — figures are embedded)

Target wallet: `5brv79eFZ2rGprXNvqgVJBkBptkkw8GJX1XydJyZLyAr`

---

## What we found

This sniper does **not** forecast price. At `t_decision` = the deploy `blockTime` he:

1. Buys in the **same Solana block** as the create (**79.6%** same slot, median wait **0 s**, sells in **~1 s**).
2. Mostly follows **deployers he already bought** (June buy rate **38.8%** hot vs **4.8%** cold).
3. When the deployer is new, **skips token factories** (3+ launches / hour → 0.75% buy rate vs 7.5%; last launch ~26 h vs ~2 min; create size ~1 SOL vs 0.16).

After gas + tip + DEX: **+8,894 SOL** net (hit 55.6%, drawdown ~−47 SOL).

**June holdout** (time split; threshold chosen on valid only): ROC **0.95**, PR-AUC **0.70**, F1 **0.66**, P@100 **0.93**, replica recall **81%** (1,989 / 2,445), **+970 SOL** of the bot’s +1,192 test P&L on overlapping tokens.

PR-AUC is on a **sample** of negatives (~200k of ~5M). A public kernel on the full June universe reports AP **0.22**. Same story, harder exam.

---

## Reproduce

Challenge files are **not** in git (~30 GB). Point `config.yaml` at the hosts, then:

```bash
python -m pip install -r requirements.txt

python scripts/download_wallet.py
python scripts/download_positives.py --yes
python scripts/sample_negatives.py --execute --i-approve-large-download
python scripts/extract_tx_features.py
python scripts/extract_deployer_activity.py --yes --i-approve-large-download
python scripts/filter_deployer_activity.py
python scripts/extract_factory_features.py
python scripts/train_eval.py
python scripts/make_figures.py
```

| Script | Writes |
|--------|--------|
| `download_wallet.py` | `data/raw/wallet/` |
| `download_positives.py` | `data/raw/positives/` |
| `sample_negatives.py` | `data/samples/negative_200k.parquet` |
| `extract_tx_features.py` | `pos_tx_features.parquet`, `neg_tx_features.parquet` |
| `extract_deployer_activity.py` + `filter_deployer_activity.py` | filtered deployer history |
| `extract_factory_features.py` | `labeled_features.parquet`, `cold_hypothesis_table.parquet` |
| `train_eval.py` | scores, net P&L, `kaggle_train_backtest.json` (official metrics) |
| `make_figures.py` | `data/metadata/kaggle_writeup/*.png` from those outputs |

The notebook only needs `notebooks/assets/`. It does not download the TAR.

No future prices as features. Post-deploy data is P&L only.

---

## Repo map

| Path | What |
|------|------|
| `notebooks/solana-sniper-reverse-engineering.ipynb` | Paper: rules, figures, metrics, replica |
| `scripts/`, `src/` | Pipeline above |
| `data/metadata/` | Metrics JSON + English figures |
| `data/raw/`, `data/processed/` | Not in git |
