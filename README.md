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

## Repo map (for reviewers)

| Path | What it is |
|------|------------|
| `notebooks/solana-sniper-reverse-engineering.ipynb` | The paper: rules, figures, metrics, replica |
| `notebooks/assets/` | Figure PNGs (also embedded in the notebook) |
| `data/metadata/kaggle_writeup/WRITEUP_BODY.md` | Short writeup text for the Kaggle form |
| `data/metadata/kaggle_train_backtest.json` | Numbers dump |
| `scripts/00`–`24`, `src/` | Full pipeline if you have the challenge files |
| `data/raw/`, `data/processed/` | **Not in git** (~30 GB). Download from the competition hosts. |

No future prices are used as features. Post-deploy data is P&L only.

---

## Reproduce locally (optional)

Needs the challenge wallet + sampled deploys on disk, then:

```bash
python -m pip install -r requirements.txt
python scripts/19_train_backtest_kaggle.py
python scripts/23_kaggle_figures_en.py
```

The notebook itself only needs `notebooks/assets/` — it does not download the TAR.

---

## Infra notes (Phase 1)

Downloads are gated (`config.yaml`, `--i-approve-large-download`). The half-year TAR is ~39 GiB and has no HTTP Range. See `scripts/00`–`07` if you are rebuilding the sample from scratch.
