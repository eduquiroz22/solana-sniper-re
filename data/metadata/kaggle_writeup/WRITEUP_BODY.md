# Same-block sniper: known deployers + anti-factory

**One line.** Wallet `5brv79e…` does not forecast price. It buys in the same Solana block as the create, mostly from deployers it already bought, and — when the deployer is new — skips factories that spray tokens every few minutes.

`t_decision` is the deploy `blockTime`. Every feature below is known at or before that instant. Post-deploy candles are used only for P&L, never as inputs.

---

## Part 1 — How the bot trades

From its own activity (87,007 rows; 15,927 sniped deploys, Mar–Jun 2026):

- **79.6%** of first buys land in the **same slot** as the create (~400 ms). 18.7% in the next slot. Median wait: **0 seconds**.
- The buy is a *separate* transaction (the sniper never signs the create), typically **~118 indices later** in the same block — mempool/bundle speed, not a research window.
- Median entry **1.98 SOL**; median time to first sell **1 second**.
- Closed SOL/WSOL books, after gas + tip + DEX: **+8,894 SOL** net on 15,708 positions. Hit rate **55.6%** (mean win +1.40, mean loss −0.47). Max drawdown about **−47 SOL**.
- Fees eat roughly half of gross (+17,629 → +8,894). Without fees the hit rate looks like 78%; with fees it is 56%.

Known vs first-time deployers both pay: hot **+5,794 SOL**, cold **+2,938 SOL**. Cold trades are rarer but a bit larger on average.

See gallery: same-block, equity curve, fees.

---

## Part 2 — Decision rules (no future)

**Rule A — repeat deployer (hot).** If the sniper already bought this `tx_signer` *before* this deploy, June buy rate is **38.8%** vs **4.8%** for first-time (cold). That single flag already ranks at ROC **0.77**.

**Rule B — anti-factory (cold).** On strangers, the bot is not reading a secret price. It filters spray wallets:

| Signal at `t_decision` | Bought | Skipped |
|---|---|---|
| 3+ launches in the last hour | 0.75% buy rate | 7.5% if no burst |
| Hours since previous launch (median) | **~26 h** | **~2.5 min** |
| SOL spent in the create tx (median) | **~1.03** | **~0.16** |

Tips to astra/rapid/Jito, shared txs with hot wallets, and “rare token” co-occurrence did **not** hold out of time. Same-tx create+buy is nearly universal and does not separate classes. Cold buys are not slower than hot buys — there is no extra think-time.

**Classifier.** HistGradientBoosting on those features plus deploy-tx shape (CU, index, Pump.fun, inner ix, …). Time split: train < 2026-05-29 ≤ valid < 2026-06-12 ≤ test. Threshold chosen by **max F1 on valid only** (0.23 after we dropped a 15× positive weight that inflated scores and *hurt* PR/F1).

June **sample** holdout (all 15,927 positives + ~197k sampled negatives — not the full ~5M):

| | ROC | PR-AUC | P | R | F1 | P@100 |
|---|---|---|---|---|---|---|
| Valid | 0.94 | 0.71 | 0.62 | 0.78 | 0.69 | 0.93 |
| **Test** | **0.95** | **0.70** | **0.56** | **0.81** | **0.66** | **0.93** |

Ablation on test ROC: known-flag 0.77 → +tx shape 0.92 → all features **0.95**. A one-line rule (“if hot, buy”) has F1 0.49; the model reaches 0.66.

**Caveat judges should see.** A public notebook on this competition reports June AP **0.22** on the **full** ~852k deploys. Our 0.70 is the same story on an easier sample. We do not claim 0.70 on the raw universe.

---

## Part 3 — Replica vs the bot (June test)

Threshold frozen from valid:

| | Bot | Replica |
|---|---|---|
| Buys | 2,445 | 3,564 |
| Overlap | — | **1,989 (81% recall)** |
| Precision | — | 56% |
| Net SOL on tokens with a wallet path | **+1,192** | **+970** (overlap only) |

1,575 extra replica buys have **no local price path** (`mcap_candles` not downloaded). We do not invent that P&L.

A public replica that fills **one slot late** loses ~5 SOL. Copying *who* to buy is not the same as landing in the same block.

---

## Reproducibility

- Features: `t_decision` = deploy `blockTime` only.
- Code: scripts `00`–`23` + `src/`. Writeup figures: `data/metadata/kaggle_writeup/`.
- Notebook + public repo linked below (same pipeline: features → train → valid threshold → test → wallet P&L).

---

## What we would not claim

The sniper is not an oracle. It is a same-block hunter with a trust list and a factory filter. That is enough to reconstruct most of its buys and to explain the +8.9k SOL book. Execution latency is a separate war.
