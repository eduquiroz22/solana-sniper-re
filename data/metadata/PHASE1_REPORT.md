# Phase 1 Report — Solana Sniper Infra & Data Prep

Frozen lab notes from the first data pass. The live pipeline is `scripts/` (see repo README). Script names below are historical.

## A. Servers

| Host | Role | Finding |
|------|------|---------|
| `http://65.21.203.147:48102/` | Core half-year dataset | Python SimpleHTTP; only `half_year_dataset.tar` (~**38.69 GiB**). **No `Accept-Ranges`** (Range → full 200). Individual `bought_*` / `not_bought_*` paths → **404**. |
| `http://154.12.118.112:48114/` | Target wallet activity | nginx; **Accept-Ranges: bytes** (206 OK). Three files listed. |
| June supplements (`:48110`–`:48113`) | Optional deep-dive | Documented; **not probed for download**; disabled in `config.yaml`. |

TAR member order (verified TOC + extract):

1. `bought_deploy_txs.jsonl.gz` — 50,815,788 B @ offset 0  
2. `bought_deploy_txs_index.parquet` — 1,947,279 B @ offset 50,816,512  
3. `bought_deployers_activity.parquet` — 656,959,815 B @ offset 52,764,672  
4. `not_bought_*` — later in archive (not streamed yet)

## B. Files we can download safely now

- Wallet trio (~106 MiB total) — **done**
- Positive trio via TAR stream-extract (~677 MiB) — **done**
- Kaggle Data Explorer also hosts the three `bought_*` files (~1.06 GiB) after accepting rules (alternative path)

## C. Files to avoid (for now)

- Full `half_year_dataset.tar` (~38.7 GiB) — does not fit (~30 GiB free on `/home`)
- `not_bought_deploy_txs.jsonl.gz` (~13.9 GiB) and `not_bought_deployers_activity.parquet` (~23.5 GiB) as full local copies
- June raw blocks (~429 GiB), full Jito, full `pumpfun_trades`, full `mcap_candles`

## D. Negative sampling strategy

- Stream `not_bought_deploy_txs.jsonl.gz` from the TAR (must transfer ~prefix of bought members + the jsonl).
- One-pass **hash-stratified by ISO week** (`RANDOM_SEED` + `line_number`), configurable `NEGATIVE_SAMPLE_SIZE`.
- Output: `data/samples/negative_{N}k.parquet`.
- Default CLI is **dry-run**. Execute needs:

```bash
python scripts/07_sample_negatives.py --execute --i-approve-large-download
```

Estimated network for 200k sample: **~14.5 GiB** streamed (disk only keeps the sample parquet).

`bought_deploy_txs_index` columns (useful for positives / join design):  
`tx_hash`, `line_number`, `blockTime`, `blockSlot`, `token_address`, `tx_signer`, `creator_address`  
Note: `creator_address` is null on almost all positive index rows — use `tx_signer` carefully / parse from JSONL when needed.

## E. Disk / RAM

| Item | Value |
|------|-------|
| `/home` free (post-download) | ~**30 GiB** |
| Wallet on disk | ~106 MiB |
| Positives on disk | ~677 MiB |
| RAM | ~15 GiB total — prefer Polars/DuckDB streaming |
| Full TAR | **not feasible** without freeing ≥45 GiB |

## F. Commands / scripts executed

```bash
python3 -m pip install -r requirements.txt
python3 scripts/00_system_report.py
python3 scripts/01_probe_servers.py
python3 scripts/02_tar_toc_probe.py --max-transfer-mib 180
python3 scripts/03_download_wallet.py --with-jsonl
python3 scripts/04_download_positives.py --method tar --yes
python3 scripts/05_inspect_indexes.py
python3 scripts/06_temporal_eda.py
python3 scripts/07_sample_negatives.py   # dry-run only
```

Metadata artifacts: `data/metadata/{system_report,server_probe,tar_toc,index_schemas,temporal_eda,negative_sample_plan}.json`

## G. Files actually downloaded

**Wallet** (`data/raw/wallet/`):

| File | Size | sha256 |
|------|------|--------|
| `5brv79e_activity.parquet` | 11.96 MiB | `631594b8…780f4` |
| `5brv79e_activity_txs_index.parquet` | 6.45 MiB | `8a88fe22…b6ffb5` |
| `5brv79e_activity_txs.jsonl.gz` | 87.53 MiB | `3befb560…9d6e6a` |

**Positives** (`data/raw/positives/`):

| File | Size | sha256 |
|------|------|--------|
| `bought_deploy_txs.jsonl.gz` | 48.46 MiB | `7e8330ba…5bff69` |
| `bought_deploy_txs_index.parquet` | 1.86 MiB | `48c949e9…79850f` |
| `bought_deployers_activity.parquet` | 626.53 MiB | `1e76adf3…c29c8d3e` |

No `not_bought_*`, no June supplements, no full TAR.

## H. What remains

1. **Approve** multi-GB negative stream (`07 --execute --i-approve-large-download`) if you want `negative_200k.parquet`.
2. Optionally free disk / attach volume before larger extracts (neg index 569 MiB + activity filter later).
3. **Confirm temporal cuts** in `config.yaml` (proposed, not auto-written):
   - TRAIN: `2026-03-12 → 2026-05-29`
   - VALID: `2026-05-29 → 2026-06-12`
   - TEST: `2026-06-12 → 2026-06-30`
4. Decide later whether June supplements are needed for Part 1 in-block positioning / Part 3 P&L.
5. Phase 2+: features under `t_decision`, modeling, backtest — **not started**.
