# Solana Sniper Bot Reverse-Engineering

Kaggle hackathon: [Solana Sniper Bot Reverse-Engineering](https://www.kaggle.com/competitions/solana-sniper-bot-reverse-engineering).

**Read this first (the paper):** [`notebooks/solana-sniper-reverse-engineering.ipynb`](notebooks/solana-sniper-reverse-engineering.ipynb)

**How to submit:** [`data/metadata/kaggle_writeup/ENTREGA.md`](data/metadata/kaggle_writeup/ENTREGA.md)

Target wallet: `5brv79eFZ2rGprXNvqgVJBkBptkkw8GJX1XydJyZLyAr`

---

# Phase 1 notes (infra)

Local infrastructure and data-prep pipeline for the same hackathon.

**Phase 1 scope:** system probes, server investigation, project layout, wallet + positive downloads (gated), index inspection, temporal EDA design, and a gated negative sampler. **No ML features yet.**

Target wallet: `5brv79eFZ2rGprXNvqgVJBkBptkkw8GJX1XydJyZLyAr`

## Safety rules

- Auto-download capped by `max_auto_download_bytes` in `config.yaml` (**1 GiB**).
- Transfers **> 1 GiB** require `--i-approve-large-download`.
- Script `04` TAR extract also requires `--yes`.
- Script `07` defaults to **dry-run**; `--execute` requires `--i-approve-large-download`.
- June supplements (raw blocks / Jito / trades / mcap) are disabled in config.

## Setup (laptop)

```bash
cd /home/eduardo/Programming/KAGGLE/solana_sniper
python -m pip install -r requirements.txt
```

## Setup on multivac (fresh host)

`multivac` starts empty. Copy code from the laptop, then bootstrap a Python 3.12 venv there.

**On the laptop:**

```bash
cd /home/eduardo/Programming/KAGGLE/solana_sniper
bash scripts/rsync_to_multivac.sh                 # code + metadata
# bash scripts/rsync_to_multivac.sh --with-data  # also ~783 MiB wallet+positives
```

If SSH is not `eduardo@multivac`:

```bash
bash scripts/rsync_to_multivac.sh --host eduardo@HOST --dest ~/solana_sniper --with-data
```

**On multivac:**

```bash
cd ~/solana_sniper
bash scripts/08_bootstrap_host.sh
source .venv/bin/activate
python scripts/00_system_report.py
```

Check `nproc`, RAM and `df -h $HOME` before streaming negatives (~14.5 GiB network). Do not download the full TAR unless there is ≥50 GiB free.

## Pipeline scripts (run in order)

| Script | Purpose |
|--------|---------|
| `00_system_report.py` | OS / Python / disk / RAM / tools → `data/metadata/system_report.json` |
| `01_probe_servers.py` | HEAD / Range / listings → `data/metadata/server_probe.json` |
| `02_tar_toc_probe.py` | Stream TAR headers only (default ≤180 MiB) → `data/metadata/tar_toc.json` |
| `03_download_wallet.py` | Wallet activity + index (~18.5 MiB); optional `--with-jsonl` |
| `04_download_positives.py` | Stream-extract `bought_*` from TAR (`--yes`) |
| `05_inspect_indexes.py` | Schemas / row counts → `data/metadata/index_schemas.json` |
| `06_temporal_eda.py` | Temporal distribution + proposed train/valid/test cuts |
| `07_sample_negatives.py` | Dry-run plan by default; gated streaming sample of negatives |

Examples:

```bash
python scripts/00_system_report.py
python scripts/01_probe_servers.py
python scripts/02_tar_toc_probe.py --max-transfer-mib 180
python scripts/03_download_wallet.py --with-jsonl
python scripts/04_download_positives.py --yes
python scripts/05_inspect_indexes.py
python scripts/06_temporal_eda.py
python scripts/07_sample_negatives.py          # dry-run only
```

## `t_decision` constraint

All entry-decision features must use only information available at or before token deployment (`t_decision_rule: deployment_blockTime`). Post-deployment trades / mcap may be used later for outcomes and backtest P&L only — never as model inputs.

## Data layout

```
data/raw/{wallet,positives,negatives}/
data/samples/          # e.g. negative_200k.parquet
data/processed/
data/metadata/         # probe JSON, schemas, EDA
```

## Known hosting constraints

- Core archive `half_year_dataset.tar` (~38.7 GiB) is served by Python SimpleHTTP **without HTTP Range**; individual members are not URL-accessible.
- Full TAR does not fit on a disk with only ~31 GiB free — prefer streaming + samples.
- Wallet host supports Range requests.
