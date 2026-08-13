#!/usr/bin/env bash
# Copy this repo from the laptop to a remote host with scp (no remote rsync).
# Default remote: eduardo@multivac:~/solana_sniper
#
# Usage (on the LAPTOP, from repo root):
#   bash scripts/rsync_to_multivac.sh
#   bash scripts/rsync_to_multivac.sh --with-data
#   bash scripts/rsync_to_multivac.sh --host eduardo@multivac --dest ~/solana_sniper

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${MULTIVAC_HOST:-eduardo@multivac}"
DEST="${MULTIVAC_DEST:-~/solana_sniper}"
WITH_DATA=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --with-data) WITH_DATA=1; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

echo "scp $ROOT -> $HOST:$DEST"
echo "with_data=$WITH_DATA"

ssh "$HOST" "mkdir -p $DEST/src $DEST/scripts $DEST/data/metadata $DEST/data/raw/wallet $DEST/data/raw/positives $DEST/data/raw/negatives $DEST/data/samples $DEST/data/processed $DEST/notebooks $DEST/models $DEST/results/figures $DEST/results/tables"

scp \
  "$ROOT/config.yaml" \
  "$ROOT/README.md" \
  "$ROOT/requirements.txt" \
  "$HOST:$DEST/"

if [[ -f "$ROOT/.gitattributes" ]]; then
  scp "$ROOT/.gitattributes" "$HOST:$DEST/"
fi

scp -r "$ROOT/src" "$ROOT/scripts" "$HOST:$DEST/"
scp "$ROOT"/data/metadata/* "$HOST:$DEST/data/metadata/"

if [[ "$WITH_DATA" -eq 1 ]]; then
  scp -r "$ROOT/data/raw/wallet/." "$HOST:$DEST/data/raw/wallet/"
  scp -r "$ROOT/data/raw/positives/." "$HOST:$DEST/data/raw/positives/"
fi

echo
echo "On the remote host:"
echo "  ssh $HOST"
echo "  cd $DEST"
echo "  bash scripts/08_bootstrap_host.sh --python /usr/bin/python3.12"
if [[ "$WITH_DATA" -eq 0 ]]; then
  echo
  echo "Data not copied. Re-run with --with-data or download on the remote:"
  echo "  python scripts/03_download_wallet.py --with-jsonl"
  echo "  python scripts/04_download_positives.py --yes"
fi
