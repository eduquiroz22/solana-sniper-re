#!/usr/bin/env bash
# After enrich PID exits: finalize tx-feature report, then extract ~23.5 GiB activity.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/logs/nightly_continue.log"
PIDFILE="$ROOT/logs/enrich_negatives.pid"
mkdir -p logs

exec >>"$LOG" 2>&1
echo "=== nightly_continue $(date -Is) ==="

wait_pid() {
  local pid="$1"
  echo "waiting for pid $pid"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
}

ENRICH_PID="$(cat "$PIDFILE" 2>/dev/null || true)"
# Prefer the real python if still running (avoid SIGPIPE + set -o pipefail)
if pgrep -f "scripts/10_enrich_deploys.py --source negatives" >/dev/null; then
  ENRICH_PID="$(pgrep -n -f "scripts/10_enrich_deploys.py --source negatives" || true)"
fi
if [[ -n "${ENRICH_PID:-}" ]] && kill -0 "$ENRICH_PID" 2>/dev/null; then
  wait_pid "$ENRICH_PID"
else
  echo "enrich already stopped"
fi

echo "=== 11_final_analysis $(date -Is) ==="
python3 scripts/11_final_analysis.py || echo "11 failed (continuing)"

echo "=== 12_extract_neg_activity $(date -Is) ==="
python3 scripts/12_extract_neg_activity.py --yes --i-approve-large-download --timeout 86400

echo "=== 13_filter_activity $(date -Is) ==="
python3 scripts/13_filter_activity.py

echo "=== 14_coldstart_train $(date -Is) ==="
python3 scripts/14_coldstart_train.py

echo "=== nightly_continue DONE $(date -Is) ==="
