#!/usr/bin/env bash
# Bootstrap this repo on a fresh host (e.g. multivac): venv + deps + dirs + system report.
# Usage (from repo root):
#   bash scripts/08_bootstrap_host.sh
#   bash scripts/08_bootstrap_host.sh --python /usr/bin/python3.12

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3.12}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

echo "=== host bootstrap ==="
echo "root:    $ROOT"
echo "python:  $PYTHON_BIN"
echo "venv:    $VENV_DIR"
echo "host:    $(hostname 2>/dev/null || true)"
echo "user:    $(whoami)"
echo

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "ERROR: python not executable: $PYTHON_BIN" >&2
  echo "Tried candidates:" >&2
  command -v python3.12 || true
  command -v python3 || true
  exit 1
fi

"$PYTHON_BIN" --version
echo "nproc=$(nproc 2>/dev/null || echo '?')"
free -h 2>/dev/null | head -3 || true
df -h "$HOME" "$ROOT" 2>/dev/null || df -h . || true
echo

mkdir -p \
  data/raw/wallet data/raw/positives data/raw/negatives \
  data/samples data/processed data/metadata \
  notebooks models results/figures results/tables

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating venv..."
  if ! "$PYTHON_BIN" -m venv "$VENV_DIR"; then
    echo "venv module failed. Trying ensurepip + venv --without-pip..." >&2
    "$PYTHON_BIN" -m ensurepip --upgrade || true
    "$PYTHON_BIN" -m venv --without-pip "$VENV_DIR"
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
    "$VENV_DIR/bin/python" /tmp/get-pip.py
  fi
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install -r "$ROOT/requirements.txt"

echo
echo "=== installed ==="
python -c "import sys, polars, pyarrow, duckdb, pandas, yaml, httpx; print('python', sys.version.split()[0]); print('polars', polars.__version__); print('pyarrow', pyarrow.__version__); print('duckdb', duckdb.__version__); print('pandas', pandas.__version__)"

echo
echo "=== system report ==="
python "$ROOT/scripts/00_system_report.py"

echo
echo "Bootstrap OK."
echo "Activate with:  source $VENV_DIR/bin/activate"
echo "Next (optional, from laptop): rsync data/raw/{wallet,positives} here, then scripts 05–07."
