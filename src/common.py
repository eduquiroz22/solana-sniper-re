"""Shared helpers for pipeline scripts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_CONFIG_CACHE: dict[str, Any] | None = None


def project_root() -> Path:
    return PROJECT_ROOT


def load_config(path: str | Path | None = None, *, reload: bool = False) -> dict[str, Any]:
    """Load config.yaml from project root (cached unless reload=True)."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not reload and path is None:
        return _CONFIG_CACHE

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyyaml. Install with: pip install pyyaml"
        ) from exc

    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config.yaml must be a mapping, got {type(data)}")

    if path is None:
        _CONFIG_CACHE = data
    return data


def ensure_dirs(cfg: dict[str, Any] | None = None) -> dict[str, Path]:
    """Create standard data directories from config paths; return resolved Paths."""
    cfg = cfg or load_config()
    paths_cfg = cfg.get("paths") or {}
    defaults = {
        "raw": "data/raw",
        "samples": "data/samples",
        "processed": "data/processed",
        "metadata": "data/metadata",
        "positives": "data/raw/positives",
        "wallet": "data/raw/wallet",
        "negatives": "data/raw/negatives",
    }
    resolved: dict[str, Path] = {}
    for key, default in defaults.items():
        rel = paths_cfg.get(key, default)
        p = PROJECT_ROOT / rel if not Path(rel).is_absolute() else Path(rel)
        p.mkdir(parents=True, exist_ok=True)
        resolved[key] = p

    # Nested raw dirs used by download scripts
    for sub in ("positives", "wallet", "negatives"):
        (resolved["raw"] / sub).mkdir(parents=True, exist_ok=True)
    return resolved


def format_bytes(n: int | float | None) -> str:
    if n is None:
        return "unknown"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return str(n)
    if n < 0:
        return f"{n} B"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(n)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{n} B"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, obj: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=str, ensure_ascii=False)
        fh.write("\n")
    return path


def max_auto_download_bytes(cfg: dict[str, Any] | None = None) -> int:
    cfg = cfg or load_config()
    return int(cfg.get("max_auto_download_bytes", 1073741824))


def require_large_download_approval(
    nbytes: int | None,
    *,
    approved: bool,
    limit: int | None = None,
    label: str = "transfer",
) -> None:
    """Exit if nbytes exceeds the auto-download limit without approval."""
    limit = limit if limit is not None else max_auto_download_bytes()
    if nbytes is None:
        return
    if nbytes > limit and not approved:
        raise SystemExit(
            f"Refusing {label} of {format_bytes(nbytes)} "
            f"(limit {format_bytes(limit)}). "
            "Re-run with --i-approve-large-download to proceed."
        )


def add_project_to_syspath() -> Path:
    """Ensure PROJECT_ROOT is importable when running scripts/*.py."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT
