#!/usr/bin/env python3
"""Inspect parquet indexes/activity schemas (pyarrow/polars); soft-fail if missing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import ensure_dirs, format_bytes, load_config, write_json  # noqa: E402


def _inspect_parquet(path: Path, sample_rows: int = 5) -> dict:
    info: dict = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        return info

    info["size_bytes"] = path.stat().st_size
    info["size_human"] = format_bytes(info["size_bytes"])

    # Prefer pyarrow for schema; polars for sample/nulls
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        info["num_row_groups"] = pf.metadata.num_row_groups if pf.metadata else None
        info["num_rows"] = pf.metadata.num_rows if pf.metadata else None
        info["schema"] = [
            {"name": f.name, "type": str(f.type)} for f in schema
        ]
    except ImportError:
        info["pyarrow"] = "NOT INSTALLED"
    except Exception as exc:  # noqa: BLE001
        info["pyarrow_error"] = str(exc)

    try:
        import polars as pl

        lf = pl.scan_parquet(str(path))
        # row count via metadata if possible
        try:
            n = lf.select(pl.len()).collect().item()
            info["num_rows_polars"] = int(n)
        except Exception:  # noqa: BLE001
            pass
        sample = lf.head(sample_rows).collect()
        info["sample_rows"] = sample.to_dicts()
        # null counts on a streamed sample of up to 100k rows if huge
        try:
            nulls = (
                lf.head(100_000)
                .collect()
                .null_count()
                .to_dicts()[0]
            )
            info["null_counts_first_100k"] = nulls
        except Exception as exc:  # noqa: BLE001
            info["null_counts_error"] = str(exc)
    except ImportError:
        info["polars"] = "NOT INSTALLED"
        # Fallback sample via pyarrow
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(path, columns=None)
            if info.get("num_rows") is None:
                info["num_rows"] = table.num_rows
            info["sample_rows"] = table.slice(0, sample_rows).to_pylist()
            nulls = {}
            for i, name in enumerate(table.column_names):
                nulls[name] = int(table.column(i).null_count)
            info["null_counts"] = nulls
        except Exception as exc:  # noqa: BLE001
            info["fallback_error"] = str(exc)
    except Exception as exc:  # noqa: BLE001
        info["polars_error"] = str(exc)

    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rows", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config()
    paths = ensure_dirs(cfg)
    positives = paths["positives"]
    wallet = paths["wallet"]
    urls = cfg.get("urls") or {}
    wf = urls.get("wallet_files") or {}

    candidates = {
        "bought_deploy_txs_index": positives / "bought_deploy_txs_index.parquet",
        "bought_deployers_activity": positives / "bought_deployers_activity.parquet",
        "wallet_activity": wallet
        / wf.get("activity", "5brv79e_activity.parquet"),
        "wallet_activity_txs_index": wallet
        / wf.get("activity_txs_index", "5brv79e_activity_txs_index.parquet"),
    }

    missing = [k for k, p in candidates.items() if not p.is_file()]
    present = [k for k, p in candidates.items() if p.is_file()]

    if not present:
        print(
            "No index/activity parquet files found yet. Missing:\n  - "
            + "\n  - ".join(f"{k}: {candidates[k]}" for k in missing)
        )
        print(
            "\nRun scripts/03_download_wallet.py and/or scripts/04_download_positives.py first."
        )
        write_json(
            paths["metadata"] / "index_schemas.json",
            {"files": {k: {"path": str(p), "exists": False} for k, p in candidates.items()}},
        )
        return 0

    if "pyarrow" not in sys.modules:
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pass
    try:
        import polars  # noqa: F401
    except ImportError:
        if "pyarrow" not in sys.modules:
            try:
                import pyarrow  # noqa: F401
            except ImportError:
                print(
                    "WARNING: neither polars nor pyarrow installed. "
                    "pip install polars pyarrow"
                )

    report = {"files": {}}
    for key, path in candidates.items():
        print(f"\n=== {key} ===")
        print(f"path: {path}")
        info = _inspect_parquet(path, sample_rows=args.sample_rows)
        report["files"][key] = info
        if not info.get("exists"):
            print("  MISSING")
            continue
        print(f"  size: {info.get('size_human')}")
        print(f"  rows: {info.get('num_rows') or info.get('num_rows_polars')}")
        schema = info.get("schema") or []
        if schema:
            print("  schema:")
            for col in schema:
                print(f"    - {col['name']}: {col['type']}")
        nulls = info.get("null_counts") or info.get("null_counts_first_100k")
        if nulls:
            print(f"  null_counts: {nulls}")
        sample = info.get("sample_rows") or []
        if sample:
            print(f"  sample[0]: {sample[0]}")

    out = paths["metadata"] / "index_schemas.json"
    write_json(out, report)
    print(f"\nWrote {out}")
    if missing:
        print("Still missing:")
        for k in missing:
            print(f"  - {k}: {candidates[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
