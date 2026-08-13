#!/usr/bin/env python3
"""Collect local system metadata for Phase 1."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import ensure_dirs, load_config, write_json  # noqa: E402


TOOLS = ["curl", "wget", "aria2c", "git", "pigz", "gzip", "zstd", "duckdb"]
PY_PACKAGES = ["polars", "pyarrow", "duckdb", "pandas", "yaml", "httpx", "tqdm"]


def _run(cmd: list[str], timeout: float = 30.0) -> str:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr and not p.stdout else "")
        return out.strip()
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def _pkg_version(name: str) -> str | None:
    mod_name = "yaml" if name == "yaml" else name
    try:
        mod = __import__(mod_name if name != "yaml" else "yaml")
        ver = getattr(mod, "__version__", None)
        if ver:
            return str(ver)
    except ImportError:
        pass
    dist = {
        "yaml": "PyYAML",
        "pyarrow": "pyarrow",
    }.get(name, name)
    try:
        return importlib.metadata.version(dist)
    except Exception:  # noqa: BLE001
        return None


def collect() -> dict:
    os_release: dict[str, str] = {}
    os_release_path = Path("/etc/os-release")
    if os_release_path.is_file():
        for line in os_release_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                os_release[k] = v.strip().strip('"')

    df_out = _run(["df", "-h", "/", "/home", str(ROOT)])
    free_out = _run(["free", "-h"])

    tools = {}
    for t in TOOLS:
        path = shutil.which(t)
        tools[t] = {"present": path is not None, "path": path}

    py_pkgs = {p: _pkg_version(p) for p in PY_PACKAGES}

    report = {
        "ubuntu": {
            "pretty_name": os_release.get("PRETTY_NAME"),
            "version": os_release.get("VERSION"),
            "version_id": os_release.get("VERSION_ID"),
            "os_release": os_release,
        },
        "kernel": platform.release(),
        "uname": platform.uname()._asdict(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "version_info": list(sys.version_info[:3]),
        },
        "disk_df": df_out,
        "memory_free": free_out,
        "tools": tools,
        "python_packages": py_pkgs,
        "cwd": str(Path.cwd()),
        "project_root": str(ROOT),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: data/metadata/system_report.json)",
    )
    args = parser.parse_args()

    cfg = load_config()
    paths = ensure_dirs(cfg)
    out = args.out or (paths["metadata"] / "system_report.json")

    report = collect()
    write_json(out, report)

    print("=== System report ===")
    print(f"OS:      {report['ubuntu'].get('pretty_name')}")
    print(f"Kernel:  {report['kernel']}")
    print(f"Python:  {report['python']['version_info']} @ {report['python']['executable']}")
    print("--- disk (df -h) ---")
    print(report["disk_df"])
    print("--- memory (free -h) ---")
    print(report["memory_free"])
    print("--- tools ---")
    for name, info in report["tools"].items():
        mark = "OK" if info["present"] else "MISSING"
        print(f"  {name:12} {mark:7} {info['path'] or ''}")
    print("--- python packages ---")
    for name, ver in report["python_packages"].items():
        print(f"  {name:12} {ver or 'NOT INSTALLED'}")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
