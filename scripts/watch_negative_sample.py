#!/usr/bin/env python3
"""Live progress bar for a running 07_sample_negatives.py job (tails the log)."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROGRESS_RE = re.compile(
    r"progress lines=([0-9,]+)\s+rate=([0-9,]+)/s\s+elapsed=([0-9.]+)m\s+net=([0-9.]+)\s+(MiB|GiB)"
)
MEMBER_RE = re.compile(r"Found not_bought_deploy_txs\.jsonl\.gz size=([0-9.]+) (MiB|GiB)")
DONE_RE = re.compile(r"^Wrote (.+\.parquet)")

# Kaggle docs: ~5.06M not_bought deploy rows
DEFAULT_TOTAL_LINES = 5_060_000
DEFAULT_TOTAL_GIB = 14.54


def _to_gib(value: float, unit: str) -> float:
    return value / 1024.0 if unit == "MiB" else value


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _bar(frac: float, width: int = 32) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(round(frac * width))
    return "█" * filled + "░" * (width - filled)


def _eta(remaining: float, rate: float) -> str:
    if rate <= 0 or remaining <= 0:
        return "--:--"
    secs = remaining / rate
    if secs < 60:
        return f"{secs:.0f}s"
    mins, secs = divmod(secs, 60)
    if mins < 60:
        return f"{mins:.0f}m {secs:.0f}s"
    hours, mins = divmod(mins, 60)
    return f"{hours:.0f}h {mins:.0f}m"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=ROOT / "logs" / "negative_sample_200k.log",
    )
    parser.add_argument(
        "--pid-file",
        type=Path,
        default=ROOT / "logs" / "negative_sample_200k.pid",
    )
    parser.add_argument("--total-lines", type=int, default=DEFAULT_TOTAL_LINES)
    parser.add_argument("--total-gib", type=float, default=DEFAULT_TOTAL_GIB)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    lines = 0
    rate = 0.0
    elapsed_m = 0.0
    net_gib = 0.0
    done_path: str | None = None
    last_pos = 0

    print(f"watching {args.log}")
    print("Ctrl+C to stop the watcher (does NOT kill the sampler)\n")

    while True:
        pid = _read_pid(args.pid_file)
        alive = _pid_alive(pid)

        if args.log.is_file():
            with args.log.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(last_pos)
                chunk = fh.read()
                last_pos = fh.tell()
            for raw in chunk.splitlines():
                m = PROGRESS_RE.search(raw)
                if m:
                    lines = int(m.group(1).replace(",", ""))
                    rate = float(m.group(2).replace(",", ""))
                    elapsed_m = float(m.group(3))
                    net_gib = _to_gib(float(m.group(4)), m.group(5))
                dm = DONE_RE.match(raw.strip())
                if dm:
                    done_path = dm.group(1)

        line_frac = lines / args.total_lines if args.total_lines else 0.0
        net_frac = net_gib / args.total_gib if args.total_gib else 0.0
        status = "RUNNING" if alive else ("DONE" if done_path else "IDLE/FINISHED")

        sys.stdout.write("\033[2J\033[H")  # clear
        print("negative sample 200k")
        print(f"status: {status}   pid: {pid or '?'}   elapsed: {elapsed_m:.1f}m")
        print()
        print(
            f"lines  {_bar(line_frac)}  {100*line_frac:5.1f}%   "
            f"{lines:,} / ~{args.total_lines:,}   {rate:,.0f}/s   "
            f"ETA {_eta(args.total_lines - lines, rate)}"
        )
        print(
            f"net    {_bar(net_frac)}  {100*net_frac:5.1f}%   "
            f"{net_gib:.2f} / ~{args.total_gib:.2f} GiB"
        )
        print()
        if done_path:
            print(f"output: {done_path}")
            return 0
        if not alive and lines > 0:
            print("process ended; check the log if no parquet yet")
            return 0
        if not args.log.is_file():
            print("waiting for log...")
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nstopped watcher")
        raise SystemExit(0)
