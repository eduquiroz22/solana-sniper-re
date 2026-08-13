#!/usr/bin/env python3
"""
Sample negatives from not_bought_deploy_txs.jsonl.gz via TAR stream.

Default: --dry-run (prints plan + transfer estimate only).
Execute requires --execute AND --i-approve-large-download (multi-GB network).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import (  # noqa: E402
    ensure_dirs,
    format_bytes,
    load_config,
    max_auto_download_bytes,
    write_json,
)
from src.download.tar_stream import (  # noqa: E402
    BLOCK,
    discard_bytes,
    padded_size,
    parse_ustar_header,
)
from src.sampling.negative_sample import (  # noqa: E402
    describe_sampling_plan,
    estimate_negative_tar_transfer_bytes,
    rows_to_parquet,
    sample_from_jsonl_gz_stream,
)


def _output_path(samples_dir: Path, size: int) -> Path:
    if size % 1000 == 0 and size >= 1000:
        return samples_dir / f"negative_{size // 1000}k.parquet"
    return samples_dir / f"negative_{size}.parquet"


def _open_url(url: str, timeout: float):
    try:
        import httpx

        # Long stream: no read timeout; still bound connect.
        to = httpx.Timeout(timeout, connect=min(60.0, timeout), read=None, write=None, pool=None)
        client = httpx.Client(follow_redirects=True, timeout=to)
        req = client.build_request(
            "GET", url, headers={"User-Agent": "solana-sniper/neg-sample"}
        )
        resp = client.send(req, stream=True)
        if resp.status_code >= 400:
            resp.close()
            client.close()
            raise RuntimeError(f"HTTP {resp.status_code}")

        class _Reader:
            def __init__(self):
                self._it = resp.iter_bytes(1024 * 1024)
                self._buf = bytearray()
                self.bytes_read = 0

            def read(self, n: int) -> bytes:
                while len(self._buf) < n:
                    try:
                        chunk = next(self._it)
                    except StopIteration:
                        break
                    self._buf.extend(chunk)
                out = bytes(self._buf[:n])
                del self._buf[:n]
                self.bytes_read += len(out)
                return out

            def close(self):
                resp.close()
                client.close()

        return _Reader()
    except ImportError:
        from urllib.request import Request, urlopen

        return urlopen(
            Request(url, headers={"User-Agent": "solana-sniper/neg-sample"}),
            timeout=timeout,
        )


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


class _TarMemberStream:
    """File-like that yields only the raw bytes of a single TAR member body."""

    def __init__(self, stream, size: int):
        self._stream = stream
        self._left = size
        self._pad = padded_size(size) - size

    def read(self, n: int = -1) -> bytes:
        if self._left <= 0:
            return b""
        if n < 0 or n > self._left:
            n = self._left
        data = _read_exact(self._stream, n)
        self._left -= len(data)
        if self._left <= 0 and self._pad:
            discard_bytes(self._stream, self._pad)
            self._pad = 0
        return data

    def close(self):
        # Drain remaining body + padding so caller can continue if needed
        if self._left > 0:
            discard_bytes(self._stream, self._left)
            self._left = 0
        if self._pad:
            discard_bytes(self._stream, self._pad)
            self._pad = 0


def stream_sample_from_tar(
    url: str,
    member_name: str,
    *,
    sample_size: int,
    seed: int,
    max_transfer_bytes: int | None,
    approve_large: bool,
    timeout: float,
    workers: int = 6,
    executor=None,
) -> tuple[list[dict], dict]:
    """Locate member in TAR stream, then stratified-sample its gzip jsonl body."""
    target = member_name.lstrip("./")
    target_base = Path(target).name
    stream = _open_url(url, timeout=timeout)
    transferred = 0
    try:
        empty = 0
        while True:
            need = BLOCK
            if (
                max_transfer_bytes is not None
                and transferred + need > max_transfer_bytes
                and not approve_large
            ):
                raise RuntimeError(
                    f"Transfer budget exceeded before finding {member_name} "
                    f"(transferred={transferred})."
                )
            hdr = _read_exact(stream, BLOCK)
            if len(hdr) < BLOCK:
                raise RuntimeError(f"EOF before finding member {member_name}")
            transferred += len(hdr)
            info = parse_ustar_header(hdr)
            if info.get("raw_empty"):
                empty += 1
                if empty >= 2:
                    raise RuntimeError(f"Member not found: {member_name}")
                continue
            empty = 0
            name = info["name"].lstrip("./")
            size = int(info["size"])
            padded = padded_size(size)
            match = name == target or Path(name).name == target_base
            if match and info["typeflag"] in ("0", "\0", ""):
                if (
                    max_transfer_bytes is not None
                    and transferred + padded > max_transfer_bytes
                    and not approve_large
                ):
                    raise RuntimeError(
                        "Member body would exceed max_transfer_bytes without approval."
                    )
                print(
                    f"Found {name} size={format_bytes(size)} "
                    f"(prefix transferred {format_bytes(transferred)})"
                )
                member_fh = _TarMemberStream(stream, size)

                def _progress(info: dict) -> None:
                    n = info.get("lines_parsed") or 0
                    rate = info.get("lines_per_s") or 0.0
                    elapsed = info.get("elapsed_s") or 0.0
                    net = getattr(stream, "bytes_read", transferred)
                    print(
                        f"  progress lines={n:,} rate={rate:,.0f}/s "
                        f"elapsed={elapsed/60:.1f}m net={format_bytes(net)}",
                        flush=True,
                    )

                print(
                    f"Sampling with workers={workers} (gzip sequential, JSON parse parallel)"
                )
                rows, stats = sample_from_jsonl_gz_stream(
                    member_fh,
                    sample_size=sample_size,
                    seed=seed,
                    workers=workers,
                    progress_fn=_progress,
                    executor=executor,
                )
                # Ensure padding consumed
                member_fh.close()
                transferred += padded
                stats["transferred_bytes"] = getattr(stream, "bytes_read", transferred)
                stats["member"] = name
                stats["member_size"] = size
                stats["workers"] = workers
                return rows, stats

            # discard body
            if (
                max_transfer_bytes is not None
                and transferred + padded > max_transfer_bytes
                and not approve_large
            ):
                raise RuntimeError(
                    f"Discarding preceding member {name} would exceed budget "
                    f"(need {format_bytes(padded)} more)."
                )
            transferred += discard_bytes(stream, padded)
            print(f"  skip {name} ({format_bytes(size)}) cumulative={format_bytes(transferred)}")
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print plan only (default)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually stream TAR and write sample parquet",
    )
    parser.add_argument("--size", type=int, default=None, help="Sample size (default from config)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default from config)")
    parser.add_argument(
        "--i-approve-large-download",
        action="store_true",
        help="Required with --execute (multi-GB network transfer)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=86400.0,
        help="HTTP connect timeout / overall bound (read timeout disabled)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="JSON parse workers (default config n_workers, cap 6)",
    )
    parser.add_argument(
        "--max-transfer-bytes",
        type=int,
        default=None,
        help="Optional hard transfer cap (defaults to estimate gate only)",
    )
    args = parser.parse_args()

    cfg = load_config()
    paths = ensure_dirs(cfg)
    size = int(args.size or cfg.get("negative_sample_size") or 200_000)
    seed = int(args.seed if args.seed is not None else cfg.get("random_seed", 42))
    urls = cfg.get("urls") or {}
    tar_url = urls.get(
        "half_year_dataset_tar",
        "http://65.21.203.147:48102/half_year_dataset.tar",
    )
    tm = cfg.get("tar_members") or {}
    member = (
        tm.get("negatives_deploy_jsonl")
        or tm.get("negatives_deploy_txs")
        or "not_bought_deploy_txs.jsonl.gz"
    )
    workers = int(args.workers if args.workers is not None else cfg.get("n_workers") or 6)
    workers = max(1, min(workers, 6))

    estimate = estimate_negative_tar_transfer_bytes(cfg)
    plan = describe_sampling_plan(
        sample_size=size,
        seed=seed,
        source=f"{tar_url}#{member}",
    )
    out_path = _output_path(paths["samples"], size)
    limit = max_auto_download_bytes(cfg)

    print("=== Negative sampling ===")
    print(f"size={size} seed={seed} workers={workers}")
    print(f"output={out_path}")
    print(f"member={member}")
    print(f"auto_download_limit={format_bytes(limit)}")
    print("\nTransfer estimate:")
    for k, v in estimate.items():
        if isinstance(v, int):
            print(f"  {k}: {format_bytes(v)}")
        else:
            print(f"  {k}: {v}")
    print("\nStrategy:")
    for note in plan["notes"]:
        print(f"  - {note}")
    print(f"  approach={plan['approach']}")
    print(f"  columns={plan['output_columns']}")

    meta_out = paths["metadata"] / "negative_sample_plan.json"
    write_json(
        meta_out,
        {
            "plan": plan,
            "estimate": estimate,
            "output": str(out_path),
            "execute_requested": bool(args.execute),
        },
    )
    print(f"\nWrote plan → {meta_out}")

    if not args.execute:
        print(
            "\nDRY-RUN only (default). To execute, re-run with:\n"
            "  python scripts/sample_negatives.py --execute "
            "--i-approve-large-download [--size N] [--seed S]\n"
            "WARNING: worst-case network transfer is many GB even though the "
            "on-disk sample is small."
        )
        return 0

    if not args.i_approve_large_download:
        print(
            "\nRefusing --execute without --i-approve-large-download "
            f"(estimated worst-case {format_bytes(estimate['worst_case_stream_bytes'])} "
            f"> limit {format_bytes(limit)})."
        )
        return 2

    worst = int(estimate["worst_case_stream_bytes"])
    if worst > limit and not args.i_approve_large_download:
        return 2

    print("\n=== EXECUTE: streaming TAR (this will take a long time / much bandwidth) ===")
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    ctx = mp.get_context("forkserver")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        from src.sampling.negative_sample import _parse_line_batch

        executor.submit(_parse_line_batch, []).result()  # start workers before HTTP
        rows, stats = stream_sample_from_tar(
            tar_url,
            member,
            sample_size=size,
            seed=seed,
            max_transfer_bytes=args.max_transfer_bytes,
            approve_large=True,
            timeout=args.timeout,
            workers=workers,
            executor=executor,
        )
    dest = rows_to_parquet(rows, out_path)
    print(f"Sampled {stats.get('rows_sampled')} / requested {size}")
    print(f"Lines parsed: {stats.get('lines_parsed')}")
    print(f"Transferred: {format_bytes(stats.get('transferred_bytes'))}")
    print(f"Wrote {dest}")
    write_json(
        paths["metadata"] / "negative_sample_result.json",
        {"stats": stats, "output": str(dest)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
