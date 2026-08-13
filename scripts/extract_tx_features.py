#!/usr/bin/env python3
"""Attach t_decision tx-structure features to positives (local) or sampled negatives (TAR)."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import ensure_dirs, format_bytes, load_config  # noqa: E402
from src.features.tx_extract import extract_tx_features  # noqa: E402
from src.download.tar_stream import (  # noqa: E402
    BLOCK,
    discard_bytes,
    padded_size,
    parse_ustar_header,
)


def enrich_positives(dest: Path) -> Path:
    import polars as pl

    src = ROOT / "data/raw/positives/bought_deploy_txs.jsonl.gz"
    rows = []
    with gzip.open(src, "rt", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(extract_tx_features(obj, i))
            if i % 2000 == 0:
                print(f"  positives parsed {i:,}", flush=True)
    df = pl.DataFrame(rows)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dest)
    print(f"Wrote {dest} rows={df.height}")
    return dest


def _open_url(url: str, timeout: float):
    import httpx

    client = httpx.Client(follow_redirects=True, timeout=httpx.Timeout(timeout, read=None))
    req = client.build_request("GET", url, headers={"User-Agent": "solana-sniper/enrich"})
    resp = client.send(req, stream=True)
    if resp.status_code >= 400:
        resp.close()
        client.close()
        raise RuntimeError(f"HTTP {resp.status_code}")

    class _R:
        def __init__(self):
            self._it = resp.iter_bytes()
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

    return _R()


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def _parse_batch(batch: list[tuple[int, str]]) -> list[dict]:
    from src.features.tx_extract import extract_tx_features as _ex

    out = []
    for line_no, text in batch:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            continue
        out.append(_ex(obj, line_no))
    return out


def enrich_negatives(dest: Path, workers: int = 6) -> Path:
    import gzip as gzmod
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

    import polars as pl

    cfg = load_config()
    paths = ensure_dirs(cfg)
    sample = pl.read_parquet(paths["samples"] / "negative_200k.parquet")
    wanted = set(int(x) for x in sample["line_number"].to_list())
    print(f"Wanted negative lines: {len(wanted):,}")

    tar_url = (cfg.get("urls") or {}).get("half_year_dataset_tar")
    member = (cfg.get("tar_members") or {}).get(
        "negatives_deploy_jsonl", "not_bought_deploy_txs.jsonl.gz"
    )
    target = member.lstrip("./")
    stream = _open_url(tar_url, timeout=86400.0)
    rows: list[dict] = []
    try:
        empty = 0
        while True:
            hdr = _read_exact(stream, BLOCK)
            if len(hdr) < BLOCK:
                raise RuntimeError("EOF before negative jsonl")
            info = parse_ustar_header(hdr)
            if info.get("raw_empty"):
                empty += 1
                if empty >= 2:
                    raise RuntimeError("member not found")
                continue
            empty = 0
            name = info["name"].lstrip("./")
            size = int(info["size"])
            padded = padded_size(size)
            match = name == target or Path(name).name == Path(target).name
            if not match:
                print(f"  skip {name} ({format_bytes(size)})", flush=True)
                discard_bytes(stream, padded)
                continue

            print(f"Found {name} size={format_bytes(size)}", flush=True)

            class _Member:
                def __init__(self):
                    self._left = size
                    self._pad = padded - size

                def read(self, n: int = -1) -> bytes:
                    if self._left <= 0:
                        return b""
                    if n < 0 or n > self._left:
                        n = self._left
                    data = _read_exact(stream, n)
                    self._left -= len(data)
                    if self._left <= 0 and self._pad:
                        discard_bytes(stream, self._pad)
                        self._pad = 0
                    return data

            raw = gzmod.GzipFile(fileobj=_Member())  # type: ignore[arg-type]
            ctx = mp.get_context("forkserver")
            batch: list[tuple[int, str]] = []
            inflight: list = []
            parsed = 0
            kept = 0
            with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx) as pool:
                for line_no, line in enumerate(raw, start=1):
                    parsed += 1
                    if line_no in wanted:
                        text = line.decode("utf-8", errors="replace").strip()
                        if text:
                            batch.append((line_no, text))
                    if len(batch) >= 256:
                        inflight.append(pool.submit(_parse_batch, batch))
                        batch = []
                        if len(inflight) >= 8:
                            done, pending = wait(inflight, return_when=FIRST_COMPLETED)
                            for fut in done:
                                part = fut.result()
                                rows.extend(part)
                                kept += len(part)
                            inflight = list(pending)
                    if parsed % 50_000 == 0:
                        print(
                            f"  scanned {parsed:,} kept={kept:,} "
                            f"net={format_bytes(getattr(stream, 'bytes_read', 0))}",
                            flush=True,
                        )
                    if kept >= len(wanted) and not batch and not inflight:
                        break
                if batch:
                    rows.extend(_parse_batch(batch))
                    kept += len(batch)
                for fut in inflight:
                    part = fut.result()
                    rows.extend(part)
                    kept += len(part)
            break
    finally:
        stream.close()

    df = pl.DataFrame(rows)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dest)
    print(f"Wrote {dest} rows={df.height}")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["positives", "negatives", "both"], default="both")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    out_dir = ROOT / "data/processed"
    if args.source in ("positives", "both"):
        enrich_positives(out_dir / "pos_tx_features.parquet")
    if args.source in ("negatives", "both"):
        enrich_negatives(out_dir / "neg_tx_features.parquet", workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
