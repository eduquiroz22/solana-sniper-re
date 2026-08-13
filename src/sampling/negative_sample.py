"""Stratified temporal negative sampling from streaming gzip JSONL deploy txs."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterator

OUTPUT_COLUMNS = [
    "line_number",
    "token_address",
    "creator_address",
    "tx_signer",
    "tx_hash",
    "blockTime",
    "blockSlot",
    "sample_seed",
    "sample_size",
]


def _dig(obj: Any, *paths: tuple[str, ...]) -> Any:
    for path in paths:
        cur = obj
        ok = True
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok and cur is not None:
            return cur
    return None


def extract_deploy_fields(obj: dict, line_number: int) -> dict[str, Any]:
    """
    Defensively extract fields from Solana jsonParsed tx or flat index-like dicts.
    """
    # Flat / index-style
    token = obj.get("token_address") or obj.get("tokenAddress") or obj.get("mint")
    creator = obj.get("creator_address") or obj.get("creatorAddress") or obj.get("creator")
    tx_signer = obj.get("tx_signer") or obj.get("signer") or obj.get("feePayer")
    tx_hash = obj.get("tx_hash") or obj.get("signature") or obj.get("txid")
    block_time = obj.get("blockTime") or obj.get("block_time") or obj.get("timestamp")
    block_slot = obj.get("blockSlot") or obj.get("block_slot") or obj.get("slot")

    # Nested transaction.result / transaction structure
    if tx_hash is None:
        tx_hash = _dig(
            obj,
            ("transaction", "signatures"),
            ("result", "transaction", "signatures"),
            ("meta", "signature"),
        )
        if isinstance(tx_hash, list) and tx_hash:
            tx_hash = tx_hash[0]

    if block_time is None:
        block_time = _dig(
            obj,
            ("blockTime",),
            ("result", "blockTime"),
            ("transaction", "blockTime"),
        )
    if block_slot is None:
        block_slot = _dig(
            obj,
            ("slot",),
            ("blockSlot",),
            ("result", "slot"),
            ("transaction", "slot"),
        )

    # message accountKeys / instructions for mint / creator heuristics
    message = _dig(
        obj,
        ("transaction", "message"),
        ("result", "transaction", "message"),
        ("transaction", "transaction", "message"),
    )
    if isinstance(message, dict):
        if tx_signer is None:
            keys = message.get("accountKeys") or message.get("accounts")
            if isinstance(keys, list) and keys:
                first = keys[0]
                if isinstance(first, dict):
                    tx_signer = first.get("pubkey") or first.get("address")
                else:
                    tx_signer = first
        if token is None or creator is None:
            instructions = message.get("instructions") or []
            for ix in instructions:
                if not isinstance(ix, dict):
                    continue
                parsed = ix.get("parsed")
                if not isinstance(parsed, dict):
                    continue
                info = parsed.get("info") if isinstance(parsed.get("info"), dict) else {}
                ptype = parsed.get("type") or ""
                if token is None:
                    token = info.get("mint") or info.get("tokenAddress")
                    if token is None and isinstance(info.get("extensions"), dict):
                        token = info["extensions"].get("mint")
                if creator is None and ptype:
                    creator = (
                        info.get("mintAuthority")
                        or info.get("authority")
                        or info.get("creator")
                    )

    # postTokenBalances mint
    if token is None:
        meta = _dig(obj, ("meta",), ("result", "meta"), ("transaction", "meta"))
        if isinstance(meta, dict):
            for bal in meta.get("postTokenBalances") or []:
                if isinstance(bal, dict) and bal.get("mint"):
                    token = bal["mint"]
                    break

    try:
        block_time_i = int(block_time) if block_time is not None else None
    except (TypeError, ValueError):
        block_time_i = None
    try:
        block_slot_i = int(block_slot) if block_slot is not None else None
    except (TypeError, ValueError):
        block_slot_i = None

    return {
        "line_number": line_number,
        "token_address": token,
        "creator_address": creator,
        "tx_signer": tx_signer,
        "tx_hash": tx_hash,
        "blockTime": block_time_i,
        "blockSlot": block_slot_i,
    }


def week_bucket(block_time: int | None) -> str:
    """ISO week bucket string for stratification; 'unknown' if missing."""
    if block_time is None:
        return "unknown"
    try:
        dt = datetime.fromtimestamp(int(block_time), tz=timezone.utc)
        iso = dt.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    except (OverflowError, OSError, ValueError):
        return "unknown"


def _stable_unit_interval(seed: int, line_number: int, salt: str = "") -> float:
    """Deterministic U[0,1) from seed + line_number (+ optional salt)."""
    payload = f"{seed}|{line_number}|{salt}".encode()
    digest = hashlib.sha256(payload).digest()
    # Use 8 bytes -> [0, 1)
    n = int.from_bytes(digest[:8], "big")
    return n / 2**64


def describe_sampling_plan(
    *,
    sample_size: int,
    seed: int,
    source: str,
    approach: str = "one_pass_hash_stratified",
) -> dict[str, Any]:
    """
    Document the dry-run sampling strategy.

    Approach (one-pass streaming):
      1. Stream gzip JSONL (optionally as a TAR member).
      2. For each line i, parse blockTime defensively; assign week_bucket.
      3. Inclusion via hash(seed, i): keep if unit < sample_size / N_hat.
         Without knowing N, use reservoir stratified by week:
         - Maintain up to k_b slots per bucket with capacity proportional to
           observed bucket mass so far, OR global reservoir of size sample_size
           with key = hash(seed, line) for reproducibility.
      4. Preferred reproducible one-pass method implemented below:
         **global hash-keyed reservoir** of size sample_size, then optionally
         rebalance is skipped (approximation). Stratification: within each week
         bucket, keep items whose hash rank is among the top
         ceil(sample_size * count_b / total) as counts grow — implemented as
         per-bucket reservoirs with dynamic caps that sum to sample_size.

    Two-pass ideal (not default for multi-GB network streams):
      Pass 1: count lines per week (still full transfer).
      Pass 2: sample with exact quotas. Same network cost twice — avoided.
    """
    return {
        "mode": "plan",
        "approach": approach,
        "sample_size": sample_size,
        "seed": seed,
        "source": source,
        "output_columns": OUTPUT_COLUMNS,
        "notes": [
            "Default CLI is dry-run: prints this plan and transfer estimate only.",
            "Execute streams not_bought_deploy_txs.jsonl.gz via TAR (many GB network).",
            "Requires --i-approve-large-download because transfer >> 1 GiB.",
            "Reproducibility: hash(seed, line_number) selects reservoir keys.",
            "Stratification: per-ISO-week reservoirs; caps rebalanced to sum to N.",
        ],
    }


class StratifiedHashReservoir:
    """
    One-pass stratified reservoir using hash(seed, line_number) as rank key.

    Each week bucket keeps the lowest-rank items in a bounded heap. Caps are
    rebalanced periodically so they sum to sample_size proportional to observed
    counts (approximation when streaming once).
    """

    def __init__(self, sample_size: int, seed: int, rebalance_every: int = 50_000):
        import heapq

        self._heapq = heapq
        self.sample_size = int(sample_size)
        self.seed = int(seed)
        self.rebalance_every = rebalance_every
        self.bucket_counts: dict[str, int] = {}
        # max-heap via negated rank so we can pop worst (highest rank) quickly
        self.bucket_heaps: dict[str, list[tuple[float, int, dict]]] = {}
        self.total = 0
        self._caps: dict[str, int] = {}

    def _rebalance_caps(self) -> dict[str, int]:
        total = max(self.total, 1)
        buckets = list(self.bucket_counts.keys())
        if not buckets:
            return {}
        raw = {
            b: self.sample_size * (self.bucket_counts[b] / total) for b in buckets
        }
        caps = {b: max(1, int(math.floor(v))) for b, v in raw.items()}
        while sum(caps.values()) > self.sample_size:
            b = max(caps, key=lambda k: caps[k])
            if caps[b] > 1:
                caps[b] -= 1
            else:
                break
        while sum(caps.values()) < self.sample_size:
            b = max(buckets, key=lambda k: raw[k] - math.floor(raw[k]))
            caps[b] = caps.get(b, 0) + 1
        return caps

    def _trim_bucket(self, bucket: str, cap: int) -> None:
        heap = self.bucket_heaps.setdefault(bucket, [])
        while len(heap) > cap:
            self._heapq.heappop(heap)

    def consider(self, row: dict) -> None:
        self.total += 1
        bucket = week_bucket(row.get("blockTime"))
        self.bucket_counts[bucket] = self.bucket_counts.get(bucket, 0) + 1
        rank = _stable_unit_interval(self.seed, int(row["line_number"]), salt=bucket)
        heap = self.bucket_heaps.setdefault(bucket, [])

        if self.total == 1 or self.total % self.rebalance_every == 0:
            self._caps = self._rebalance_caps()
            for b, cap in self._caps.items():
                self._trim_bucket(b, cap)

        cap = self._caps.get(bucket) or max(
            1, self.sample_size // max(len(self.bucket_counts), 1)
        )
        # Store as max-heap on rank: keep lowest ranks → push (-rank)
        item = (-rank, int(row["line_number"]), row)
        if len(heap) < cap:
            self._heapq.heappush(heap, item)
        else:
            # If better (lower rank) than current worst, replace
            if item[0] > heap[0][0]:  # -rank greater means rank smaller
                self._heapq.heapreplace(heap, item)

    def finalize(self) -> list[dict]:
        self._caps = self._rebalance_caps()
        for b, cap in self._caps.items():
            self._trim_bucket(b, cap)
        out: list[dict] = []
        for b, heap in self.bucket_heaps.items():
            for neg_rank, _line, row in heap:
                enriched = dict(row)
                enriched["sample_seed"] = self.seed
                enriched["sample_size"] = self.sample_size
                enriched["week_bucket"] = b
                enriched["_rank"] = -neg_rank
                out.append(enriched)
        if len(out) > self.sample_size:
            out.sort(key=lambda r: r["_rank"])
            out = out[: self.sample_size]
        for r in out:
            r.pop("_rank", None)
        out.sort(
            key=lambda r: (
                r.get("blockTime") is None,
                r.get("blockTime") or 0,
                r["line_number"],
            )
        )
        return out


def _loads(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _parse_line_batch(items: list[tuple[int, str]]) -> list[dict]:
    """Worker: parse a batch of (line_number, raw_line) into extracted rows."""
    out: list[dict] = []
    for line_no, line in items:
        obj = _loads(line)
        if obj is None:
            continue
        out.append(extract_deploy_fields(obj, line_no))
    return out


def iter_jsonl_gz_raw_lines(fh: BinaryIO) -> Iterator[tuple[int, str]]:
    """Yield (line_number 1-based, raw line) from a binary gzip stream."""
    with gzip.GzipFile(fileobj=fh, mode="rb") as gz:
        text = _TextWrapper(gz)
        for i, line in enumerate(text, start=1):
            if line.strip():
                yield i, line


def iter_jsonl_gz_lines(fh: BinaryIO) -> Iterator[tuple[int, dict]]:
    """Yield (line_number 1-based, obj) from a binary gzip stream."""
    for i, line in iter_jsonl_gz_raw_lines(fh):
        obj = _loads(line)
        if obj is not None:
            yield i, obj


class _TextWrapper:
    def __init__(self, binary_fh: BinaryIO):
        self._fh = binary_fh
        self._buf = b""

    def __iter__(self) -> Iterator[str]:
        while True:
            chunk = self._fh.read(1024 * 1024)
            if not chunk:
                if self._buf:
                    yield self._buf.decode("utf-8", errors="replace")
                    self._buf = b""
                break
            self._buf += chunk
            while True:
                idx = self._buf.find(b"\n")
                if idx < 0:
                    break
                line = self._buf[:idx]
                self._buf = self._buf[idx + 1 :]
                yield line.decode("utf-8", errors="replace")


def sample_from_jsonl_gz_stream(
    binary_stream: BinaryIO,
    *,
    sample_size: int,
    seed: int,
    max_lines: int | None = None,
    workers: int = 1,
    batch_size: int = 512,
    max_inflight: int = 8,
    progress_every: int = 25_000,
    progress_fn: Any | None = None,
    executor: Any | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Run stratified hash reservoir over a gzip jsonl binary stream.

    TAR/gzip remain sequential. ``workers>1`` parallelizes JSON parse + field
    extract (capped; laptop default 6). Reservoir updates stay in this process.
    Pass a pre-started ``executor`` (created before opening HTTP) to avoid
    forking after httpx threads.
    """
    import time

    workers = max(1, int(workers))
    sampler = StratifiedHashReservoir(sample_size, seed)
    parsed = 0
    t0 = time.monotonic()

    def _on_progress() -> None:
        if progress_fn is None:
            return
        elapsed = max(time.monotonic() - t0, 1e-6)
        progress_fn(
            {
                "lines_parsed": parsed,
                "elapsed_s": elapsed,
                "lines_per_s": parsed / elapsed,
            }
        )

    if workers == 1 and executor is None:
        for line_no, obj in iter_jsonl_gz_lines(binary_stream):
            sampler.consider(extract_deploy_fields(obj, line_no))
            parsed += 1
            if progress_every and parsed % progress_every == 0:
                _on_progress()
            if max_lines is not None and parsed >= max_lines:
                break
    else:
        from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
        import multiprocessing as mp

        inflight: list = []
        batch: list[tuple[int, str]] = []
        pool_workers = min(workers, 6)
        own_pool = executor is None
        pool = executor
        if pool is None:
            ctx = mp.get_context("forkserver")
            pool = ProcessPoolExecutor(max_workers=pool_workers, mp_context=ctx)

        def _drain_one(block: bool) -> None:
            nonlocal parsed
            if not inflight:
                return
            if block:
                done, _ = wait(inflight, return_when=FIRST_COMPLETED)
            else:
                done = {f for f in inflight if f.done()}
            for fut in done:
                inflight.remove(fut)
                for row in fut.result():
                    sampler.consider(row)
                    parsed += 1
                    if progress_every and parsed % progress_every == 0:
                        _on_progress()

        try:
            for item in iter_jsonl_gz_raw_lines(binary_stream):
                batch.append(item)
                if len(batch) >= batch_size:
                    while len(inflight) >= max_inflight:
                        _drain_one(block=True)
                    inflight.append(pool.submit(_parse_line_batch, batch))
                    batch = []
                    _drain_one(block=False)
                if max_lines is not None and parsed >= max_lines:
                    break
            if batch and (max_lines is None or parsed < max_lines):
                inflight.append(pool.submit(_parse_line_batch, batch))
            while inflight:
                _drain_one(block=True)
                if max_lines is not None and parsed >= max_lines:
                    break
        finally:
            if own_pool and pool is not None:
                pool.shutdown(wait=True)

    _on_progress()
    rows = sampler.finalize()
    stats = {
        "lines_parsed": parsed,
        "rows_sampled": len(rows),
        "bucket_counts": dict(sorted(sampler.bucket_counts.items())),
        "seed": seed,
        "sample_size": sample_size,
        "workers": workers,
        "elapsed_s": time.monotonic() - t0,
    }
    return rows, stats


def _scalar_str(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    if isinstance(v, (list, dict, tuple)):
        return json.dumps(v, default=str)
    return str(v)


def _scalar_int(v: Any) -> int | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def rows_to_parquet(rows: list[dict], dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    str_cols = (
        "token_address",
        "creator_address",
        "tx_signer",
        "tx_hash",
    )
    int_cols = ("line_number", "blockTime", "blockSlot", "sample_seed", "sample_size")
    normalized = []
    for r in rows:
        rec: dict[str, Any] = {}
        for c in OUTPUT_COLUMNS:
            if c in str_cols:
                rec[c] = _scalar_str(r.get(c))
            elif c in int_cols:
                rec[c] = _scalar_int(r.get(c))
            else:
                rec[c] = r.get(c)
        normalized.append(rec)

    # Durable sidecar so a parquet schema glitch does not lose a 14 GiB stream.
    jsonl_path = dest.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for rec in normalized:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    try:
        import polars as pl

        schema = {c: pl.Utf8 for c in str_cols}
        schema.update({c: pl.Int64 for c in int_cols})
        pl.DataFrame(normalized, schema=schema, infer_schema_length=0).write_parquet(dest)
        return dest
    except Exception as exc:  # noqa: BLE001
        print(f"polars parquet write failed ({exc}); trying pyarrow", flush=True)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(
            normalized,
            schema=pa.schema(
                [(c, pa.string()) for c in str_cols]
                + [(c, pa.int64()) for c in int_cols]
            ),
        )
        pq.write_table(table, dest)
        return dest
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Failed to write parquet ({exc}). Rows saved at {jsonl_path}"
        ) from exc


def estimate_negative_tar_transfer_bytes(cfg: dict | None = None) -> dict[str, Any]:
    """
    Rough transfer estimate to reach not_bought_deploy_txs.jsonl.gz in the TAR.

    Without TOC, use documented sizes: positives ~677 MiB then negatives jsonl
    ~13.9 GiB fully streamed. Sampling can stop early only after enough lines;
    worst case ~14+ GiB network.
    """
    positives_est = 710_000_000
    not_bought_jsonl_est = 14_900_000_000  # ~13.9 GiB compressed estimate
    if cfg:
        positives_est = int(
            (cfg.get("tar_members") or {}).get("positives_estimate_bytes", positives_est)
        )
    return {
        "positives_prefix_estimate_bytes": positives_est,
        "not_bought_deploy_txs_jsonl_gz_estimate_bytes": not_bought_jsonl_est,
        "worst_case_stream_bytes": positives_est + not_bought_jsonl_est,
        "note": (
            "TAR has no HTTP Range; streaming not_bought jsonl requires "
            "transferring (and discarding) all preceding members plus the member body."
        ),
    }
