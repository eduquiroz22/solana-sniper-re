"""HTTP helpers with size gates for safe downloads."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

USER_AGENT = "solana-sniper/phase1 (+local research)"


def _client_available() -> str:
    try:
        import httpx  # noqa: F401

        return "httpx"
    except ImportError:
        return "urllib"


def head_request(url: str, timeout: float = 60.0) -> dict[str, Any]:
    """
    HEAD request. Returns dict with status, headers, content_length, accept_ranges.
    Falls back to GET with Range bytes=0-0 if HEAD fails.
    """
    headers_out: dict[str, str] = {}
    status = None
    error = None

    try:
        import httpx

        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            resp = client.head(url, headers={"User-Agent": USER_AGENT})
            status = resp.status_code
            headers_out = {k.lower(): v for k, v in resp.headers.items()}
            if status in (405, 501) or (status >= 400 and status != 404):
                # Some servers mishandle HEAD; try tiny GET
                resp2 = client.get(
                    url,
                    headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"},
                )
                status = resp2.status_code
                headers_out = {k.lower(): v for k, v in resp2.headers.items()}
    except ImportError:
        try:
            req = Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                headers_out = {k.lower(): v for k, v in resp.headers.items()}
        except (HTTPError, URLError, TypeError, ValueError) as exc:
            error = str(exc)
            try:
                req = Request(
                    url,
                    headers={"User-Agent": USER_AGENT, "Range": "bytes=0-0"},
                )
                with urlopen(req, timeout=timeout) as resp:
                    status = getattr(resp, "status", None) or resp.getcode()
                    headers_out = {k.lower(): v for k, v in resp.headers.items()}
                    error = None
            except Exception as exc2:  # noqa: BLE001
                error = f"{error}; fallback GET failed: {exc2}"
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    content_length = None
    cl = headers_out.get("content-length")
    if cl is not None:
        try:
            content_length = int(cl)
        except ValueError:
            content_length = None

    # Content-Range: bytes 0-0/TOTAL
    cr = headers_out.get("content-range")
    if content_length is None and cr:
        try:
            total = cr.split("/")[-1]
            if total != "*":
                content_length = int(total)
        except ValueError:
            pass

    accept = headers_out.get("accept-ranges", "")
    accept_ranges = bool(accept and accept.lower() != "none")

    return {
        "url": url,
        "status": status,
        "headers": headers_out,
        "content_length": content_length,
        "accept_ranges": accept_ranges,
        "error": error,
        "client": _client_available(),
    }


def range_probe(
    url: str, start: int = 0, end: int = 1023, timeout: float = 60.0
) -> dict[str, Any]:
    """GET with Range header; report status, content_range, and body length."""
    headers = {"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
    body = b""
    status = None
    headers_out: dict[str, str] = {}
    error = None

    try:
        import httpx

        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            with client.stream("GET", url, headers=headers) as resp:
                status = resp.status_code
                headers_out = {k.lower(): v for k, v in resp.headers.items()}
                # Cap read: requested range size + small slack
                limit = (end - start + 1) + 4096
                for chunk in resp.iter_bytes():
                    body += chunk
                    if len(body) >= limit:
                        body = body[:limit]
                        break
    except ImportError:
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                headers_out = {k.lower(): v for k, v in resp.headers.items()}
                limit = (end - start + 1) + 4096
                while len(body) < limit:
                    chunk = resp.read(min(65536, limit - len(body)))
                    if not chunk:
                        break
                    body += chunk
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    return {
        "url": url,
        "status": status,
        "content_range": headers_out.get("content-range"),
        "content_length": _safe_int(headers_out.get("content-length")),
        "body_len": len(body),
        "accept_ranges": bool(
            (headers_out.get("accept-ranges") or "").lower() not in ("", "none")
        )
        or status == 206,
        "headers": headers_out,
        "error": error,
    }


def _safe_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def download_file(
    url: str,
    dest: str | Path,
    *,
    max_bytes: int | None = None,
    resume: bool = True,
    approve_large: bool = False,
    chunk_size: int = 1024 * 1024,
    timeout: float = 120.0,
    progress: bool = True,
) -> dict[str, Any]:
    """
    Stream-download url to dest.

    If Content-Length (or remaining bytes) exceeds max_bytes and approve_large is
    False, abort before transferring. Supports resume via HTTP Range when possible.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    head = head_request(url, timeout=timeout)
    remote_size = head.get("content_length")
    existing = dest.stat().st_size if dest.is_file() else 0

    if remote_size is not None and max_bytes is not None:
        remaining = remote_size - existing if resume and existing < remote_size else remote_size
        if remaining > max_bytes and not approve_large:
            raise RuntimeError(
                f"Content-Length/remaining {remaining} bytes exceeds max_bytes={max_bytes}. "
                "Pass approve_large=True / --i-approve-large-download."
            )

    if resume and existing and remote_size is not None and existing == remote_size:
        return {
            "url": url,
            "dest": str(dest),
            "bytes_written": 0,
            "total_size": existing,
            "resumed": True,
            "skipped": True,
            "sha256": None,
        }

    start_at = existing if resume and existing > 0 else 0
    headers = {"User-Agent": USER_AGENT}
    mode = "ab" if start_at else "wb"
    if start_at:
        headers["Range"] = f"bytes={start_at}-"

    bytes_written = 0
    t0 = time.time()

    def _report(done: int, total: int | None) -> None:
        if not progress:
            return
        if total:
            pct = 100.0 * done / total
            msg = f"\r  {dest.name}: {done:,}/{total:,} ({pct:.1f}%)"
        else:
            msg = f"\r  {dest.name}: {done:,} bytes"
        sys.stderr.write(msg)
        sys.stderr.flush()

    try:
        import httpx

        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            with client.stream("GET", url, headers=headers) as resp:
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code} for {url}")
                # If server ignored Range and sent full body, truncate and restart
                if start_at and resp.status_code == 200:
                    mode = "wb"
                    start_at = 0
                    bytes_written = 0
                total_hint = remote_size
                cl = resp.headers.get("content-length")
                if cl and start_at == 0:
                    total_hint = int(cl)
                with dest.open(mode) as out:
                    for chunk in resp.iter_bytes(chunk_size=chunk_size):
                        if max_bytes is not None and not approve_large:
                            # Hard stop if we somehow exceed mid-stream without CL
                            if start_at + bytes_written + len(chunk) > max_bytes and (
                                remote_size is None or remote_size > max_bytes
                            ):
                                raise RuntimeError(
                                    "Download exceeded max_bytes mid-stream; aborting."
                                )
                        out.write(chunk)
                        bytes_written += len(chunk)
                        _report(start_at + bytes_written, total_hint)
    except ImportError:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            if start_at and status == 200:
                mode = "wb"
                start_at = 0
                bytes_written = 0
            total_hint = remote_size
            cl = resp.headers.get("Content-Length") or resp.headers.get("content-length")
            if cl and start_at == 0:
                try:
                    total_hint = int(cl)
                except ValueError:
                    pass
            with dest.open(mode) as out:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    bytes_written += len(chunk)
                    _report(start_at + bytes_written, total_hint)
    finally:
        if progress:
            sys.stderr.write("\n")

    elapsed = time.time() - t0
    final_size = dest.stat().st_size
    return {
        "url": url,
        "dest": str(dest),
        "bytes_written": bytes_written,
        "total_size": final_size,
        "elapsed_sec": round(elapsed, 3),
        "resumed": bool(start_at),
        "skipped": False,
    }
