#!/usr/bin/env python3
"""Probe dataset and wallet HTTP servers (HEAD + small Range only)."""

from __future__ import annotations

import argparse
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import ensure_dirs, load_config, write_json  # noqa: E402
from src.download.http_utils import head_request, range_probe  # noqa: E402


class _HrefParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k.lower() == "href" and v:
                self.hrefs.append(v)


def fetch_listing(url: str, timeout: float = 60.0) -> dict:
    headers = {"User-Agent": "solana-sniper/probe"}
    text = ""
    status = None
    error = None
    try:
        try:
            import httpx

            with httpx.Client(follow_redirects=True, timeout=timeout) as client:
                resp = client.get(url, headers=headers)
                status = resp.status_code
                text = resp.text[:200_000]
        except ImportError:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                text = resp.read(200_000).decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    files: list[str] = []
    if text:
        parser = _HrefParser()
        try:
            parser.feed(text)
            for href in parser.hrefs:
                if href in ("../", "./") or href.startswith("?"):
                    continue
                files.append(href.rstrip("/"))
        except Exception:  # noqa: BLE001
            # Fallback regex
            files = re.findall(r'href=["\']([^"\']+)["\']', text, flags=re.I)

    # Unique preserve order
    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            seen.add(f)
            uniq.append(f)

    return {"url": url, "status": status, "files": uniq, "error": error, "snippet": text[:2000]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    cfg = load_config()
    paths = ensure_dirs(cfg)
    urls = cfg.get("urls") or {}

    tar_base = urls.get("half_year_base", "http://65.21.203.147:48102/")
    tar_url = urls.get(
        "half_year_dataset_tar", tar_base.rstrip("/") + "/half_year_dataset.tar"
    )
    wallet_base = urls.get("wallet_base", "http://154.12.118.112:48114/")

    individual = list(urls.get("bought_individual") or []) + list(
        urls.get("not_bought_individual") or []
    )

    result: dict = {
        "dataset_server": {},
        "individual_urls": [],
        "wallet_server": {},
    }

    print("=== Dataset base listing ===")
    listing = fetch_listing(tar_base, timeout=args.timeout)
    print(f"  status={listing['status']} files={listing['files']}")

    print("=== half_year_dataset.tar HEAD ===")
    tar_head = head_request(tar_url, timeout=args.timeout)
    print(
        f"  status={tar_head['status']} length={tar_head['content_length']} "
        f"accept_ranges={tar_head['accept_ranges']}"
    )
    print("=== half_year_dataset.tar Range probe ===")
    tar_range = range_probe(tar_url, 0, 1023, timeout=args.timeout)
    print(
        f"  status={tar_range['status']} content_range={tar_range['content_range']} "
        f"body_len={tar_range['body_len']}"
    )

    result["dataset_server"] = {
        "base": tar_base,
        "listing": listing,
        "tar": {"url": tar_url, "head": tar_head, "range_probe": tar_range},
    }

    print("=== Individual bought/not_bought URLs (expect 404) ===")
    for u in individual:
        h = head_request(u, timeout=args.timeout)
        r = range_probe(u, 0, 64, timeout=args.timeout)
        print(f"  {u} -> HEAD {h['status']} RANGE {r['status']}")
        result["individual_urls"].append({"url": u, "head": h, "range_probe": r})

    print("=== Wallet server listing ===")
    w_listing = fetch_listing(wallet_base, timeout=args.timeout)
    print(f"  status={w_listing['status']} files={w_listing['files']}")

    wallet_files = []
    for name in w_listing.get("files") or []:
        if name.endswith("/"):
            continue
        file_url = wallet_base.rstrip("/") + "/" + name.lstrip("/")
        print(f"--- {name} ---")
        h = head_request(file_url, timeout=args.timeout)
        r = range_probe(file_url, 0, 1023, timeout=args.timeout)
        print(
            f"  HEAD status={h['status']} len={h['content_length']} "
            f"accept_ranges={h['accept_ranges']}"
        )
        print(
            f"  RANGE status={r['status']} content_range={r['content_range']} "
            f"body_len={r['body_len']}"
        )
        wallet_files.append({"name": name, "url": file_url, "head": h, "range_probe": r})

    # Also probe configured wallet filenames if listing empty
    if not wallet_files:
        wf = urls.get("wallet_files") or {}
        for key, name in wf.items():
            file_url = wallet_base.rstrip("/") + "/" + name
            h = head_request(file_url, timeout=args.timeout)
            r = range_probe(file_url, 0, 1023, timeout=args.timeout)
            wallet_files.append(
                {"name": name, "key": key, "url": file_url, "head": h, "range_probe": r}
            )

    result["wallet_server"] = {
        "base": wallet_base,
        "listing": w_listing,
        "files": wallet_files,
    }

    out = paths["metadata"] / "server_probe.json"
    write_json(out, result)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
