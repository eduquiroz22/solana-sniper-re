"""Extract t_decision features from a Solana jsonParsed deploy transaction."""

from __future__ import annotations

import struct
from typing import Any

PUMP_FUN = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
COMPUTE_BUDGET = "ComputeBudget111111111111111111111111111111"
TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
TOKEN_2022 = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
ASSOC_TOKEN = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"

JITO_TIPS = {
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNmrzFZp",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8iR8m7xhmvFgvW8",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEj",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
}

# Paid launch / bundle relays seen in this dataset (prefix match).
SERVICE_PREFIXES = ("astra", "rapid", "moonX", "uxto", "devAA")

_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _b58decode(s: str) -> bytes:
    n = 0
    for ch in s:
        idx = _B58.find(ch)
        if idx < 0:
            return b""
        n = n * 58 + idx
    raw = n.to_bytes((n.bit_length() + 7) // 8 or 1, "big")
    pad = 0
    for ch in s:
        if ch == "1":
            pad += 1
        else:
            break
    return b"\x00" * pad + raw


def _pubkey(k: Any) -> str | None:
    if isinstance(k, dict):
        p = k.get("pubkey")
        return p if isinstance(p, str) else None
    return k if isinstance(k, str) else None


def _is_service_dest(dest: str) -> bool:
    if dest in JITO_TIPS:
        return True
    dlow = dest.lower()
    return any(dest.startswith(p) or dlow.startswith(p.lower()) for p in SERVICE_PREFIXES)


def extract_tx_features(obj: dict[str, Any], line_number: int | None = None) -> dict[str, Any]:
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    tx = obj.get("transaction") if isinstance(obj.get("transaction"), dict) else {}
    msg = tx.get("message") if isinstance(tx.get("message"), dict) else {}
    keys = msg.get("accountKeys") or []
    ixs = msg.get("instructions") or []
    lookups = msg.get("addressTableLookups") or []
    inner = meta.get("innerInstructions") or []
    logs = meta.get("logMessages") or []
    sigs = tx.get("signatures") or []

    n_signers = 0
    for k in keys:
        if isinstance(k, dict) and k.get("signer"):
            n_signers += 1
        elif not isinstance(k, dict) and n_signers == 0:
            n_signers = 1  # legacy string keys: fee payer first

    programs: list[str] = []
    cu_limit = None
    cu_price = None
    n_pump_ix = 0
    n_sol_xfer = 0
    tip_lamports = 0
    max_xfer = 0
    has_jito_tip = 0
    has_service_tip = 0

    for ix in ixs:
        if not isinstance(ix, dict):
            continue
        pid = ix.get("programId") or ix.get("program")
        if isinstance(pid, str):
            programs.append(pid)
            if pid == PUMP_FUN:
                n_pump_ix += 1
        if pid == COMPUTE_BUDGET or pid == "compute-budget":
            data = ix.get("data")
            if isinstance(data, str) and data:
                raw = _b58decode(data)
                if raw and raw[0] == 2 and len(raw) >= 5:
                    cu_limit = int(struct.unpack_from("<I", raw, 1)[0])
                elif raw and raw[0] == 3 and len(raw) >= 9:
                    cu_price = int(struct.unpack_from("<Q", raw, 1)[0])
        parsed = ix.get("parsed")
        if isinstance(parsed, dict) and parsed.get("type") == "transfer":
            info = parsed.get("info") if isinstance(parsed.get("info"), dict) else {}
            dest = info.get("destination")
            try:
                lamports = int(info.get("lamports") or 0)
            except (TypeError, ValueError):
                lamports = 0
            n_sol_xfer += 1
            if lamports > max_xfer:
                max_xfer = lamports
            if isinstance(dest, str):
                if dest in JITO_TIPS:
                    has_jito_tip = 1
                    tip_lamports += lamports
                elif _is_service_dest(dest):
                    has_service_tip = 1
                    tip_lamports += lamports

    n_inner = 0
    for block in inner:
        if isinstance(block, dict):
            n_inner += len(block.get("instructions") or [])

    log_blob = " ".join(str(x) for x in logs)
    has_pump = int(
        PUMP_FUN in programs
        or "Program 6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P" in log_blob
        or " pump" in log_blob.lower()
    )
    has_buy_same_tx = int(
        "Instruction: Buy" in log_blob or "BuyExactSolIn" in log_blob
    )
    has_create_v2 = int("CreateV2" in log_blob)
    has_token_2022 = int(TOKEN_2022 in programs or TOKEN_2022 in log_blob)

    cu = meta.get("computeUnitsConsumed")
    fee = meta.get("fee")
    try:
        cu_i = int(cu) if cu is not None else None
    except (TypeError, ValueError):
        cu_i = None
    try:
        fee_i = int(fee) if fee is not None else None
    except (TypeError, ValueError):
        fee_i = None

    tx_index = obj.get("transactionIndex")
    try:
        tx_index_i = int(tx_index) if tx_index is not None else None
    except (TypeError, ValueError):
        tx_index_i = None

    pre = meta.get("preBalances") or []
    post = meta.get("postBalances") or []
    try:
        payer_sol_pre = int(pre[0]) if pre else None
    except (TypeError, ValueError, IndexError):
        payer_sol_pre = None
    try:
        payer_sol_post = int(post[0]) if post else None
    except (TypeError, ValueError, IndexError):
        payer_sol_post = None

    sig = sigs[0] if isinstance(sigs, list) and sigs else None

    return {
        "line_number": line_number,
        "tx_hash": sig,
        "tx_index": tx_index_i,
        "cu": cu_i,
        "fee_lamports": fee_i,
        "has_err": 1 if meta.get("err") else 0,
        "n_accounts": len(keys) if isinstance(keys, list) else 0,
        "n_signers": n_signers,
        "n_ix": len(ixs) if isinstance(ixs, list) else 0,
        "n_inner_ix": n_inner,
        "n_lookups": len(lookups) if isinstance(lookups, list) else 0,
        "n_post_tb": len(meta.get("postTokenBalances") or []),
        "n_logs": len(logs) if isinstance(logs, list) else 0,
        "has_pump_program": has_pump,
        "has_compute_budget": int(COMPUTE_BUDGET in programs),
        "has_token_program": int(TOKEN_PROGRAM in programs),
        "has_ata": int(ASSOC_TOKEN in programs),
        "n_programs": len(set(programs)),
        "cu_limit": cu_limit,
        "cu_price_micro": cu_price,
        "n_pump_ix": n_pump_ix,
        "n_sol_transfers": n_sol_xfer,
        "max_xfer_lamports": max_xfer,
        "tip_lamports": tip_lamports,
        "has_jito_tip": has_jito_tip,
        "has_service_tip": has_service_tip,
        "has_any_tip": int(has_jito_tip or has_service_tip),
        "has_buy_same_tx": has_buy_same_tx,
        "has_create_v2": has_create_v2,
        "has_token_2022": has_token_2022,
        "payer_sol_pre": payer_sol_pre,
        "payer_sol_post": payer_sol_post,
    }
