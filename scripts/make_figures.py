#!/usr/bin/env python3
"""English figures + 560x280 cover for the Kaggle writeup form."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import auc, average_precision_score, roc_curve, precision_recall_curve

from src.common import ensure_dirs, load_config  # noqa: E402

OUT = ROOT / "data" / "metadata" / "kaggle_writeup"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2a2a2a",
            "axes.labelcolor": "#1a1a1a",
            "text.color": "#1a1a1a",
            "xtick.color": "#2a2a2a",
            "ytick.color": "#2a2a2a",
            "axes.grid": True,
            "grid.color": "#e6e6e6",
            "grid.linewidth": 0.6,
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.labelsize": 11,
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
        }
    )


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print(" ", p.name, flush=True)


def _font(size: int, bold: bool = False):
    from matplotlib import font_manager
    from PIL import ImageFont

    name = "DejaVu Sans:style=Bold" if bold else "DejaVu Sans"
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def cover() -> None:
    """560x280 card. Left 280x280 is a self-contained square thumb."""
    from PIL import Image, ImageDraw

    OUT.mkdir(parents=True, exist_ok=True)
    bg, ink, mute, accent = "#f7f6f2", "#1a1a1a", "#5a5a5a", "#2f6f4e"
    im = Image.new("RGB", (560, 280), bg)
    d = ImageDraw.Draw(im)

    # left square 280x280 — this is the thumbnail crop
    d.rectangle((0, 0, 8, 280), fill=accent)
    d.text((28, 36), "SAME-BLOCK", font=_font(28, True), fill=ink)
    d.text((28, 72), "SNIPER", font=_font(28, True), fill=ink)
    d.text((28, 118), "Known list + anti-factory", font=_font(14), fill=mute)
    d.text((28, 168), "79.6%", font=_font(44, True), fill=accent)
    d.text((28, 222), "same-slot buys", font=_font(16), fill=mute)

    # right half — only visible on the wide card
    d.line((280, 28, 280, 252), fill="#ddd8ce", width=1)
    d.text((308, 48), "+8,894", font=_font(36, True), fill=accent)
    d.text((308, 96), "SOL net after fees", font=_font(15), fill=mute)
    d.text((308, 148), "81%", font=_font(36, True), fill=accent)
    d.text((308, 196), "replica recall  ·  June holdout", font=_font(15), fill=mute)
    d.text((308, 236), "No future prices in the decision", font=_font(13), fill="#7a7a7a")

    dest = OUT / "cover_560x280.png"
    im.save(dest, "PNG")
    im.crop((0, 0, 280, 280)).save(OUT / "thumb_280x280.png", "PNG")
    print("  cover_560x280.png + thumb_280x280.png", flush=True)


def main() -> int:
    style()
    cfg = load_config()
    paths = ensure_dirs(cfg)
    print("kaggle writeup figs →", OUT, flush=True)
    cover()

    # 1 latency
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    labels = ["Same block\n(~0.4 s)", "Next block", "Later"]
    vals = [79.6, 18.7, 1.7]
    ax.bar(labels, vals, color=["#2f6f4e", "#c4a35a", "#8a8a8a"], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.3, f"{v:.1f}%", ha="center", fontsize=12)
    ax.set_ylabel("% of sniper buys")
    ax.set_ylim(0, 100)
    ax.set_title("80% of buys land in the same block as token creation")
    ax.set_xlabel("Buy timing vs deploy")
    save(fig, "01_same_block.png")

    # 2 hot cold
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.bar(["Known deployer\n(hot)", "First time\n(cold)"], [38.8, 4.8], color=["#2f6f4e", "#6b7c93"], width=0.55)
    ax.text(0, 40.0, "38.8%", ha="center")
    ax.text(1, 6.2, "4.8%", ha="center")
    ax.set_ylabel("% of deploys the sniper bought (June sample)")
    ax.set_ylim(0, 50)
    ax.set_title("Repeat deployers are bought ~8× more often")
    save(fig, "02_hot_vs_cold.png")

    # 3 factory
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.bar(["Burst: 3+ tokens / hour", "No burst"], [0.75, 7.48], color=["#a33b3b", "#2f6f4e"], width=0.55)
    ax.text(0, 0.95, "0.75%", ha="center")
    ax.text(1, 7.7, "7.5%", ha="center")
    ax.set_ylabel("% bought (cold deployers only, June)")
    ax.set_ylim(0, 10)
    ax.set_title("On strangers, the sniper skips token factories")
    save(fig, "03_anti_factory.png")

    # 4 ablation
    names = ["Known-deployer\nflag only", "Anti-factory\nonly", "Deploy-tx\nshape only", "Known +\ntx shape", "All features"]
    roc = [0.77, 0.87, 0.88, 0.92, 0.95]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    y = np.arange(len(names))
    ax.barh(y, roc, color=["#8a8a8a", "#6b7c93", "#6b7c93", "#3d6b8a", "#2f6f4e"], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("ROC-AUC on June holdout")
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color="#999999", ls="--", lw=1, label="Chance (0.50)")
    for i, v in enumerate(roc):
        ax.text(v + 0.008, i, f"{v:.2f}", va="center")
    ax.set_title("Each signal adds; together they rank best")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    save(fig, "04_ablation.png")

    # 5 equity
    pnl = pl.read_parquet(paths["processed"] / "bot_token_pnl_net.parquet")
    lab = pl.read_parquet(paths["processed"] / "labeled_features.parquet").select(
        ["token_address", "label"]
    )
    bot = (
        lab.filter(pl.col("label") == 1)
        .join(pnl, on="token_address", how="inner")
        .filter(pl.col("n_sells") > 0)
        .sort("first_ts")
        .with_columns(
            [
                pl.col("net_sol").cum_sum().alias("equity"),
                pl.from_epoch(pl.col("first_ts"), time_unit="s").dt.strftime("%Y-%m-%d").alias("day"),
            ]
        )
    )
    daily = bot.group_by("day").agg(pl.col("equity").last()).sort("day")
    days, eq = daily["day"].to_list(), daily["equity"].to_list()
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(range(len(eq)), eq, color="#2f6f4e", lw=1.8)
    ax.fill_between(range(len(eq)), eq, color="#2f6f4e", alpha=0.12)
    ax.set_ylabel("Cumulative net SOL (after gas + tip + DEX)")
    ax.set_title("Sniper P&L, Mar–Jun 2026  ·  +8,894 SOL net")
    step = max(len(days) // 6, 1)
    ticks = list(range(0, len(days), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([days[i] for i in ticks], rotation=18, ha="right")
    ax.set_xlabel("Date of first sell")
    save(fig, "05_equity.png")

    # 6 fees
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.bar(["Gross", "Fees\n(gas+tip+DEX)", "Net"], [17629, 8735, 8894], color=["#6b7c93", "#a33b3b", "#2f6f4e"], width=0.55)
    for i, v in enumerate([17629, 8735, 8894]):
        ax.text(i, v + 280, f"{v:,.0f}", ha="center")
    ax.set_ylabel("SOL")
    ax.set_title("Fees consume about half of gross P&L")
    save(fig, "06_fees.png")

    # 7 replica — live from scored
    sc = pl.read_parquet(paths["processed"] / "scored_deploys.parquet").filter(pl.col("split") == "test")
    thr = 0.23
    n_bot = int(sc["label"].sum())
    n_rep = int((sc["score"] >= thr).sum())
    n_tp = int(((sc["score"] >= thr) & (sc["label"] == 1)).sum())
    n_fn = n_bot - n_tp
    n_fp = n_rep - n_tp
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    labs = ["Bot buys", "Replica buys", "Overlap", "Missed", "Extra\n(no local price)"]
    vals = [n_bot, n_rep, n_tp, n_fn, n_fp]
    ax.bar(labs, vals, color=["#2f6f4e", "#3d6b8a", "#2f6f4e", "#c4a35a", "#8a8a8a"], width=0.65)
    for i, v in enumerate(vals):
        ax.text(i, v + 40, f"{v:,}", ha="center", fontsize=10)
    ax.set_ylabel("Deploys (June holdout sample)")
    ax.set_title(f"Replica vs bot: {n_tp:,} of {n_bot:,} ({100*n_tp/n_bot:.0f}% recall)")
    save(fig, "07_replica.png")

    # 8 ROC + calib from live scores
    y = sc["label"].to_numpy()
    s = sc["score"].to_numpy()
    fpr, tpr, _ = roc_curve(y, s)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(fpr, tpr, color="#2f6f4e", lw=2, label=f"Model (AUC {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC on June holdout (sampled negatives)")
    ax.legend(loc="lower right")
    save(fig, "08_roc.png")

    prec, rec, _ = precision_recall_curve(y, s)
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(rec, prec, color="#3d6b8a", lw=2, label=f"Model (AP {average_precision_score(y, s):.2f})")
    ax.axhline(float(y.mean()), color="#999999", ls="--", label=f"Base rate in sample ({y.mean():.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1)
    ax.set_title("PR curve — sample, not the full 5M universe")
    ax.legend(loc="upper right")
    save(fig, "09_pr.png")

    # calibration from live
    bins = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.01]
    ms, fr = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (s >= a) & (s < b)
        if not m.any():
            continue
        ms.append(float(s[m].mean()))
        fr.append(float(y[m].mean()))
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", label="Perfect calibration")
    ax.plot(ms, fr, marker="o", color="#3d6b8a", lw=2, label="June holdout")
    ax.set_xlabel("Mean model score")
    ax.set_ylabel("Actual sniper-buy rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration after dropping 15× class weight")
    ax.legend(loc="upper left")
    save(fig, "10_calibration.png")

    # quiet + sol
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1))
    axes[0].bar(["Bought", "Skipped"], [26.2, 0.041], color=["#2f6f4e", "#a33b3b"], width=0.55)
    axes[0].set_ylabel("Hours since last launch (median)")
    axes[0].set_title("Quiet stranger (~26 h) vs spray (~2 min)")
    axes[0].set_yscale("log")
    axes[1].bar(["Bought", "Skipped"], [1.03, 0.16], color=["#2f6f4e", "#a33b3b"], width=0.55)
    axes[1].set_ylabel("SOL spent in the create tx (median)")
    axes[1].set_title("Creates with ~1 SOL, not cents")
    fig.tight_layout()
    save(fig, "11_quiet_and_sol.png")

    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
