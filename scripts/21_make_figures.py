#!/usr/bin/env python3
"""Figuras para el writeup: español, una idea por gráfico."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.common import ensure_dirs, load_config, write_json  # noqa: E402

OUT = ROOT / "data" / "metadata" / "figures"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print(f"  {p.name}", flush=True)
    return p


def fig_latency(meta: dict) -> None:
    lat = meta["behavior"]["latency"]
    same = lat["frac_same_slot"] * 100
    # next slot from cold_hypotheses ~18.7%
    rest_same_sec = max(lat["frac_same_second"] * 100 - same, 0)
    later = 100 - same - 18.7
    # use known fractions
    same_s = 79.6
    next_s = 18.7
    later_s = 100 - same_s - next_s
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["Mismo bloque\n(~0,4 s)", "Bloque siguiente", "Más tarde"]
    vals = [same_s, next_s, later_s]
    colors = ["#2f6f4e", "#c4a35a", "#8a8a8a"]
    ax.bar(labels, vals, color=colors, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=12)
    ax.set_ylabel("% de compras del sniper")
    ax.set_ylim(0, 100)
    ax.set_title("El 80% de las compras caen en el mismo bloque que el nacimiento")
    ax.set_xlabel("Cuándo llega la compra respecto al deploy")
    save(fig, "01_latencia_mismo_bloque.png")


def fig_hot_cold(fa: dict) -> None:
    t = fa["rule_and_cold_start"]["splits"]["test"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["Ya le había\ncomprado (hot)", "Primera vez\n(cold)"]
    vals = [t["hot_pos_rate"] * 100, t["cold_pos_rate"] * 100]
    ax.bar(labels, vals, color=["#2f6f4e", "#6b7c93"], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.8, f"{v:.1f}%", ha="center", fontsize=12)
    ax.set_ylabel("% de deploys que el sniper compró (junio, muestra)")
    ax.set_ylim(0, 50)
    ax.set_title("Si ya lo conoce, dispara ~8 veces más que si es extraño")
    save(fig, "02_hot_vs_cold.png")


def fig_factory() -> None:
    # test cold from informe / cold_hypotheses
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["Fábrica:\n3+ tokens en 1 h", "Sin ráfaga"]
    vals = [0.75, 7.48]
    ax.bar(labels, vals, color=["#a33b3b", "#2f6f4e"], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.15, f"{v:.2f}%", ha="center", fontsize=12)
    ax.set_ylabel("% que el sniper compra (solo desconocidos, junio)")
    ax.set_ylim(0, 10)
    ax.set_title("En extraños, evita a quien está lanzando tokens en ráfaga")
    save(fig, "03_anti_fabrica.png")


def fig_ablation() -> None:
    names = [
        "Solo «¿ya lo conocía?»",
        "Solo anti-fábrica",
        "Solo forma de la tx",
        "Hot + forma de la tx",
        "Todo junto",
    ]
    roc = [0.772, 0.872, 0.878, 0.917, 0.942]
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    y = np.arange(len(names))
    ax.barh(y, roc, color=["#8a8a8a", "#6b7c93", "#6b7c93", "#3d6b8a", "#2f6f4e"], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("ROC-AUC en test (junio)")
    ax.set_xlim(0.5, 1.0)
    ax.axvline(0.5, color="#999999", ls="--", lw=1, label="Moneda al aire (0,50)")
    for i, v in enumerate(roc):
        ax.text(v + 0.008, i, f"{v:.2f}", va="center")
    ax.set_title("Cada pieza suma; las tres juntas ordenan mejor")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    save(fig, "04_ablacion_roc.png")


def fig_equity(paths) -> dict:
    pnl = pl.read_parquet(paths["processed"] / "bot_token_pnl_net.parquet")
    lab = pl.read_parquet(paths["processed"] / "labeled_features.parquet").select(
        ["token_address", "split", "label"]
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
    days = daily["day"].to_list()
    eq = daily["equity"].to_list()
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(range(len(eq)), eq, color="#2f6f4e", lw=1.8)
    ax.fill_between(range(len(eq)), eq, color="#2f6f4e", alpha=0.12)
    ax.set_ylabel("SOL netos acumulados (tras fees)")
    ax.set_title("Curva de beneficios del sniper (mar–jun 2026)")
    step = max(len(days) // 6, 1)
    ticks = list(range(0, len(days), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([days[i] for i in ticks], rotation=20, ha="right")
    ax.set_xlabel("Fecha de la primera venta de cada posición")
    save(fig, "05_equity_bot.png")

    peak = np.maximum.accumulate(np.array(eq, dtype=float))
    dd = np.array(eq) - peak
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.fill_between(range(len(dd)), dd, 0, color="#a33b3b", alpha=0.35)
    ax.plot(range(len(dd)), dd, color="#a33b3b", lw=1.2)
    ax.set_ylabel("Caída desde el máximo (SOL)")
    ax.set_title("Drawdown: la peor bajada fue ~−47 SOL")
    ax.set_xticks(ticks)
    ax.set_xticklabels([days[i] for i in ticks], rotation=20, ha="right")
    ax.set_xlabel("Fecha")
    save(fig, "06_drawdown.png")
    # downsample for canvas (~40 pts)
    idx = np.linspace(0, len(eq) - 1, min(40, len(eq))).astype(int)
    return {
        "days": [days[i] for i in idx],
        "equity": [float(eq[i]) for i in idx],
        "drawdown": [float(dd[i]) for i in idx],
    }


def fig_fees() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["Bruto\n(compras−ventas)", "Fees\n(gas+tip+DEX)", "Neto"]
    vals = [17629, 8735, 8894]
    colors = ["#6b7c93", "#a33b3b", "#2f6f4e"]
    ax.bar(labels, vals, color=colors, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 250, f"{v:,.0f} SOL", ha="center", fontsize=11)
    ax.set_ylabel("SOL")
    ax.set_title("Las comisiones se comen la mitad del bruto")
    save(fig, "07_bruto_fees_neto.png")


def fig_hot_cold_pnl() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(2)
    w = 0.35
    net = [5794, 2938]
    hit = [56.6, 52.8]
    b1 = ax.bar(x - w / 2, net, w, color="#2f6f4e", label="P&L neto (SOL)")
    ax2 = ax.twinx()
    ax2.grid(False)
    b2 = ax2.bar(x + w / 2, hit, w, color="#c4a35a", label="Hit rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Conocidos (hot)", "Extraños (cold)"])
    ax.set_ylabel("SOL netos")
    ax2.set_ylabel("Hit rate (%)")
    ax2.set_ylim(0, 100)
    ax.set_title("También gana con desconocidos (media por trade incluso mayor)")
    ax.legend(handles=[b1, b2], loc="upper right")
    save(fig, "08_pnl_hot_cold.png")


def fig_replica() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    labels = ["Compras\ndel bot", "Compras\nde la réplica", "Coincidencia", "Se nos\nescaparon", "De más\n(sin precio)"]
    vals = [2445, 3564, 1989, 456, 1575]
    colors = ["#2f6f4e", "#3d6b8a", "#2f6f4e", "#c4a35a", "#8a8a8a"]
    ax.bar(labels, vals, color=colors, width=0.65)
    for i, v in enumerate(vals):
        ax.text(i, v + 40, f"{v:,}", ha="center", fontsize=10)
    ax.set_ylabel("Número de deploys (test junio)")
    ax.set_title("Réplica vs bot: pillamos 1 989 de 2 445 (81%)")
    save(fig, "09_replica_vs_bot.png")


def fig_captured() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    thr = ["0.50", "0.75\n(elegido)", "0.90", "0.95"]
    sol = [1121, 927, 776, 627]
    prec = [44.0, 54.2, 64.2, 73.0]
    ax.bar(thr, sol, color="#2f6f4e", width=0.55, label="SOL netos pillados (solo coincidencias)")
    ax.set_ylabel("SOL netos capturados")
    ax2 = ax.twinx()
    ax2.grid(False)
    ax2.plot(thr, prec, color="#a33b3b", marker="o", lw=2, label="Precisión (%)")
    ax2.set_ylabel("Precisión (%)")
    ax2.set_ylim(0, 100)
    ax.set_title("Más estricto: más limpio, menos dinero de su curva")
    ax.set_xlabel("Umbral del score")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    save(fig, "10_umbral_sol_vs_precision.png")


def fig_calibration() -> None:
    mean_s = [0.020, 0.166, 0.361, 0.620, 0.834, 0.958]
    real = [0.003, 0.029, 0.070, 0.179, 0.337, 0.642]
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", label="Si el score fuera una probabilidad real")
    ax.plot(mean_s, real, marker="o", color="#3d6b8a", lw=2, label="Lo que pasó en junio")
    ax.set_xlabel("Score medio del modelo")
    ax.set_ylabel("Fracción que de verdad eran snipes")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibración en test (modelo sin peso extra)")
    ax.legend(loc="upper left")
    save(fig, "11_calibracion.png")


def fig_roc_pr(paths) -> None:
    sc = pl.read_parquet(paths["processed"] / "scored_deploys.parquet")
    te = sc.filter(pl.col("split") == "test")
    y = te["label"].to_numpy()
    s = te["score"].to_numpy()
    from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score

    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.plot(fpr, tpr, color="#2f6f4e", lw=2, label=f"Modelo (AUC {auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", label="Moneda al aire")
    ax.set_xlabel("Falsos positivos (de los que NO eran snipe)")
    ax.set_ylabel("Verdaderos positivos (de los que SÍ eran snipe)")
    ax.set_title("Curva ROC en el examen de junio")
    ax.legend(loc="lower right")
    save(fig, "12_roc_test.png")

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    base = float(y.mean())
    ax.plot(rec, prec, color="#3d6b8a", lw=2, label=f"Modelo (AP {average_precision_score(y, s):.2f})")
    ax.axhline(base, color="#999999", ls="--", label=f"Azar en esta muestra ({base:.2f})")
    ax.set_xlabel("Recall (fracción de snipes reales pillados)")
    ax.set_ylabel("Precisión (de las que compramos, cuántas eran suyas)")
    ax.set_ylim(0, 1)
    ax.set_title("Curva precisión–recall en junio (muestra, no el universo)")
    ax.legend(loc="upper right")
    save(fig, "13_pr_test.png")


def fig_entry_hold(paths) -> None:
    w = pl.read_parquet(paths["wallet"] / "5brv79e_activity.parquet").with_columns(
        pl.col("quote_amount").cast(pl.Float64, strict=False).alias("qty")
    )
    buys = w.filter(
        (pl.col("event_type") == "buy") & pl.col("quote_token_symbol").is_in(["SOL", "WSOL"])
    )
    qty = buys["qty"].drop_nulls().clip(0, 20).to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(qty, bins=40, color="#3d6b8a", edgecolor="white")
    ax.axvline(1.98, color="#a33b3b", ls="--", lw=1.6, label="Mediana 1,98 SOL")
    ax.set_xlabel("SOL puestos en la compra")
    ax.set_ylabel("Número de compras")
    ax.set_title("Tamaño de entrada: casi siempre ~2 SOL")
    ax.legend()
    save(fig, "14_tamano_entrada.png")

    first_buy = (
        w.filter(pl.col("event_type") == "buy")
        .sort("timestamp")
        .group_by("token_address")
        .agg(pl.col("timestamp").first().alias("t0"))
    )
    first_sell = (
        w.filter(pl.col("event_type") == "sell")
        .sort("timestamp")
        .group_by("token_address")
        .agg(pl.col("timestamp").first().alias("t1"))
    )
    hold = (
        first_buy.join(first_sell, on="token_address", how="inner")
        .with_columns((pl.col("t1") - pl.col("t0")).alias("h"))
        .filter((pl.col("h") >= 0) & (pl.col("h") <= 30))
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(hold["h"].to_numpy(), bins=31, color="#2f6f4e", edgecolor="white")
    ax.axvline(1.0, color="#a33b3b", ls="--", lw=1.6, label="Mediana 1 s")
    ax.set_xlabel("Segundos hasta la primera venta")
    ax.set_ylabel("Número de tokens")
    ax.set_title("No es un inversor: vende en ~1 segundo")
    ax.legend()
    save(fig, "15_hold_segundos.png")


def fig_quiet_sol() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    ax = axes[0]
    ax.bar(["Los que compra", "Los que no"], [26.2, 0.041], color=["#2f6f4e", "#a33b3b"], width=0.55)
    ax.set_ylabel("Horas desde el token anterior (mediana)")
    ax.set_title("Extraño callado (~26 h) vs ametralladora (~2 min)")
    ax.set_yscale("log")
    ax = axes[1]
    ax.bar(["Los que compra", "Los que no"], [1.03, 0.16], color=["#2f6f4e", "#a33b3b"], width=0.55)
    ax.set_ylabel("SOL gastados al crear (mediana)")
    ax.set_title("En el nacimiento mete ~1 SOL, no céntimos")
    fig.tight_layout()
    save(fig, "16_callado_y_sol.png")


def main() -> int:
    _style()
    cfg = load_config()
    paths = ensure_dirs(cfg)
    print("figures →", OUT, flush=True)
    meta = json.loads((paths["metadata"] / "kaggle_train_backtest.json").read_text())
    fa = json.loads((paths["metadata"] / "final_analysis.json").read_text())

    fig_latency(meta)
    fig_hot_cold(fa)
    fig_factory()
    fig_ablation()
    eq = fig_equity(paths)
    fig_fees()
    fig_hot_cold_pnl()
    fig_replica()
    fig_captured()
    fig_calibration()
    fig_roc_pr(paths)
    fig_entry_hold(paths)
    fig_quiet_sol()

    write_json(paths["metadata"] / "figures_index.json", {"dir": str(OUT), "equity_downsampled": eq})
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
