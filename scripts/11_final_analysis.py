#!/usr/bin/env python3
"""Rule baseline, cold-start split, optional tx-feature retrain, final Spanish report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common import ensure_dirs, load_config, write_json  # noqa: E402

TX_FEATS = [
    "tx_index",
    "cu",
    "fee_lamports",
    "has_err",
    "n_accounts",
    "n_signers",
    "n_ix",
    "n_inner_ix",
    "n_lookups",
    "n_post_tb",
    "n_logs",
    "has_pump_program",
    "has_compute_budget",
    "has_token_program",
    "has_ata",
    "n_programs",
]


def _auc(y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score

    if min(y) == max(y):
        return None, None
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def _prec_at_k(y, p, k=100):
    import numpy as np

    y = np.asarray(y)
    p = np.asarray(p)
    k = min(k, len(p))
    return float(y[np.argsort(-p)[:k]].mean())


def main() -> int:
    import numpy as np
    import polars as pl
    from sklearn.ensemble import HistGradientBoostingClassifier

    cfg = load_config()
    paths = ensure_dirs(cfg)
    labeled = pl.read_parquet(paths["processed"] / "labeled_features.parquet")
    pos_tx_p = paths["processed"] / "pos_tx_features.parquet"
    neg_tx_p = paths["processed"] / "neg_tx_features.parquet"

    notes = []
    if pos_tx_p.is_file():
        pos_tx = pl.read_parquet(pos_tx_p)
        labeled = labeled.join(pos_tx, on="tx_hash", how="left", suffix="_dup")
        notes.append(f"joined pos tx features rows={pos_tx.height}")
    if neg_tx_p.is_file():
        sample = pl.read_parquet(paths["samples"] / "negative_200k.parquet").select(
            ["token_address", "line_number"]
        )
        neg_tx = pl.read_parquet(neg_tx_p).join(sample, on="line_number", how="left")
        labeled = labeled.join(neg_tx, on="token_address", how="left", suffix="_neg")
        notes.append(f"joined neg tx features rows={neg_tx.height}")

    # Coalesce tx feature columns if duplicated
    for c in TX_FEATS:
        cands = [c, f"{c}_neg", f"{c}_posjoin"]
        have = [x for x in cands if x in labeled.columns]
        if not have:
            continue
        expr = pl.col(have[0])
        for extra in have[1:]:
            expr = expr.fill_null(pl.col(extra))
        labeled = labeled.with_columns(expr.alias(c))

    # Only train on tx features if BOTH classes have them (else "has tx_index" == label)
    n_pos_tx = 0
    n_neg_tx = 0
    if "tx_index" in labeled.columns:
        n_pos_tx = labeled.filter((pl.col("label") == 1) & pl.col("tx_index").is_not_null()).height
        n_neg_tx = labeled.filter((pl.col("label") == 0) & pl.col("tx_index").is_not_null()).height
    has_tx = n_pos_tx > 1000 and n_neg_tx > 1000
    notes.append(f"tx_index coverage pos={n_pos_tx} neg={n_neg_tx} usable={has_tx}")

    # --- rule baseline: prior_bought_same_signer > 0 ---
    results: dict = {"notes": notes, "splits": {}}
    for sp in ("train", "valid", "test"):
        sub = labeled.filter(pl.col("split") == sp)
        y = sub["label"].to_numpy()
        rule = (sub["prior_bought_same_signer"] > 0).cast(pl.Int8).to_numpy()
        prior = sub["prior_bought_same_signer"].to_numpy()
        roc, pr = _auc(y, prior)
        roc_r, pr_r = _auc(y, rule)
        cold = sub.filter(pl.col("prior_bought_same_signer") == 0)
        hot = sub.filter(pl.col("prior_bought_same_signer") > 0)
        results["splits"][sp] = {
            "n": sub.height,
            "n_pos": int(sub["label"].sum()),
            "rule_precision": float(y[rule == 1].mean()) if rule.any() else None,
            "rule_recall": float(rule[y == 1].mean()) if y.sum() else None,
            "prior_as_score_roc": roc,
            "prior_as_score_pr": pr,
            "rule_roc": roc_r,
            "cold_n": cold.height,
            "cold_n_pos": int(cold["label"].sum()),
            "hot_n": hot.height,
            "hot_n_pos": int(hot["label"].sum()),
            "hot_pos_rate": float(hot["label"].mean()) if hot.height else None,
            "cold_pos_rate": float(cold["label"].mean()) if cold.height else None,
        }

    # --- retrain with tx features if mostly present ---
    base_cols = [
        "hour_utc",
        "dow",
        "month",
        "days_since_bot_start",
        "token_is_pump",
        "token_len",
        "has_signer",
        "creator_missing",
        "signer_eq_creator",
        "prior_bought_same_signer",
        "wallet_events_before",
        "wallet_hits_signer_before",
    ]
    extra = [c for c in TX_FEATS if c in labeled.columns]
    use = [c for c in base_cols + extra if c in labeled.columns]
    model_metrics = {}
    if extra and has_tx:
        print(f"Retraining with extra tx features: {extra}")
        tr = labeled.filter(pl.col("split") == "train")
        x_tr = tr.select(use).to_pandas()
        y_tr = tr["label"].to_numpy()
        n_pos = max(int(y_tr.sum()), 1)
        n_neg = max(len(y_tr) - n_pos, 1)
        w = np.where(y_tr == 1, n_neg / n_pos, 1.0)
        clf = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=250,
            l2_regularization=0.1,
            min_samples_leaf=40,
            random_state=42,
        )
        clf.fit(x_tr, y_tr, sample_weight=w)
        for sp in ("valid", "test"):
            sub = labeled.filter(pl.col("split") == sp)
            x = sub.select(use).to_pandas()
            y = sub["label"].to_numpy()
            p = clf.predict_proba(x)[:, 1]
            roc, pr = _auc(y, p)
            cold = sub["prior_bought_same_signer"].to_numpy() == 0
            hot = ~cold
            roc_c, pr_c = _auc(y[cold], p[cold]) if cold.any() else (None, None)
            roc_h, pr_h = _auc(y[hot], p[hot]) if hot.any() else (None, None)
            model_metrics[sp] = {
                "roc_auc": roc,
                "pr_auc": pr,
                "precision_at_100": _prec_at_k(y, p, 100),
                "cold_roc": roc_c,
                "hot_roc": roc_h,
                "n_features": len(use),
            }
            print(f"  {sp} ROC={roc} PR={pr} P@100={model_metrics[sp]['precision_at_100']} cold_roc={roc_c} hot_roc={roc_h}")
        import joblib

        joblib.dump({"model": clf, "features": use}, ROOT / "models" / "baseline_hgb_tx.joblib")
    else:
        notes.append("tx features not complete yet — report uses Phase-2 baseline + rule analysis only")

    # Load original baseline metrics if present
    base_json = paths["metadata"] / "baseline_metrics.json"
    baseline = json.loads(base_json.read_text()) if base_json.is_file() else {}

    write_json(
        paths["metadata"] / "final_analysis.json",
        {
            "rule_and_cold_start": results,
            "tx_model": model_metrics,
            "features_used": use,
            "notes": notes,
            "baseline_test_roc": (baseline.get("metrics") or {}).get("test", {}).get("roc_auc"),
        },
    )

    te = results["splits"]["test"]
    va = results["splits"]["valid"]
    bte = (baseline.get("metrics") or {}).get("test") or {}
    tx_te = model_metrics.get("test") or {}

    report = f"""# Informe final — Solana Sniper (Phase 1 + 2)

Este documento explica **qué se hizo**, **qué se puede afirmar** y **qué no**.
Está escrito para que se entienda sin ser experto, pero sin suavizar los números.

---

## 1. El problema, en una frase

Hay un bot (wallet `5brv79e…`) que compra algunos tokens **en el momento en que se crean**.
El challenge pide: *con la información que existía en ese instante*, ¿se puede saber cuáles iba a comprar?*

Ese instante se llama **`t_decision`**: el `blockTime` del deploy (creación del token).
Todo lo que pasa *después* (precio, trades, si “moonó”) **no se puede usar** para decidir.
Eso se llama **look-ahead / leakage**: sería hacer trampa, porque el bot no veía el futuro.

---

## 2. Qué datos usamos (y cuáles no)

**Sí (ya en disco, laptop):**
- Positivos: 15 927 tokens que el bot **sí** compró + su tx de deploy.
- Negativos: muestra de **196 878** tokens que **no** compró (de ~5.06 millones).
- Actividad del propio bot (~87 k eventos).

**No (a propósito):**
- Activity de deployers *no comprados* (~23 GiB): no cabe bien y, además, la activity de *comprados* está **definida por el label** — usarla como feature sería circular.
- TAR completo (~39 GiB), bloques raw de junio, etc.

Los negativos son una **muestra**. Contra los 5 millones reales, acertar se vuelve más difícil.
Los números de precisión de abajo son **optimistas** respecto al universo completo.

---

## 3. Cómo se separó el tiempo (no es un random split)

El bot opera ~12 mar 2026 → 30 jun 2026. Un split aleatorio mezclaría pasado y futuro.

| Split | Fechas | Filas | Positivos | % positivos |
|-------|--------|-------|-----------|-------------|
| train | 12 mar → 29 may | {results['splits']['train']['n']:,} | {results['splits']['train']['n_pos']:,} | {100*results['splits']['train']['n_pos']/max(results['splits']['train']['n'],1):.2f}% |
| valid | 29 may → 12 jun | {va['n']:,} | {va['n_pos']:,} | {100*va['n_pos']/max(va['n'],1):.2f}% |
| **test** | **12 jun → 30 jun** | **{te['n']:,}** | **{te['n_pos']:,}** | **{100*te['n_pos']/max(te['n'],1):.2f}%** |

El **test no se usó para entrenar**. Es el examen.

---

## 4. Qué significa cada métrica (para no sobreinterpretar)

- **ROC-AUC**: si tomas un positivo al azar y un negativo al azar, ¿con qué probabilidad el modelo le da *más score* al positivo? 0.5 = moneda; 1.0 = perfecto. **0.82 es bueno** para ranking.
- **PR-AUC**: qué tan bien rankinguea cuando hay pocos positivos. Hay que compararla con la tasa base (~11% en test). **0.49 vs 0.11** es claramente mejor que azar.
- **Precision@100**: de los 100 tokens con score más alto en test, ¿cuántos compró el bot de verdad? **86/100**.
- **Precision / Recall a umbral 0.5**: si dices “snipe” a todo lo que el modelo puntúa ≥ 0.5, pillas muchos (recall alto) pero también muchos falsos (precision baja). Eso **no** es el punto de operación útil.
- Una regla útil: subir el umbral hasta pillar ~20% de los snipes → en test la precisión sube a ~**0.67**.

---

## 5. Resultado del modelo baseline (features simples)

En test (junio, no visto):

| Métrica | Valor |
|---------|-------|
| ROC-AUC | **{bte.get('roc_auc') and round(bte['roc_auc'], 3)}** |
| PR-AUC | **{bte.get('pr_auc') and round(bte['pr_auc'], 3)}** (base {bte.get('pos_rate') and round(bte['pos_rate'], 3)}) |
| Precision@100 | **{bte.get('precision_at_100')}** |
| P / R @ 0.5 | {bte.get('precision') and round(bte['precision'], 3)} / {bte.get('recall') and round(bte['recall'], 3)} |

---

## 6. El hallazgo de verdad (no es magia de ML)

Casi todo el poder viene de **una** idea:

> El bot **vuelve a comprar** tokens de wallets (`tx_signer`) a las que **ya les había comprado antes**.

Eso se llama `prior_bought_same_signer`: cuántos tokens comprados previos tiene ese deployer **antes** de este deploy (solo pasado → no hay leakage).

### Regla tonta: “si ya le compré, sniper”

En **test**:
- Precision de la regla: **{te.get('rule_precision') and round(te['rule_precision'], 3)}**
- Recall de la regla: **{te.get('rule_recall') and round(te['rule_recall'], 3)}**
- ROC si usas el *conteo* (0,1,2,…) como score: **{te.get('prior_as_score_roc') and round(te['prior_as_score_roc'], 3)}**

### Dos mundos distintos

| Grupo en test | Qué es | Pos rate |
|---------------|--------|----------|
| **Hot** (ya le había comprado) | `{te['hot_n']:,}` deploys | **{te.get('hot_pos_rate') and round(100*te['hot_pos_rate'], 1)}%** |
| **Cold** (primera vez que vemos al deployer) | `{te['cold_n']:,}` deploys | **{te.get('cold_pos_rate') and round(100*te['cold_pos_rate'], 1)}%** |

Interpretación: si el deployer es “conocido”, el bot es **mucho** más probable que compre.
Si es la **primera vez**, el problema es otro — y el modelo actual ayuda poco ahí.
Eso no es un fallo escondido: **es el comportamiento del sniper**.

El ML (hora, pump.fun, etc.) solo **pulió** un poco esa regla. No descubrió un patrón secreto enorme.

---

## 7. Features extra de la transacción de deploy

La tx de creación **sí** es información de `t_decision`: fee, compute units, nº de instrucciones, `transactionIndex` (posición **dentro del bloque** — típico de snipers), programas (pump.fun, compute budget…).

- Positivos: parseados del JSONL local (~48 MiB).
- Negativos: hay que **volver a streamear** el TAR (~14.5 GiB) para leer solo las ~197k líneas muestreadas.

{"Modelo con tx features en test: ROC=" + str(tx_te.get("roc_auc")) + "  PR=" + str(tx_te.get("pr_auc")) + "  P@100=" + str(tx_te.get("precision_at_100")) + "  (cold ROC=" + str(tx_te.get("cold_roc")) + ", hot ROC=" + str(tx_te.get("hot_roc")) + ")" if tx_te else "Esta noche, si el stream de negativos terminó, aquí aparece la comparación. Si no, el baseline de la sección 5 sigue siendo el resultado oficial."}

---

## 8. Qué **no** concluimos

- No tenemos un bot listo para producción.
- No ganamos el Kaggle solo con esto (hace falta cold-start: quién es el deployer *antes* de que el sniper le compre, sin usar la tabla de activity de positivos).
- Precision@100 = 0.86 **no** significa “86% de acierto en general”; es solo el top 100 del test muestreados.
- El servidor del instituto no aportó (firewall a los puertos del dataset). Todo esto es **laptop**.

---

## 9. Archivos

| Archivo | Qué es |
|---------|--------|
| `data/metadata/PHASE2_REPORT.md` | Informe técnico corto del baseline |
| `data/metadata/baseline_metrics.json` | Números del primer modelo |
| `data/metadata/final_analysis.json` | Regla, cold-start, modelo+tx |
| `data/processed/labeled_features.parquet` | Tabla usada para entrenar |
| `models/baseline_hgb.joblib` | Modelo simple |
| `models/baseline_hgb_tx.joblib` | Modelo + features de tx (si existió) |
| `data/samples/negative_200k.parquet` | Muestra de no-comprados |

---

## 10. Siguiente paso (si se quiere mejorar de verdad)

El agujero es **cold-start**: deployers nuevos. Para eso haría falta historial de *esos* wallets **antes** del deploy, sin construir la tabla a partir del label. Eso es el archivo grande `not_bought_deployers_activity` o un recorte por los signers del sample. Es otra noche de stream, no más árboles.

Hasta entonces, la afirmación honesta es:

> Sabemos bastante bien **cuándo el sniper insiste en un deployer conocido**.  
> Sabemos poco sobre **por qué elige a alguien la primera vez**.
"""
    dest = paths["metadata"] / "INFORME_FINAL.md"
    dest.write_text(report, encoding="utf-8")
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
