# Informe corto — El sniper en una sentada

La versión **completa** (paso a paso, con definiciones de cada término, hipótesis y ablaciones) está en:

**`data/metadata/INFORME_COMPLETO.md`**

Léela si no quieres dar nada por sabido. Esto de aquí es el resumen.

---

## En una frase

El sniper no adivina el precio. **Sigue a creadores que ya le funcionaron** y, si es un extraño, **evita a quien dispara tokens cada dos minutos** y mira si esta vez mete ~1 SOL al nacer. Compra en el **mismo bloque**. Históricamente: **+8 894 SOL** netos (hit 56% tras fees).

---

## Números del examen (junio, no se usó para entrenar)

| Qué | Valor |
|-----|--------|
| ROC-AUC | 0.95 |
| PR-AUC | 0.70 (muestra; en el universo completo un competidor saca 0.22) |
| Precisión / recall / F1 (umbral 0.23, elegido en valid) | 0.56 / 0.81 / 0.66 |
| De los 100 más seguros | 93 eran snipes |
| Coincidencia con el bot | 1 989 de 2 445 (81%) |
| P&L del bot en test / capturado | +1 192 / +970 SOL |

Ablación: solo “¿ya lo conocía?” ROC 0.77; + forma de la tx 0.92; + anti-fábrica y todo **0.94**. Las reglas humanas explican; el modelo afina.

Drawdown máximo del bot ~−47 SOL. Gana también en desconocidos (+2 938 SOL).

Muestra para mirar: `data/processed/test_holdout_muestra.csv`.

Cierre Kaggle: **14 ago 2026, 21:00 UTC**. Falta publicar notebook + repo.
