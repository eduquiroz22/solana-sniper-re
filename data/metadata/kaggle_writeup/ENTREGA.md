# Cómo entregar (para maximizar nota)

Tres piezas. **No** un álbum de PNGs.

```
Ficha Kaggle (2 min)  →  Notebook (10 min)  →  GitHub (código)
     title + texto           historia+figuras        scripts
```

El juez lee la ficha, abre el notebook, y si duda mira el repo. Cada sitio tiene un trabajo.

---

## 1. Formulario del writeup (la ficha)

**Sí llena**

| Campo | Qué poner |
|-------|-----------|
| Title | `Same-block sniper, reverse-engineered` |
| Subtitle | `Same-block buys. A trust list. A factory filter.` |
| Cover 560×280 | `cover_560x280.png` — recorte cuadrado = **mitad izquierda** |
| Project Description | pega `WRITEUP_BODY.md` (es el texto, no las fotos) |
| Project links | 1) este notebook en Kaggle  2) el repo de GitHub |

**No llena / no subas**

- Media gallery (cero fotos extra)
- Files del writeup (no parquet, no modelo, no 16 PNG)
- DOI (da igual)

El cover **sí vale la pena**: es la tarjetita del listado. Sin foto pareces incompleto. No es “la presentación”; es el gancho. La explicación va en Description + notebook.

---

## 2. Notebook (esto es el paper)

Archivo: `notebooks/solana-sniper-reverse-engineering.ipynb`  
Figuras: `notebooks/assets/` (~700 KB, viajan con el repo)

Cómo publicarlo:

1. Sube el repo a GitHub (abajo).
2. En Kaggle → **New Notebook** → File → **Import from GitHub** (o sube el `.ipynb` + carpeta `assets`).
3. **Run All**. Debe pintar las 10 figuras con pie de foto.
4. Ponlo **Public**. Settings → Internet off está bien (no necesita red).
5. Copia la URL y pégala en Project links del writeup.

No hace falta Dataset de 30 GB. El notebook **narra** resultados ya calculados. El código pesado está en GitHub.

---

## 3. GitHub (código, no datos)

Crea un repo público, por ejemplo `solana-sniper-re`.

**Sube**

- `src/`, `scripts/`, `notebooks/` (con `assets/`)
- `data/metadata/*.json`, `data/metadata/kaggle_writeup/`, `data/metadata/*.md`
- `requirements.txt`, `config.yaml`, `README.md`

**No subas** (ya están en `.gitignore`)

- `data/raw/` (wallet, jsonl, TAR)
- `data/processed/`, `data/samples/`
- `.joblib` / parquet / el TAR de 39 GB

El “modelo” es un `HistGradientBoosting` de sklearn: se **reentrena en 20 s** con `scripts/19_train_backtest_kaggle.py`. No aporta nota subirlo como archivo de 2 MB. El juez quiere el **pipeline**, no un binario opaco.

---

## 4. Qué no hacer

- No adjuntar 16 imágenes al writeup sin texto.
- No subir el dataset entero a Kaggle (no cabe, no suma).
- No dejar el notebook privado.
- No poner el PR 0.70 como si fuera el universo de 5 M (el notebook ya lo dice: un público saca 0.22 en 852 k).

---

## Orden de clic (esta noche)

1. GitHub: repo público con lo de arriba.  
2. Kaggle Notebook: importar, Run All, Public, copiar URL.  
3. Writeup: title + subtitle + cover + pegar Description + dos links.  
4. Submit.
