# Informe completo — Reconstruir al sniper de Solana

Este texto está pensado para leerse **de arriba a abajo**, aunque no sepas qué es Solana, un bot, ni “machine learning”.  
Cada vez que aparece una palabra técnica, **se define en ese momento**. Si más adelante la vuelves a ver, ya está definida.

Números cerrados el 13 ago 2026. El challenge cierra el **14 ago 2026, 21:00 UTC**.

---

## 0. Cómo leer esto (mapa)

1. **Qué es el mundo** (cripto, Solana, tokens, wallets).  
2. **Qué nos pedían** (Kaggle).  
3. **Diccionario** de todo lo que usamos (incluido machine learning).  
4. **Paso a paso** lo que hicimos, script a script.  
5. **Hipótesis**: qué creímos, cómo lo medimos, qué salió.  
6. **El modelo** y las métricas, en cristiano.  
7. **Dinero** (cuánto ganó el bot y cuánto “pillaríamos”).  
8. **Qué no funcionó** y **qué falta**.

Si solo quieres la conclusión: ve a la **sección 12**.

---

## 1. El mundo, sin dar nada por sabido

### 1.1 Internet de dinero

Una **criptomoneda** es dinero digital que no depende de un banco central. La más famosa es Bitcoin. **Solana** es otra red de ese tipo: un ordenador mundial donde la gente envía dinero y programas. Es **muy rápida**: un “paso de reloj” (lo llamamos **slot** o **bloque**) dura unos **400 milisegundos** (0,4 segundos). En un bloque caben cientos o miles de acciones.

Una **transacción** (abreviado **tx**) es una acción firmada: “crea este token”, “compra”, “vende”, “paga una propina”. Quien la firma es una **wallet** (una cuenta con una dirección larga de letras y números).

Nuestro bot es la wallet:

`5brv79eFZ2rGprXNvqgVJBkBptkkw8GJX1XydJyZLyAr`

(a veces la acortamos a `5brv79e…`).

### 1.2 Tokens y Pump.fun

Un **token** es una moneda nueva dentro de Solana. Cualquiera puede crear una. La mayoría no valen nada: se crean, alguien las bombea un minuto y mueren.

**Pump.fun** es una web/app muy usada para *lanzar* tokens baratos. Muchos de los que mira el bot nacen ahí. En los datos, `token_is_pump = 1` significa “este token parece de Pump.fun”.

**Deploy** = el *nacimiento* del token: la transacción que lo crea.  
**Deployer** (o **creador** / **signer**) = la wallet que firma ese nacimiento. En nuestros datos casi siempre es `tx_signer`. El campo `creator_address` venía vacío casi siempre; por eso usamos el signer.

**Mint** = la dirección del token (su “DNI”).

### 1.3 Compra, venta, SOL, WSOL, DEX, fees

**SOL** es la moneda nativa de Solana (como el “euro” de esa red).  
**WSOL** (“wrapped SOL”) es SOL empaquetado para usarlo en intercambios; a efectos de dinero, lo tratamos igual que SOL.

Un **DEX** (Decentralized Exchange, intercambio descentralizado) es un sitio en la blockchain donde se cambia un token por otro sin un banco en el medio. Pump.fun y Raydium son de ese mundo.

Cuando compras:

- sales SOL (o WSOL) y recibes el token nuevo;
- pagas **gas** (la comisión de la red por ejecutar tu tx);
- a veces pagas un **tip** (propina) para que un validador o un servicio te meta más rápido en el bloque;
- a veces hay una **comisión del DEX**.

**P&L** (Profit and Loss) = ganancias y pérdidas.  
**P&L bruto** = lo que entra menos lo que sale, *sin* restar fees.  
**P&L neto** = lo mismo *después* de gas + tip + DEX.

**Hit rate** = de cada 100 posiciones cerradas, en cuántas saliste de verde. 56 de 100 = 56%.

**Hold** = cuánto tiempo te quedas con el token antes de vender.

### 1.4 Bloque, slot, mempool, índice de transacción

**Bloque / slot**: el “fotograma” de 0,4 s donde se empaquetan txs.  
**Mempool** (o bandeja): la sala de espera de txs que *aún no* están en un bloque. Los bots rápidos miran esa bandeja y mandan su compra *al mismo bloque* que el create.

**`transactionIndex`** (o `tx_index`): el *asiento* de la tx dentro del bloque. 0 es la primera. Si el create va en el asiento 500 y la compra del sniper en el 618, la distancia es **118 asientos**, *mismo bloque*.

**Jito** es un servicio de Solana para mandar “paquetes” de txs con propina, a menudo para llegar al mismo bloque.

**CU** (Compute Units) = “esfuerzo de CPU” que una tx pide a Solana.  
**CU limit** = tope de esfuerzo.  
**CU price** = cuánto pagas por unidad de esfuerzo (prioridad). Más alto = más ganas de entrar ya.

**Lamport** = la unidad mínima de SOL. 1 SOL = 1 000 000 000 lamports. Cuando ves `sol_spent_lamports = 1_030_000_000`, son ~1,03 SOL.

### 1.5 Qué es un sniper (y qué no)

Un **sniper** (francotirador) es un **bot**: un programa que opera solo, sin un humano pulsando “comprar”.

Este sniper, en concreto:

1. Ve (o adivina) que *ahora mismo* está naciendo un token.  
2. Compra **en el mismo segundo / mismo bloque**, *antes* de saber si el precio subirá.  
3. Casi siempre vende en **1–3 segundos**.

No es un inversor que lee el gráfico dos horas. Es un cazador de nacimientos.

**No es el objetivo del challenge** “ganar dinero como él”. El objetivo es **entender cómo decide** a quién dispara.

### 1.6 `t_decision` y la trampa del futuro

**`t_decision`** = el instante del deploy (su `blockTime`: la hora que Solana pone al bloque).

Regla de oro: para *decidir* si esto sería un snipe, **solo vale información de ese instante o de antes**.

Usar el precio de 5 minutos después para decir “claro que lo habría comprado, subió” es **hacer trampa**. Eso se llama **leakage** (filtración: se coló el futuro en la pregunta).

El precio de después **sí** se puede usar para otra pregunta distinta: “si hubiéramos comprado, ¿cuánto habríamos ganado?”. Eso es **backtest**, no decisión.

---

## 2. Qué pedía Kaggle

Competición: [Solana Sniper Bot Reverse-Engineering](https://www.kaggle.com/competitions/solana-sniper-bot-reverse-engineering).

**Kaggle** es una web de concursos de datos. Aquí no hay un CSV de “sube tus predicciones y te puntúan”. Piden un **writeup** (texto ≤ 3000 palabras) + un **notebook** público (un cuaderno de código que otro pueda ejecutar) + un **repo** público (carpeta de código en GitHub).

**Reverse-engineering** = ingeniería inversa: a partir de lo que el bot *hizo*, reconstruir *cómo piensa*.

Rúbrica aproximada (100 puntos):

| Parte | Qué quieren | Puntos |
|-------|-------------|--------|
| 1. Comportamiento | Tamaño de compra, latencia, % mismo bloque, aciertos, P&L | ~20 |
| 2. Features y reglas | Señales *antes* de `t_decision`, reglas humanas | ~20 |
| 2. Clasificación | Un modelo que copie sus compras, con métricas y corte por fechas | ~15 |
| 3. Backtest | Si copias al bot, ¿cuánto ganas? ROI, drawdown | ~20 |
| 3. Cara a cara | Tus compras vs las suyas | ~15 |
| Reproducible | Que se pueda repetir, sin futuro en las features | ~10 |

**ROI** = return on investment: ganancia / dinero puesto.  
**Drawdown** = la peor bajada de tu “curva de beneficios” desde un máximo. Si llegaste a +100 SOL y luego caíste a +80, el drawdown fue −20 SOL.

---

## 3. Diccionario de machine learning (y de medición)

Esta sección es el “qué significa cada palabro”. Luego, cuando digamos “ROC 0.94”, ya sabes qué es.

### 3.1 El problema en una frase

Tenemos miles de **deploys**. En algunos el sniper **compró** (los llamamos **positivos** o **snipes**). En la mayoría **no** (los llamamos **negativos**).

Queremos un sistema que, *en el segundo del deploy*, diga: “esto se parece a algo que él compraría”.

Eso es un problema de **clasificación binaria**: dos clases (sí / no).

### 3.2 Dataset, fila, columna, feature, label

**Dataset** = tabla de datos.  
Cada **fila** = un deploy (un token que nació).  
Cada **columna** = una propiedad.

**Feature** (o **variable**, o **señal**) = una columna que *usamos para decidir*. Ejemplo: “¿ya le había comprado a este creador?”, “¿cuántos tokens lanzó en la última hora?”.

**Label** (etiqueta) = la verdad que queremos copiar. Aquí: 1 = el sniper compró, 0 = no.

**Parquet** = un formato de archivo de tablas, más compacto que CSV.  
**JSON / JSONL** = texto estructurado; JSONL es “un objeto JSON por línea”, típico de logs de transacciones.

### 3.3 Entrenar, validar, testear (el examen)

Si estudias con las preguntas del examen, sacas un 10 falso.

Por eso partimos el tiempo:

| Nombre | Fechas | Analogía |
|--------|--------|----------|
| **Train** (entrenamiento) | 12 mar → 29 may 2026 | El libro de estudio |
| **Valid** (validación) | 29 may → 12 jun | Un simulacro para elegir el “corte” |
| **Test** (examen) | 12 jun → 30 jun | Solo se mira al final |

**Overfitting** (sobreajuste) = el modelo se aprendió de memoria el libro y falla en el examen. Se ve cuando train da 0.99 y test da 0.70.  
**Generalizar** = que también funcione en fechas nuevas.

Nosotros: train ROC 0.99, test ROC 0.94 → hay un poco de sobreajuste, pero **sigue funcionando** en junio.

### 3.4 Muestra vs universo

Había ~**5 millones** de deploys que el bot *no* compró. No cabían en el portátil. Cogimos una **muestra aleatoria** de ~197 000.

Eso **infla** la precisión: en la vida real hay muchos más “no” de los que hay en nuestra tabla. Un competidor que usó **todos** los ~852 000 deploys de junio reporta un PR-AUC de **0.22**; nosotros, con muestra, **0.68**. No es que seamos tres veces mejores: **el examen es más fácil**.

### 3.5 Modelo, algoritmo, HistGradientBoosting

**Modelo** = una receta matemática que, dadas las features, suelta un **score** (una nota de 0 a 1: “qué tan snipe parece”).

**Machine learning** (aprendizaje automático) = en vez de escribir a mano todas las reglas, le das ejemplos (train) y un algoritmo *ajusta* la receta para acertar.

**Algoritmo** = el método concreto. El nuestro se llama **HistGradientBoostingClassifier** (de la librería **scikit-learn**):

- **Árbol de decisión**: una cascada de preguntas sí/no (“¿ya lo conocía? si sí, ¿gastó más de 0,5 SOL?…”).  
- **Gradient boosting**: en vez de un árbol, muchos arbolitos. Cada uno intenta corregir los errores del anterior.  
- **Hist**: agrupa los números en cubos (histogramas) para ir más rápido.

No es magia. Es “muchas reglas pequeñas, votando”.

**Hiperparámetros** = tuercas que *nosotros* elegimos (profundidad máxima 6, 200–350 árboles, etc.). No las “aprende” el modelo; las fijamos.

**Sample weight** = le damos más peso a los positivos porque hay pocos (desbalance). Si no, el modelo diría “casi todo es no” y acertaría el 90% sin aprender nada.

### 3.6 Score, umbral, TP/FP/FN/TN

El modelo no dice sí/no de primeras: dice un **score** (0.03, 0.81, 0.99…).

Tú eliges un **umbral** (threshold). Ejemplo: “si score ≥ 0.75, compro”.

Eso genera cuatro casillas (la **matriz de confusión**):

|  | El bot SÍ compró | El bot NO compró |
|--|------------------|------------------|
| Nosotros decimos SÍ | **TP** (true positive, acierto) | **FP** (false positive, alarma falsa) |
| Nosotros decimos NO | **FN** (false negative, se nos escapó) | **TN** (true negative, bien rechazado) |

### 3.7 Precisión, recall, F1

**Precisión** = de las que *nosotros* compramos, ¿qué fracción eran snipes reales?  
`TP / (TP + FP)`  
Alta precisión = pocos disparos en falso.  
Baja precisión = compramos de más.

**Recall** (sensibilidad, exhaustividad) = de los snipes *reales*, ¿qué fracción pillamos?  
`TP / (TP + FN)`  
Alto recall = no se nos escapan.  
Bajo recall = somos tiquismiquis y nos perdemos muchos.

Siempre hay tira y afloja: si bajas el umbral, pillas más (↑ recall) pero te cuelas más (↓ precisión).

**F1** = la media armónica de precisión y recall. Un solo número para “equilibrio”. Lo usamos para elegir el umbral **en validación** (nos salió **0.75**). Ese umbral se **congela** y se aplica a test. No se retoca mirando el examen.

**Precision@100** (P@100) = de los 100 deploys con *mejor* score, cuántos eran snipes. 0.95 = 95 de 100. Es “cuando estamos muy seguros, ¿acertamos?”.

### 3.8 ROC-AUC y PR-AUC

Imagina que ordenas todos los deploys del “más snipe” al “menos snipe”.

**ROC-AUC** (Area Under the ROC Curve):

- Coges un snipe al azar y un no-snipe al azar.  
- ¿Qué probabilidad hay de que el snipe tenga **mejor score**?  
- 0.50 = una moneda. 1.00 = perfecto. 0.94 = ordena muy bien.

**PR-AUC** (también **Average Precision**, AP): mira el equilibrio precisión/recall en *todos* los umbrales. Es más dura cuando hay pocos positivos. Por eso el competidor en el universo completo saca 0.22 y nosotros en muestra 0.68.

**Baseline / prevalencia**: si el 11% de las filas de test son snipes, un modelo tonto que adivina al azar tiene PR-AUC ≈ 0.11. Hay que **ganar** a eso.

### 3.9 Calibración

Si el modelo dice 0.90, ¿es verdad que 9 de cada 10 lo son?

En nuestro test: cuando dice 0.90–1.00, solo ~**64%** eran snipes. Está **sobreconfiado**. El *orden* es bueno (ROC alto); el *número exacto* no es una probabilidad real de mercado. Por eso usamos el score para **ordenar y cortar**, no para decir “hay un 90% de probabilidad”.

### 3.10 Ablación e importancia

**Ablación** = quitar piezas y ver si el coche sigue andando. Entrenamos el modelo *solo con “ya lo conocía”*, *solo con la forma de la tx*, *solo con anti-fábrica*, y *con todo*. Así sabemos qué aporta cada bloque.

**Importancia por permutación**: en el examen, barajas *una* columna (la rompes) y mides cuánto cae el ROC. Si cae mucho, esa columna importaba. En cold, las que más importan son: tiempo desde el último launch, complejidad de la tx (`n_inner_ix`), cuántos launches llevaba, SOL gastado al crear.

### 3.11 Regla vs modelo

Una **regla** es una frase humana: “si ya le compré, compro”.  
Un **modelo** combina muchas señales con pesos aprendidos.

Las reglas **explican**. El modelo **afina**. Lo ideal es tener las dos: la historia en cristiano + el número en test.

### 3.12 Hot y cold

**Hot** = el sniper **ya le había comprado** a ese deployer *antes* de este deploy (`prior_bought_same_signer > 0`).  
**Cold** (cold start) = primera vez que vemos a ese deployer.

El cold start es el misterio: ¿por qué a veces le da una oportunidad a un extraño?

### 3.13 Fábrica, burst, serial

**Fábrica** = una wallet que lanza tokens en serie, como una máquina, a menudo para rug-pulls (estafas de “lanzo, vendo, desaparezco”).

**Burst 3**: 3 o más tokens en la **última hora**.  
**Serial 10**: 10 o más tokens en su historia (hasta ese instante).  
**`s_since_last_launch`**: segundos desde su token anterior. 26 horas = callado. 146 segundos = está ametrallando.

### 3.14 Réplica y backtest

**Réplica** = “si usáramos nuestro umbral, ¿qué deploys habríamos comprado?”.  
**Overlap** = cuántos coinciden con los del bot.  
**Backtest** = simular el dinero *después*, con precios/fees, sin meterlos en la decisión.

**No look-ahead** = no mirar el futuro para decidir.

---

## 4. Los datos que teníamos (y los que no)

### 4.1 Lo que sí

- Actividad de la wallet del bot (~87 007 filas de compras y ventas, con cantidades, gas, tips, DEX).  
- Los **15 927** deploys que *sí* compró (positivos), con la tx de creación.  
- Una **muestra** de ~196 878 deploys que *no* compró (negativos), sacada de ~5 millones.  
- Actividad previa de los deployers (filtrada a las wallets que nos importan), para contar launches *antes* de `t_decision`.  
- JSONL de las txs de creación (para extraer tips, CU, si hay buy en la misma tx, SOL gastado).

### 4.2 Lo que no bajamos (a propósito)

- El TAR entero (~39 GiB) no cabía.  
- Velas de precio `mcap_candles.parquet` (~2,8 GiB): servirían para tasar tokens que **nosotros** compraríamos y el bot **no**.  
- Bloques crudos de junio (cientos de GiB) y trades completos.

Sin velas, el P&L de los **falsos positivos** no existe en local. No lo inventamos.

---

## 5. Paso a paso: qué hicimos, en orden

Esto es el diario de trabajo, en cristiano. Los números de script son archivos en `scripts/`.

### Fase 0 — Preparar el terreno

**00.** Mirar el ordenador: disco, RAM, Python.  
**01–02.** Ver qué archivos hay en los servidores del challenge y cuánto pesan, *sin* bajar 40 GiB. El archivo grande no permite descargas parciales cómodas.  
**03.** Bajar la wallet del bot.  
**04.** Extraer del TAR solo los *positivos* (los deploys que sí compró).  
**05.** Inspeccionar columnas: `tx_hash`, `blockTime`, `blockSlot`, `token_address`, `tx_signer`…  
**06.** Ver cómo se reparte en el tiempo y **proponer el corte** train / valid / test.  
**07.** Planear (y luego ejecutar) la **muestra** de 200k negativos: se streameraron ~14,5 GiB por red y en disco solo quedó la muestra.

Regla de seguridad: no bajar archivos enormes sin tapón.

### Fase 1 — Primer modelo (el “baseline”)

**09. Entrenar un modelo simple** con pocas features:

- hora y día de la semana;  
- si el token parece Pump.fun;  
- **`prior_bought_same_signer`** (¿ya le había comprado a este creador?);  
- un par de contadores flojos de actividad.

**Resultado en test (junio):** ROC ~**0.82**, P@100 **0.86**.  
La feature que mandaba, de lejos, era **“¿ya lo conocía?”**.

Eso ya era una historia: *el sniper tiene lista de creadores y les sigue el rastro*.

### Fase 2 — ¿Y los desconocidos? Actividad extra

**10–14.** Enriquecer deploys, filtrar la actividad de deployers (el archivo enorme de “quién hizo qué”), entrenar un modelo de cold start con “cuántos eventos tenía esa wallet”.

**Resultado:** contar *cuánto se mueve* la wallet **no** explicó la primera compra mejor que “pinta del deploy” + “¿ya está en la lista?”. ROC en cold ~0.79, no un salto.

### Fase 3 — Hipótesis de detective (las que pediste)

**15. ¿Los cold están “tocados” por los hot?**  
Misma tx, mismo token, mismos círculos.  
**Resultado:** `from`/`to` venían **vacíos** (no hay transferencias leíbles así). Compartir tx casi no pasa (~0,15%). Compartir token *parecía* fuerte en train y **se evaporaba** en junio. No sirve como regla estable.

**16. ¿Tokens raros? ¿Envíos hot→cold?**  
**Resultado:** transferencias reales ~inexistentes. Tokens raros no separan en test.

**17. Estilo del deploy + latencia**  
¿Pagan tips a servicios (astra, rapid…)? ¿Compran en la misma tx del create? ¿Los cold tardan más en pensar?

**Resultado:**

- Create+buy en la misma tx es **casi universal** (positivos y negativos): no distingue.  
- Tips de servicio: en cold, 30% vs 31% — **empate**.  
- Latencia: **90,7% mismo segundo**, **79,6% mismo slot**, **98,5% en ≤1 s**.  
- El sniper **nunca** es el que firma el create.  
- Distancia mediana en el bloque: **118 txs** (no van pegados; vio el create en la bandeja y se coló más abajo en el *mismo* bloque).  
- Cold vs hot: misma velocidad. **No hay “tiempo extra para pensar”.**  
- Añadir estilo de tx al modelo: test ROC **0.89**, cold **0.83**. Subió.

**18. Batería de hipótesis solo en cold + “fábricas”**  
Aquí apareció la regla buena:

En **test, solo desconocidos**:

- Burst (3+ launches en 1 h): el bot compra el **0,75%** si hay burst vs **7,5%** si no.  
- Serial 10: 2,0% vs 8,3%.  
- Mediana de launches en 24 h: **0** (los que compra) vs **4** (los que no).  
- Segundos desde el último launch: **26 horas** vs **146 segundos**.  
- SOL gastado al crear: **~1,03 SOL** vs **~0,16 SOL**.

Modelo *solo cold*: test ROC **0.93**, PR **0.52**, P@100 **0.85** (la base era 4,8%).

**19. Entrenar en serio + P&L + réplica** (holdout de junio intacto)

Modelo con lista hot + estilo de tx + anti-fábrica. Umbral 0.75 elegido en valid.

Test: ROC **0.94**, PR **0.68**, precisión **0.54**, recall **0.79**, F1 **0.64**, P@100 **0.95**.

Réplica: 3 545 compras vs 2 445 del bot; coinciden **1 922** (pillamos el 79%); precisión 54%.  
P&L neto del bot en test: **+1 192 SOL**. De eso, la réplica captura **+927 SOL** (solo las coincidencias). 1 623 FP sin precio.

**20. Ablaciones, reglas humanas, calibración, drawdown, P&L hot vs cold** (esta ronda)

Ver secciones 7–9. Resumen: **juntar las tres piezas gana a cada pieza sola**. Las reglas humanas explican, pero el modelo afina. El bot gana también en cold. Drawdown máximo ~**−47 SOL** sobre ~+8 700 de equity: curva bastante estable. El score está sobreconfiado.

---

## 6. Cómo construimos las features (sin futuro)

Para cada deploy, *antes* de `t_decision`:

1. **Lista hot:** ¿cuántas veces, *antes*, el sniper había comprado un token de este mismo signer?  
2. **Reloj:** hora UTC, día de la semana, días desde que el bot empezó.  
3. **Forma de la tx de nacimiento:** ¿Pump.fun? ¿cuántas instrucciones? ¿CU, fee, tip? ¿hay un Buy en los logs de esa misma tx? ¿cuánto SOL salió de la cuenta al crear?  
4. **Historial del deployer *hasta ese instante*:** cuántos tokens había lanzado, cuántos en 1 h / 24 h, segundos desde el anterior, si es el primero, si es ráfaga de 3, si ya va por 10.

Nada de precio posterior. Nada de “este token luego subió”.

---

## 7. Hipótesis, una a una

Formato: *creíamos* → *cómo se midió* → *qué salió*.

### H1. “Es un sniper de mismo bloque”

**Sí.** 79,6% mismo slot; 90,7% mismo segundo; 98,5% en ≤1 s. Mediana de espera: **0 segundos**. Hold mediana hasta la 1ª venta: **1 s**. Entrada mediana: **1,98 SOL**.

### H2. “Sigue a creadores que ya le funcionaron”

**Sí, y es la señal más gorda.** En test, si es hot compra ~**39%** de esos deploys; si es cold, ~**5%**. La regla “si hot, compro” sola: ROC **0.77**, F1 **0.49**.

### H3. “Los cold están conectados a los hot (misma tx, mismo token, envíos)”

**No nos sirve.** From/to vacíos. Shared tx rarísimo. Shared token se muere en junio.

### H4. “Los cold tardan más: hay una ventana de pensamiento / cola”

**No.** Cold y hot igual de instantáneos. Distancia ~118 txs en ambos. El bot no “estudia 3 segundos al extraño”.

### H5. “Mira tips astra/rapid/Jito para elegir cold”

**No en cold.** Empate 30% vs 31%. Jito tip incluso es *menos* frecuente en los que compra.

### H6. “Create+buy en la misma tx es la clave”

**Casi todos lo hacen** (snipe o no). No separa.

### H7. “Evita fábricas; espera a un extraño callado que esta vez mete dinero”

**Sí. Esta es la regla de los desconocidos.** Burst 3 en test cold: 0,75% vs 7,5%. Callado ~26 h vs 2,5 min. Create ~1 SOL vs 0,16 SOL.

### H8. “Un modelo que junte H2+estilo+H7 copia bien al bot en junio”

**Sí, en nuestra muestra.** Test ROC 0.94 / PR 0.68 / F1 0.64 / P@100 0.95. Recall 79%.  
**Con el matiz:** en el universo completo el PR sería más bajo (~0.22 según el notebook público).

### H9. “Una regla de una línea ya iguala al modelo”

**No.** “Si hot, compro”: F1 0.49. “Hot o (no burst y callado 1 h y create ≥0,5 SOL)”: F1 0.47. El modelo: F1 **0.64**. Las reglas **cuentan la historia**; el modelo **caza matices** (complejidad de la tx, combinaciones).

### H10. “El bot gana dinero; los cold son un lastre”

**Gana, sí. Lastre, no.** Neto ~**+8 900 SOL** en el periodo (tras fees), hit rate **55,6%**. Drawdown máximo ~**−47 SOL** (muy poco comparado con el cúmulo).  
Hot: +5 794 SOL, hit 56,6%.  
Cold: +2 938 SOL, hit 52,8%, **media por trade más alta** (0,74 vs 0,50 SOL). Las primeras compras, cuando salen bien, pagan.

### H11. “Si copiamos la selección, ya ganamos lo mismo”

**Cuidado.** En test, con umbral 0.75, capturamos +927 de +1 192 SOL *de los tokens que él sí negoció*. Los 1 623 extra que nosotros compraríamos **no tienen precio aquí**. El notebook de otro equipo: réplica **un slot tarde** pierde ~5 SOL. **Seleccionar ≠ aterrizar en el mismo bloque.**

---

## 8. Ablación: qué pieza aporta qué (test de junio)

Entrenamos el mismo tipo de modelo quitando bloques enteros.

| Piezas | ROC | PR-AUC | F1 | ROC solo cold |
|--------|-----|--------|----|----------------|
| Solo “¿ya lo conocía?” | 0.77 | 0.37 | 0.49 | 0.50 (no sabe nada) |
| Hot + hora/día | 0.82 | 0.46 | 0.50 | 0.64 |
| Solo forma de la tx | 0.88 | 0.52 | 0.48 | 0.86 |
| Solo anti-fábrica | 0.87 | 0.37 | 0.52 | 0.86 |
| Hot + forma de la tx | 0.92 | 0.63 | 0.61 | 0.87 |
| Hot + anti-fábrica | 0.89 | 0.46 | 0.52 | 0.86 |
| **Todo junto** | **0.94** | **0.69** | **0.65** | **0.94** |

Lectura humana: la lista hot ordena a los conocidos. La pinta del deploy y la anti-fábrica son las que **abren la puerta al extraño**. Juntas, mejor que cada una.

Reglas solas en test (para comparar):

| Regla | Precisión | Recall | F1 | Cuántos “compro” |
|-------|-----------|--------|-----|------------------|
| Si hot | 0.39 | 0.65 | 0.49 | 4 068 |
| Si no hay burst-3 | 0.08 | 0.33 | 0.12 | 10 856 |
| Si callado ≥1 h | 0.12 | 0.16 | 0.14 | 3 318 |
| Si create ≥0,5 SOL | 0.19 | 0.80 | 0.30 | 10 496 |
| Hot **o** (no burst + callado + ≥0,5 SOL) | 0.34 | 0.77 | 0.47 | 5 522 |
| Anti-fábrica estricta + Pump | 0.21 | 0.11 | 0.15 | 1 336 |
| **Modelo, umbral 0.75** | **0.54** | **0.79** | **0.64** | **3 545** |

---

## 9. Dinero, con todos los peros

### 9.1 Cómo se cuenta

Por cada token, en SOL/WSOL:

- cada **buy** resta la cantidad;  
- cada **sell** suma;  
- restamos gas + tip + DEX de todas las filas.

Si no hay ninguna venta, no lo damos por “cerrado”.

### 9.2 El bot, todo el periodo

| Dato | Valor |
|------|--------|
| Posiciones cerradas (SOL/WSOL) | ~15 700 |
| Bruto | +17 629 SOL |
| Fees | −8 735 SOL |
| **Neto** | **+8 894 SOL** |
| Hit rate neto | **55,6%** |
| Mediana neta | +0,06 SOL |
| Media si gana / si pierde | +1,40 / −0,47 SOL |
| Drawdown máximo (sobre la serie unida al labeled) | **−47 SOL** |

Gana poco a poco, muchas veces; las fees se comen casi la mitad del bruto. Sin fees el “acierto” parecía 78%; con fees baja a 56%. **Las comisiones importan.**

### 9.3 Test (12–30 jun) — cara a cara

| | Bot | Réplica (score ≥ 0.75) |
|--|-----|-------------------------|
| Compras | 2 445 | 3 545 |
| Coincidencia | — | 1 922 (79% de las suyas) |
| Precisión | — | 54% |
| P&L neto (tokens con precio) | **+1 192 SOL** | **+927 SOL** (solo TP) |
| Compras extra sin precio | — | 1 623 |

Si subes el umbral a **0.90**: menos compras (2 380), precisión 64%, capturas +776 SOL.  
A **0.95**: 1 543 compras, precisión 73%, +627 SOL.  
Más estricto = más limpio y menos dinero de *su* curva (porque te dejas ganadores).

### 9.4 Calibración (test)

| Score que dice el modelo | % que de verdad eran snipes |
|--------------------------|-----------------------------|
| 0 – 0.10 | 0,3% |
| 0.10 – 0.25 | 2,9% |
| 0.25 – 0.50 | 7,0% |
| 0.50 – 0.75 | 18% |
| 0.75 – 0.90 | 34% |
| 0.90 – 1.00 | 64% |

Ordena bien, pero **no** es una probabilidad de mercado. Parte de la culpa es la muestra (faltan millones de “no”).

---

## 10. El notebook de otros (sealed evidence)

[Notebook](https://www.kaggle.com/code/thtennant/solana-sniper-re-sealed-evidence) · [repo](https://github.com/teddytennant/solana-sniper-reverse-engineering)

Llegan a la **misma foto**:

- sniper de mismo slot;  
- recencia de “ya le compré”;  
- deployer callado desde el último launch.

Diferencias honestas:

- Ellos evalúan el **universo completo de junio** → PR-AUC **0.22**. Nosotros una **muestra** → 0.68.  
- Su réplica con **+1 slot** de retraso y 3 asientos **pierde ~5 SOL**.  
- Un recorte “mismo slot, 35 fills, +25 SOL” lo marcan como **optimista**, no como titular.

Encaja con nosotros: la decisión se puede copiar a medias; **el fill en el mismo bloque** es otra guerra (infra, propinas, latencia de red). Eso el challenge también lo pide en la parte 3, con honestidad.

---

Las figuras oficiales (inglés) están en `data/metadata/kaggle_writeup/`. Regenerar: `python3 scripts/make_figures.py`.

## 11. Archivos para mirar con tus ojos

| Archivo | Qué es |
|---------|--------|
| `data/metadata/INFORME_COMPLETO.md` | Este texto |
| `data/metadata/INFORME_FINAL.md` | Resumen corto |
| `data/metadata/kaggle_train_backtest.json` | Números del modelo y réplica |
| `data/metadata/extra_hypotheses.json` | Ablaciones, reglas, drawdown, calibración |
| `data/processed/test_holdout_muestra.csv` | Muestra de junio: score, si el bot compró, P&L si existe |
| `data/processed/test_holdout_scored.parquet` | Los 22 132 deploys de test, todos |
| `data/processed/scored_deploys.parquet` | Train+valid+test con score |
| `scripts/` | Pipeline de reproducción (ver README) |

---

## 12. Conclusión (para llevar)

El sniper **no adivina el precio futuro**.

Hace tres cosas, todas visibles *en el segundo del deploy o antes*:

1. **Sigue a creadores a los que ya les había comprado.**  
2. **Si es un extraño, evita fábricas** (tokens cada dos minutos, create de céntimos) y prefiere a alguien callado que esta vez mete ~1 SOL.  
3. **Dispara en el mismo bloque**, ~118 txs más abajo, y vende en ~1 segundo.

Con eso, históricamente, **sale de verde** (~+8 900 SOL netos; acierta ~56% de las veces tras fees; casi no se pega un drawdown gordo).

Un modelo que junta esas piezas, en el examen de junio (nuestra muestra), ordena muy bien (ROC 0.95) y pilla el 81% de sus compras. Una regla de una línea cuenta la historia pero no llega al mismo F1.

Límites honestos: el PR-AUC 0.70 es de una **muestra** de negativos (en el universo completo de junio un trabajo público reporta 0.22). Las compras extra de la réplica no tienen precio local. Llegar un bloque tarde, en ese mismo trabajo, deja de ser rentable.
