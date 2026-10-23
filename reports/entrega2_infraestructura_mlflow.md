# Entrega 2: infraestructura de MLflow (rama `feat/mlflow-project`)

Aquí les explico qué hice en el PR de infraestructura de la entrega 2, por qué lo hice así y qué nos falta para terminar. La idea es que, con esto, cada quien solo tenga que **agregar modelos y experimentos de su pregunta**, sin preocuparse por el tracking, los artefactos ni el registro.

## En corto

Ya funciona todo de punta a punta. Lo probé desde un clon limpio con un servidor de MLflow vacío:

```bash
make mlflow-server   # terminal 1 (puerto 5001)
make all             # terminal 2: ingesta + 23 runs + 3 modelos registrados con alias champion
```

| Lo que pide la entrega | Cómo va |
|---|---|
| Servidor de tracking con SQLite, documentado en el README | Listo |
| `MLproject` con entorno (`python_env.yaml`) y puntos de entrada `ingest` y `train` | Listo (también agregué `search` y `register`) |
| Valores por defecto desde `config.py`, sin repetir rutas, fechas ni semillas | Listo |
| Todas las corridas con `mlflow run` | Listo (`make train`, `make search`, `make all`) |
| Etiquetas `question`, `model_family`, `stage_of_work` | Listo |
| Artefactos: modelo con firma y ejemplo, manifiesto, `log_input`, gráficas y configuración, en subcarpetas | Listo |
| Búsqueda con nested runs | La infraestructura está lista y hay una búsqueda de ejemplo en P3 |
| **3 o más versiones adicionales al baseline por pregunta** | **Falta** (abajo explico) |
| Model Registry: nombres, descripción, linaje, alias `champion` y carga por alias | La infraestructura está lista; **falta elegir y justificar el champion** |
| Reporte PDF con tablas, análisis, capturas y APA 7 | **Falta** |
| Que cada quien abra y revise al menos un PR | **Falta** (este es mi PR) |

## Qué hice y por qué

Organicé los commits por tema para que sea fácil revisarlos:

1. **`feat(mlflow)`: servidor, `MLproject` y `python_env.yaml`.**
   - El Model Registry necesita una base de datos, así que el servidor usa `sqlite:///mlflow.db`.
   - Lo puse en el puerto **5001** porque en Mac el 5000 lo ocupa AirPlay. Si quieren otro puerto, lo cambian con `MLFLOW_PORT` (o directo con `MLFLOW_TRACKING_URI`).
   - `python_env.yaml` reutiliza `requirements.txt`, así las versiones viven en un solo lugar.
   - `mlflow run` arma su propio entorno con `uv`. Mientras desarrollan, `ENV_MANAGER=local` usa el entorno que ya tienen activo y es mucho más rápido.
2. **`refactor(modeling)`: separé `train.py` en un archivo por pregunta** (`p1.py`, `p2.py`, `p3.py`, más `tracking.py` y `common.py`). Comprobé que los 12 modelos que ya teníamos dan exactamente las mismas métricas que antes.
3. **`feat(modeling)`: hiperparámetros desde `config.py` y punto de entrada `train`.**
   - Cada pregunta tiene su tabla de modelos (`P1_MODELS`, etc.), y los hiperparámetros por defecto están en `config.py`.
   - Para cambiarlos sin tocar código: `-P params="k=v,k2=v2"`. Los valores se convierten al tipo correcto, y si escriben mal un nombre, marca error.
   - El nombre del run muestra lo que cambiaron, por ejemplo `p1-hgb-regressor-lags-calendario[learning_rate=0.1]`.
   - Un detalle técnico por si les pasa: con `mlflow run`, el run ya existe antes de que corra nuestro código, y MLflow ignora el nombre y las etiquetas que le pasamos a `start_run`. Por eso `start_run` en `tracking.py` los pone después con `set_tags`.
4. **`feat(modeling)`: artefactos en subcarpetas.** Todo está en `artifacts.py`:
   - **`model`:** el modelo con su firma y un ejemplo de entrada. En MLflow 3 aparece como *Logged Model* ligado al run.
   - **`data/`:** el manifiesto y un `dataset.json`, más `mlflow.log_input` del conjunto de entrenamiento.
   - **`plots/`:** las gráficas de cada tarea. En P1 agregué una de real vs. predicho y residuales.
   - **`config/`:** los hiperparámetros con que corrió y una copia de `config.py`.
   - **`metrics/`:** las métricas en JSON y, en P3, a qué grupo quedó cada estación.
   - **El umbral de P2:** el clasificador se guarda junto con el umbral que elegimos en validación (`ThresholdClassifier`). Así, cuando alguien carga el modelo desde el registro, decide igual que en el experimento.
   - **La ingesta también queda registrada** como run, con nombre `ingest-<data_version>` y el manifiesto.
5. **`feat(modeling)`: búsqueda con nested runs** (`search.py`).
   - Se usa así: `make search Q=p1 M=hgb G="learning_rate=0.03|0.1;max_leaf_nodes=31|63"`.
   - Crea un run padre y un run hijo por cada combinación. Cada hijo es un entrenamiento normal, con todos sus artefactos.
   - El padre elige al mejor hijo con la métrica de `SELECTION_METRIC` (en `config.py`), siempre en validación y nunca en test. Guarda la tabla `search/results.csv` y la etiqueta `best_run_id`.
6. **`feat(registry)`: Model Registry con aliases** (`registry.py`).
   - Los nombres están fijos en `config.py`: `mibici-equipo34-p1-demanda`, `-p2-riesgo` y `-p3-grupos`.
   - Cada versión trae descripción y etiquetas de linaje: de qué run salió, el commit, `data_version`, `train_data_hash`, la métrica y, si salió de una búsqueda, el run padre.
   - Usé **aliases** (`champion` y, si queremos, `challenger`) porque MLflow ya marcó los stages (Staging/Production) como obsoletos.
   - Después de registrar, revisa que `models:/<nombre>@champion` cargue y prediga. `predict.py` ya usa el champion.
   - El run se elige **por nombre o por la mejor métrica, nunca por `run_id`**. Abajo explico por qué.
7. **`feat(experiments)`: `scripts/run_experiments.sh` y `make all`.** Ese script es la lista oficial de runs de la entrega: si un modelo no está ahí, no sale en la evidencia final.
8. **`refactor(modeling)`: simplifiqué el código** para que sea más fácil de leer, sin cambiar ningún resultado.

### Ojo: las cifras de P2 de la primera entrega ya cambiaron

El [PR #12](https://github.com/diegofalconcostilla/TC5061-Proyecto-Equipo34/pull/12) ajustó `block_start` en `features.py`, y ahora los bloques que empiezan antes de las 06:00 ya no entran al dataset de P2.

- **Qué cambió:** las métricas de P2 se movieron un poquito. La proporción de bloques en riesgo en prueba pasó de 0.0252 a 0.0246, y el PR-AUC de la tasa histórica bajó de 0.320 a 0.315.
- **Por qué no nos dimos cuenta:** `data_version` no cambió, porque solo cubre los datos crudos y la limpieza, no `features.py`.
- **Qué hice:**
  - Ahora cada run guarda `train_data_hash`, un hash del dataset ya construido. Si alguien cambia `features.py`, se va a notar.
  - Actualicé las cifras de P2 en el README.
  - **La propuesta (`docs/TC5061_Propuesta_Proyecto_Equipo34.docx`) todavía tiene las cifras viejas.** Para el reporte usemos las nuevas y mencionemos el cambio.

| Run de P2 | test_pr_auc | test_f1 | test_recall | valid_pr_auc |
|---|---|---|---|---|
| `p2-baseline-persistencia-semanal` | 0.388 | 0.350 | 0.350 | 0.397 |
| `p2-baseline-tasa-historica-estacion-bloque` | 0.315 | 0.335 | 0.434 | 0.376 |
| `p2-logistica-flujos-lag` | 0.405 | 0.414 | 0.445 | 0.405 |
| `p2-hgb-classifier-flujos-lag` | 0.464 | 0.445 | 0.421 | 0.483 |

## Cómo juntamos la evidencia (importante)

Cada quien tiene su propio `mlflow.db`, y ese es **solo para pruebas**. Para que la evidencia sea consistente, propongo que la corrida oficial sea **una sola**:

1. Hacemos merge de todos los PRs de modelos a `main`.
2. Alguien clona `main` en una carpeta nueva, levanta un servidor vacío y corre `make all`.
3. De ese servidor sacamos las capturas y las tablas del reporte.

Así todos los runs quedan con el mismo commit, `git_dirty=false` y la misma versión de datos. Por eso el registro no usa `run_id`: cambian en cada base de datos. Los nombres de los runs y las métricas, en cambio, no cambian.

## Cómo agregar un modelo

1. En `mibici/config.py`, agreguen los hiperparámetros por defecto en `P1_MODEL_PARAMS` (o P2, P3).
2. En `mibici/modeling/pN.py`:
   - escriban la función `(parts, hp) -> ModelResult`, copiando el estilo de las que ya están. Que devuelva `estimator` y `features` para que se guarde el modelo;
   - agréguenla a la tabla `PN_MODELS` con `ModelSpec(funcion, "<familia>", "candidate", PN_MODEL_PARAMS["<nombre>"])`.
3. Pruébenla con `make train ENV_MANAGER=local Q=pN M=<nombre>` y, si aplica, con `make search ENV_MANAGER=local Q=pN M=<nombre> G="k=a|b;k2=c|d"`.
4. Agreguen sus líneas `train` o `search` en la sección de su pregunta en `scripts/run_experiments.sh`.
5. Corran `make lint` y `make test`, y abran su PR.

Las etiquetas, los artefactos, la firma del modelo y el registro salen solos. Las métricas con las que se elige al mejor son P1 `valid_wape` (menor es mejor), P2 `valid_pr_auc` y P3 `train_profile_silhouette` (mayor es mejor). Si queremos usar otra, se cambia en `SELECTION_METRIC`.

## Qué falta

Esto lo repartimos como decidamos:

- **Experimentos de P1:** hoy hay 3 baselines y 2 modelos (Poisson y HGB). La entrega pide **al menos 3 versiones adicionales al baseline**. Ideas: otro algoritmo (Random Forest, o LightGBM si lo agregamos a `requirements.txt`), variantes de features o una búsqueda sobre el HGB.
- **Experimentos de P2:** hay 2 baselines y 2 modelos (logística y HGB). Faltan más versiones y una búsqueda, por ejemplo sobre `learning_rate` y `max_leaf_nodes` del HGB, o sobre `C` de la logística.
- **Experimentos de P3:** hay 2 baselines, el KMeans por perfiles y una búsqueda de ejemplo sobre `n_clusters` y `n_init`. Ideas: Gaussian Mixture, Agglomerative o PCA antes de KMeans.
- **Elegir y justificar el champion de cada pregunta:** desempeño contra el baseline, la métrica que importa para el negocio, estabilidad y costo. Si el mejor por métrica no es el que queremos, se fija con `register pN "<run_name>"` en el script. Si queremos, también un `challenger` por pregunta.
- **La corrida oficial** desde un clon limpio de `main`, cuando todo esté mergeado, y las **capturas**:
  - la comparación con nested runs;
  - los artefactos de una candidata;
  - los tres modelos registrados con su alias y linaje.

  Las capturas van en `reports/figures/mlflow/`.
- **El reporte PDF** `TC5061_Reporte_MLFlow_Equipo34.pdf`, con:
  - el link al repo y los comandos de `mlflow run`;
  - una tabla por pregunta con validación, prueba y mejora contra el baseline;
  - el análisis y por qué elegimos cada champion;
  - el esquema de aliases, las capturas y las referencias en APA 7.
- **PRs y revisiones:** cada quien abre al menos un PR y revisa al menos uno.
- **Actualizar la tabla de resultados del README** con los modelos nuevos y los champions.

## Qué probé

- `make lint` y `make test` pasan: 24 pruebas, y las nuevas cubren hiperparámetros, mallas de búsqueda y defaults.
- Los 12 modelos que ya teníamos dan las mismas métricas antes y después del refactor. P2 solo cambia por el PR #12, y comprobé que el código original de `main` da lo mismo con los datos actuales.
- Cloné la rama en una carpeta limpia, levanté un servidor vacío y corrí `make all` con `uv`:
  - salieron 23 runs, todos con el mismo commit y `git_dirty=false`;
  - `data_version` `efa3a16661ec`;
  - los tres modelos quedaron registrados con `champion` y se pueden cargar por alias.
- `make predict Q=p1|p2|p3` saca predicciones de test con el champion de cada pregunta.
