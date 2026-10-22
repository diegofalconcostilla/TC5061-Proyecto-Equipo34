# Entrega 2: infraestructura de MLflow (rama `feat/mlflow-project`)

Este documento resume lo que hace el PR de infraestructura de la segunda entrega, por qué se decidió así y qué falta para completar la entrega. La idea es que con esto cada integrante solo tenga que **agregar modelos y experimentos de su pregunta**, sin tocar el tracking, los artefactos ni el registro.

## Resumen

Ya funciona de punta a punta, probado desde un clon limpio con un servidor de MLflow vacío:

```bash
make mlflow-server   # terminal 1 (puerto 5001)
make all             # terminal 2: ingesta + 23 runs + 3 modelos registrados con alias champion
```

| Requisito de la entrega | Estado |
|---|---|
| Servidor de tracking con SQLite, documentado en el README | Hecho |
| `MLproject` con entorno (`python_env.yaml`) y puntos de entrada `ingest` y `train` | Hecho (además `search` y `register`) |
| Valores por defecto desde `config.py`, sin repetir rutas, fechas ni semillas | Hecho |
| Todas las corridas con `mlflow run` | Hecho (`make train`, `make search`, `make all`) |
| Etiquetas `question`, `model_family`, `stage_of_work` | Hecho |
| Artefactos: modelo con firma y ejemplo, manifiesto, `log_input`, gráficas, configuración, en subcarpetas | Hecho |
| Búsqueda con nested runs | Infraestructura hecha y una búsqueda de ejemplo (P3) |
| **3 o más versiones adicionales al baseline por pregunta** | **Falta** (ver abajo) |
| Model Registry: nombres, descripción, linaje, alias `champion` y carga por alias | Infraestructura hecha; **falta elegir el champion justificado** |
| Reporte PDF con tablas, análisis, capturas y APA 7 | **Falta** |
| Cada integrante abre y revisa al menos un PR | **Falta** (este es el PR de Oscar) |

## Qué se hizo y por qué

Los commits del PR están ordenados por tema:

1. **`feat(mlflow)`: servidor, `MLproject` y `python_env.yaml`.**
   - El Model Registry necesita una base de datos, por eso el servidor usa `sqlite:///mlflow.db`.
   - Usa el puerto **5001** porque en macOS el 5000 lo ocupa AirPlay. Se puede cambiar con `MLFLOW_PORT`, o directamente con `MLFLOW_TRACKING_URI`.
   - `python_env.yaml` reutiliza `requirements.txt`, así las versiones fijas viven en un solo lugar.
   - Por defecto `mlflow run` crea su propio entorno con `uv`. Con `ENV_MANAGER=local` usa el entorno activo, que es más rápido mientras se desarrolla.
2. **`refactor(modeling)`: `train.py` separado en módulos por pregunta** (`p1.py`, `p2.py`, `p3.py`, más `tracking.py` y `common.py`). Las métricas de los 12 modelos existentes se verificaron idénticas antes y después del refactor.
3. **`feat(modeling)`: hiperparámetros desde `config.py` y punto de entrada `train`.**
   - Cada pregunta tiene una tabla de modelos (`P1_MODELS`, etc.), con sus hiperparámetros por defecto en `config.py`.
   - Los defaults se cambian con `-P params="k=v,k2=v2"`. Los valores se convierten al tipo del default, y una clave desconocida da error.
   - El nombre del run muestra lo que cambió respecto al default, por ejemplo `p1-hgb-regressor-lags-calendario[learning_rate=0.1]`.
   - Detalle técnico importante: bajo `mlflow run`, el run ya existe antes de que corra nuestro código, y `mlflow.start_run` ignora el nombre y las etiquetas que le pasemos. Por eso `start_run` en `tracking.py` los fija después con `set_tags`.
4. **`feat(modeling)`: artefactos en subcarpetas.** `artifacts.py` concentra los helpers:
   - **`model`:** el modelo con firma e `input_example` (en MLflow 3 aparece como *Logged Model* ligado al run).
   - **`data/`:** el manifiesto y `dataset.json`, más `mlflow.log_input` del conjunto de entrenamiento.
   - **`plots/`:** las gráficas de cada tarea. Para P1 se agregó real vs. predicho y residuales.
   - **`config/`:** los hiperparámetros resueltos y una copia de `config.py`.
   - **`metrics/`:** las métricas en JSON y, en P3, la asignación de estaciones a grupos.
   - **Umbral de P2:** el clasificador de P2 se guarda dentro de un `ThresholdClassifier`. Así, el modelo cargado desde el registro aplica el mismo umbral elegido en validación (`predict` da la etiqueta y `predict_proba` el score).
   - **Run de ingesta:** también queda registrado, con nombre `ingest-<data_version>`, etiqueta `stage_of_work=ingest` y el manifiesto.
5. **`feat(modeling)`: búsqueda con nested runs** (`search.py`).
   - Se usa como `make search Q=p1 M=hgb G="learning_rate=0.03|0.1;max_leaf_nodes=31|63"`.
   - Crea un run padre y un run hijo por combinación. Cada hijo es un entrenamiento normal, con los mismos artefactos.
   - El padre elige al mejor hijo con `SELECTION_METRIC` de `config.py`, siempre en validación y nunca en test. Guarda `search/results.csv` y la etiqueta `best_run_id`.
6. **`feat(registry)`: Model Registry con alias** (`registry.py`).
   - Los nombres fijos están en `config.py`: `mibici-equipo34-p1-demanda`, `-p2-riesgo` y `-p3-grupos`.
   - Cada versión lleva descripción y etiquetas de linaje: run de origen, commit, `data_version`, `train_data_hash`, métrica y, si viene de una búsqueda, el run padre.
   - Usamos **aliases** (`champion` y opcionalmente `challenger`) porque MLflow marcó los stages como obsoletos.
   - Después de registrar se verifica que `models:/<nombre>@champion` carga y predice. `predict.py` ya usa el champion.
   - El run se elige **por nombre o por la mejor métrica, nunca por `run_id`** (ver la sección de consolidación).
7. **`feat(experiments)`: `scripts/run_experiments.sh` y `make all`.** Ese script es la lista oficial de runs de la entrega. Un modelo que no esté ahí no aparece en la evidencia final.

### Hallazgo: las cifras de P2 de la primera entrega ya no son las actuales

El [PR #12](https://github.com/diegofalconcostilla/TC5061-Proyecto-Equipo34/pull/12) ajustó `block_start` en `features.py`, y desde entonces los bloques que empiezan antes de las 06:00 ya no entran al dataset de P2.

- **Efecto:** las métricas de P2 cambiaron un poco. La prevalencia de riesgo en prueba pasó de 0.0252 a 0.0246, y el PR-AUC de la tasa histórica bajó de 0.320 a 0.315.
- **Por qué no lo detectó el versionado:** `data_version` no cambió porque solo cubre los datos crudos y las reglas de limpieza.
- **Qué se cambió:**
  - Cada run registra ahora `train_data_hash`, el hash del dataset de entrenamiento ya construido. Con eso un cambio en `features.py` queda visible.
  - El README ya tiene las cifras nuevas de P2.
  - **La propuesta (`docs/TC5061_Propuesta_Proyecto_Equipo34.docx`) todavía tiene las anteriores:** en el reporte conviene usar las nuevas y mencionar el cambio.

| Run de P2 | test_pr_auc | test_f1 | test_recall | valid_pr_auc |
|---|---|---|---|---|
| `p2-baseline-persistencia-semanal` | 0.388 | 0.350 | 0.350 | 0.397 |
| `p2-baseline-tasa-historica-estacion-bloque` | 0.315 | 0.335 | 0.434 | 0.376 |
| `p2-logistica-flujos-lag` | 0.405 | 0.414 | 0.445 | 0.405 |
| `p2-hgb-classifier-flujos-lag` | 0.464 | 0.445 | 0.421 | 0.483 |

## Cómo consolidar la evidencia (importante para todos)

Cada quien tiene su propio `mlflow.db` local, que es **de pruebas**. Para que la evidencia sea consistente, la corrida oficial es **una sola**:

1. Se hace merge de todos los PRs de modelos a `main`.
2. Una persona clona `main` en una carpeta nueva, levanta un servidor vacío y corre `make all`.
3. De ese servidor salen las capturas y las tablas del reporte.

Así todos los runs tienen el mismo commit, `git_dirty=false` y la misma versión de datos. Por eso el registro no usa `run_id` (cambian en cada base), sino nombres o métricas.

## Cómo agregar un modelo (lo que hace cada integrante)

1. En `mibici/config.py`, agregar los hiperparámetros por defecto en `P1_MODEL_PARAMS` (o P2, P3).
2. En `mibici/modeling/pN.py`:
   - escribir la función `(parts, hp) -> ModelResult` siguiendo las existentes, y devolver `estimator` y `features` para que se guarde el modelo;
   - agregarla a la tabla `PN_MODELS` con `ModelSpec(funcion, "<familia>", "candidate", PN_MODEL_PARAMS["<nombre>"])`.
3. Probar con `make train ENV_MANAGER=local Q=pN M=<nombre>` y, si aplica, con `make search ENV_MANAGER=local Q=pN M=<nombre> G="k=a|b;k2=c|d"`.
4. Agregar las líneas `train` o `search` en la sección de la pregunta en `scripts/run_experiments.sh`.
5. Correr `make lint` y `make test`, y abrir el PR.

Las etiquetas, los artefactos, la firma del modelo y el registro funcionan sin más cambios. Las métricas de selección son P1 `valid_wape` (menor es mejor), P2 `valid_pr_auc` y P3 `train_profile_silhouette` (mayor es mejor). Si el equipo decide otra métrica, se cambia en `SELECTION_METRIC`.

## Qué falta

Queda para que el equipo decida cómo se reparte:

- **Experimentos de P1:** hoy hay 3 baselines y 2 modelos (Poisson y HGB). La entrega pide **al menos 3 versiones adicionales al baseline**. Ideas: otro algoritmo (por ejemplo, Random Forest o LightGBM si se agrega a `requirements.txt`), variantes de features o una búsqueda sobre el HGB.
- **Experimentos de P2:** hay 2 baselines y 2 modelos (logística y HGB). Faltan versiones adicionales y una búsqueda (por ejemplo, sobre `learning_rate` y `max_leaf_nodes` del HGB, o sobre `C` de la logística).
- **Experimentos de P3:** hay 2 baselines, KMeans por perfiles y una búsqueda de ejemplo sobre `n_clusters` y `n_init`. Ideas: Gaussian Mixture, Agglomerative, PCA antes de KMeans.
- **Elegir y justificar el champion de cada pregunta:** desempeño contra el baseline, métrica de negocio, estabilidad y costo. Si el mejor por métrica no es el que se quiere, se fija con `register pN "<run_name>"` en el script. Opcional: un `challenger` por pregunta.
- **Corrida oficial** desde un clon limpio de `main` una vez hecho el merge de todo, y las **capturas**: la comparación con nested runs, los artefactos de una candidata y los tres modelos registrados con su alias y linaje. Las capturas van en `reports/figures/mlflow/`.
- **Reporte PDF** `TC5061_Reporte_MLFlow_Equipo34.pdf`, con:
  - el enlace al repositorio y los comandos de `mlflow run`;
  - la tabla comparativa por pregunta con validación, prueba y mejora contra el baseline;
  - el análisis y la justificación del champion de cada pregunta;
  - el esquema de aliases, las capturas y las referencias APA 7.
- **PRs y revisiones:** cada integrante abre al menos un PR y revisa al menos uno. Propuesta de cadena de revisión: Oscar → Diego → Javier → Mariana → Oscar.
- **Actualizar la tabla de resultados del README** con los modelos nuevos y los champions.

## Verificación realizada

- `make lint` y `make test` pasan (24 pruebas; las nuevas cubren hiperparámetros, mallas y defaults).
- Las métricas de los 12 modelos existentes son idénticas antes y después del refactor. P2 cambia solo por el PR #12, y se comprobó que el código original de `main` da lo mismo con los datos actuales.
- Clon limpio de la rama, servidor vacío y `make all` con `uv`:
  - 23 runs, todos con el mismo commit y `git_dirty=false`;
  - `data_version` `efa3a16661ec`;
  - los tres modelos registrados con `champion` y cargables por alias.
- `make predict Q=p1|p2|p3` genera predicciones de test con el champion de cada pregunta.
