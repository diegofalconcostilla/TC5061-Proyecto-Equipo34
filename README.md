# TC5061-Proyecto-Equipo34 · Proyecto MiBici

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Propuesta de solución de Machine Learning para la operación de **MiBici**, el sistema de bicicleta pública del Área Metropolitana de Guadalajara. El repositorio tiene la ingesta reproducible y versionada de los viajes, el módulo de feature engineering y el experimento base registrado en MLflow Tracking. Es la base para las siguientes entregas del proyecto integrador de TC5061.

## Preguntas del proyecto

| # | Pregunta de negocio | Tarea | Formulación del equipo | Usuario |
|---|---|---|---|---|
| P1 | ¿Cuántos viajes se iniciarán en una estación en un periodo futuro? | Regresión | Viajes iniciados por **estación y día**, horizonte **1 día** | Operador del sistema (planeación de flota) |
| P2 | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes? | Clasificación | Riesgo en el **siguiente bloque de 3 h**: \|llegadas - salidas\| >= 0.5 x capacidad | Equipo de rebalanceo |
| P3 | ¿Qué grupos de estaciones comparten patrones de uso y dónde están? | Clustering | KMeans sobre el **perfil horario de uso** de cada estación (periodo de entrenamiento) | Planeación urbana |

MiBici no publica el inventario por estación. Para P2 el riesgo se aproxima con el flujo neto del bloque: si en 3 horas salen (o llegan) al menos media capacidad de bicicletas más de las que llegan (o salen), una estación con ocupación típica cercana al 50 % se vaciaría (o llenaría) si nadie rebalancea. Detalle en [`references/ml_canvas.md`](references/ml_canvas.md).

## Datos

| Concepto | Valor |
|---|---|
| Fuente | [Portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/): `datos_abiertos_AAAA_MM.csv` y `nomenclatura_2026_09.csv` |
| Rango | 2024-01 a 2026-08 (32 meses) |
| Viajes | 12,049,962 crudos · 11,677,716 limpios |
| Estaciones | 484 en el catálogo · 473 con viajes en el rango |
| Versión de datos | `efa3a16661ec` (ver [`data/manifest.json`](data/manifest.json)) |
| Partición temporal | train 2024-01 a 2025-12 · validación 2026-01 a 2026-04 · prueba 2026-05 a 2026-08 |

Condiciones de uso: el portal publica los datos como datos abiertos con la invitación "¡Explora, experimenta y comparte los datos de uso de Mi Bici!", pero no declara una licencia explícita (el sitio indica "Todos los derechos reservados Mi Bici © 2026" y el enlace al "Manual para uso de datos abiertos" no apunta a un manual). Por eso el proyecto usa los datos con fines académicos, cita la fuente, no redistribuye los archivos (cada quien los descarga del portal con la ingesta) y aplica el aviso de privacidad de la Agencia Metropolitana de Servicios de Infraestructura para la Movilidad, regido por la Ley General de Protección de Datos Personales en Posesión de Sujetos Obligados.

Datos personales: la ingesta elimina `Usuario_Id`, `Genero` y `Año_de_nacimiento` antes de guardar cualquier archivo; ninguno de ellos llega a los datasets de modelado ni a MLflow. Los datos de MiBici no se versionan en Git: solo el manifiesto con hashes y conteos.

## Estructura del repositorio

```
├── Makefile               <- make data | features | train | mlflow-ui | test
├── README.md
├── requirements.txt       <- dependencias con versión fija
├── pyproject.toml         <- metadatos del paquete mibici y configuración de ruff
├── data
│   ├── manifest.json      <- versionado de datos (único archivo de data/ en Git)
│   ├── raw/               <- CSV originales del portal (no versionado)
│   ├── interim/           <- Parquet limpio por mes y estaciones (no versionado)
│   └── processed/         <- datasets de P1, P2 y P3 (no versionado)
├── models/                <- modelos serializados (no versionado)
├── notebooks
│   └── 1.0-dfc-exploracion-mibici.ipynb   <- exploración (solo análisis)
├── references
│   ├── ml_canvas.md / .png               <- ML Canvas
│   └── arquitectura.md / .png            <- diagrama de arquitectura
├── reports/figures
│   └── mlflow/            <- capturas de la interfaz de MLflow
├── tests/                 <- pruebas de ingesta y de ausencia de fuga temporal
└── mibici                 <- módulo del proyecto
    ├── config.py          <- rutas, fechas, cortes, semilla y parámetros en un solo lugar
    ├── dataset.py         <- ingesta reproducible y verificación (CLI)
    ├── features.py        <- feature engineering de P1, P2 y P3 (CLI)
    ├── plots.py           <- gráficas que se registran como artefactos
    └── modeling
        └── train.py       <- experimento base en MLflow (CLI)
```

## Instalación

Requiere Python 3.12 y Git. El paquete declara `requires-python = "~=3.12.0"`, así que el entorno debe crearse con Python 3.12 explícitamente: si `python` apunta a otra versión (por ejemplo 3.13 o 3.14), la instalación falla.

```bash
git clone https://github.com/diegofalconcostilla/TC5061-Proyecto-Equipo34.git
cd TC5061-Proyecto-Equipo34
python3.12 -m venv .venv                # Windows: py -3.12 -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
python --version                        # debe mostrar Python 3.12.x
pip install -r requirements.txt         # instala también el paquete mibici en modo editable
```

Si `pip install` termina con `ERROR: Package 'mibici' requires a different Python: X.Y.Z not in '~=3.12.0'`, el entorno se creó con otra versión de Python: borra `.venv` y vuelve a crearlo con Python 3.12.

## Reproducir el experimento

```bash
python -m mibici.dataset ingest      # 1. descarga, limpia, guarda Parquet y escribe data/manifest.json (~5 min)
python -m mibici.dataset verify      #    opcional: compara los archivos locales con el manifiesto
python -m mibici.features            # 2. construye los datasets de P1, P2 y P3 (~30 s)
python -m mibici.modeling.train      # 3. registra 6 runs en MLflow (~1 min)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # 4. http://127.0.0.1:5000
```

Con `make` disponible: `make all` y después `make mlflow-ui`. Para correr un solo run: `python -m mibici.modeling.train --question p2 --model candidate`.

Si la ingesta produce la misma `data_version` (`efa3a16661ec`), los datos son idénticos a los del experimento reportado.

### Experimento base

Experimento `mibici-equipo34`: por pregunta, baselines sencillos, un modelo lineal (P1 y P2) y el candidato, para medir cuánto aporta cada nivel de complejidad. Métricas en prueba (mayo a agosto de 2026):

| Pregunta | Run | Resultado |
|---|---|---|
| P1 | `p1-baseline-historico-estacion-dia-semana` | test_mae 8.11 · test_wape 0.298 · test_r2 0.711 |
| P1 | `p1-baseline-media-movil-7d` | test_mae 7.14 · test_wape 0.262 · test_r2 0.838 |
| P1 | `p1-baseline-naive-estacional-7d` | test_mae 7.12 · test_wape 0.261 · test_r2 0.794 |
| P1 | `p1-poisson-lineal-lags-calendario` | test_mae 6.00 · test_wape 0.220 · test_r2 0.862 |
| P1 | `p1-hgb-regressor-lags-calendario` | test_mae 5.30 · test_wape 0.194 · test_r2 0.893 |
| P2 | `p2-baseline-tasa-historica-estacion-bloque` | test_pr_auc 0.320 · test_f1 0.346 · test_recall 0.400 |
| P2 | `p2-baseline-persistencia-semanal` | test_pr_auc 0.387 · test_f1 0.350 · test_recall 0.351 |
| P2 | `p2-logistica-flujos-lag` | test_pr_auc 0.405 · test_f1 0.415 · test_recall 0.450 |
| P2 | `p2-hgb-classifier-flujos-lag` | test_pr_auc 0.464 · test_f1 0.449 · test_recall 0.448 |
| P3 | `p3-baseline-kmeans-geografico` | train_profile_silhouette -0.014 (k = 6) |
| P3 | `p3-baseline-regla-manana-tarde` | train_profile_silhouette 0.051 (3 grupos) · test_stability_ari 0.271 |
| P3 | `p3-kmeans-perfiles-uso` | train_profile_silhouette 0.142 (k = 3) · test_stability_ari 0.446 |

El baseline histórico de P1 es el mejor de los tres baselines en validación (MAE 7.42 frente a 7.91 y 8.27) y el peor en prueba: el promedio de 2024-2025 representa cada vez peor la demanda de 2026 (deriva). El ARI del baseline geográfico de P3 es 1.0 por construcción (las coordenadas no cambian entre periodos), por eso no se reporta como estabilidad.

**Nota sobre P2:** los clasificadores se entrenan con `class_weight="balanced"`, así que su `predict_proba` no es una probabilidad calibrada: en prueba, la probabilidad media predicha por el candidato es 0.17, frente a una prevalencia real de riesgo de 0.025. El score sirve para ordenar estaciones por riesgo (PR-AUC, y F1 con el umbral elegido en validación), no para comunicar "X % de riesgo" (van den Goorbergh et al., 2022). Si se necesitan probabilidades, por ejemplo para decidir con costos, hay que calibrar con el periodo de validación o entrenar sin pesos.

## Buenas prácticas aplicadas

| Práctica | Cómo se aplica |
|---|---|
| Un experimento por proyecto y runs con nombre descriptivo | Todos los runs van a `mibici-equipo34`; cada uno tiene nombre `pN-<modelo>-<variables>` y una descripción. |
| Registrar lo necesario para reproducir | Parámetros `date_start`, `date_end`, `data_version`, `valid_start`, `test_start`, `seed`, granularidad, horizonte, modelo e hiperparámetros (`hp_*`). |
| Ligar el run al código y a la pregunta | Etiquetas `team`, `question`, `model_role`, `git_commit`, `git_branch` y `git_dirty`. |
| Métricas con el conjunto en el nombre | `valid_mae`, `test_mae`, `test_pr_auc`, `train_profile_silhouette`, `test_stability_ari`, etc. |
| Sin credenciales ni datos personales | No hay credenciales en el código; las columnas personales se eliminan en la ingesta; los artefactos solo contienen agregados e identificadores públicos de estaciones. |
| Notebooks solo para exploración | `notebooks/1.0-dfc-exploracion-mibici.ipynb` sigue la convención de la plantilla; la lógica vive en `mibici/`. |
| Configuración en un solo lugar | Rutas, fechas, cortes, semilla, umbrales y nombre del experimento en `mibici/config.py`. |
| Dependencias fijas | `requirements.txt` con `==` en todas las dependencias. |
| Nada de datos, modelos ni MLflow en Git | `.gitignore` excluye `data/*` (salvo el manifiesto), `*.csv`, `*.parquet`, `mlruns/`, `mlartifacts/`, `mlflow.db`, `models/*`, modelos serializados y `.env`. |
| Cambios por Pull Request revisado | `main` protegida: no admite push directo, ni siquiera de administradores. |
| Sin fuga de información del futuro | Lags con `shift` >= horizonte (validado en `add_lag_features` y en `tests/`), partición temporal y perfiles de P3 calculados solo con train. |

## Flujo de trabajo con Git

Seguimos **GitHub Flow**:

1. `main` está protegida: todo cambio entra por Pull Request, con conversaciones resueltas y sin force push.
2. Cada cambio se hace en una rama corta creada desde `main`.
3. Se abre un Pull Request con descripción y forma de probar, otro integrante lo revisa y después se hace merge. La rama se borra al hacer merge.

**Nombres de ramas:** `<tipo>/<descripcion-corta>`, por ejemplo `feat/ingesta-reproducible`, `fix/encabezados-latin1`, `docs/readme-propuesta`.

**Commits:** [Conventional Commits](https://www.conventionalcommits.org/es/v1.0.0/): `<tipo>(<ámbito opcional>): <descripción en imperativo>`. Tipos: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `style`. Ejemplo: `feat(dataset): ingesta reproducible con manifiesto de versionado`.

**Revisiones:** cada integrante abre al menos un Pull Request y revisa al menos uno de otro integrante.

## Integrantes y roles

| Integrante | Rol |
|---|---|
| Diego Falcón Costilla | Ingesta y versionado de datos, experimento base en MLflow |
| _Integrante 2_ | _Por definir_ |
| _Integrante 3_ | _Por definir_ |
| Javier Alejandro García | Feature Engineering |

## Referencias

- Dorard, L. (2018, 9 de marzo). *From data to AI with the Machine Learning Canvas (Part I)*. Own Machine Learning. Medium.
- Huyen, C. (2022). *Designing machine learning systems: An iterative process for production-ready applications*. O'Reilly Media.
- MiBici. (s.f.). *Datos abiertos* [Conjunto de datos]. Sistema de Bicicleta Pública del Área Metropolitana de Guadalajara. https://www.mibici.net/es/datos-abiertos/
- MLflow. (s.f.). *MLflow Tracking* [Documentación]. https://mlflow.org/docs/latest/ml/tracking/
- Treveil, M., Omont, N., Stenac, C., Lefevre, K., Phan, D., Zentici, J., Lavoillotte, A., Miyazaki, M., & Heidmann, L. (2020). *Introducing MLOps: How to scale machine learning in the enterprise*. O'Reilly Media.
- van den Goorbergh, R., van Smeden, M., Timmerman, D., & Van Calster, B. (2022). The harm of class imbalance corrections for risk prediction models: Illustration and simulation using logistic regression. *Journal of the American Medical Informatics Association, 29*(9), 1525-1534. https://doi.org/10.1093/jamia/ocac093
- Wilson, B. (2022). *Machine learning engineering in action*. Manning Publications.
