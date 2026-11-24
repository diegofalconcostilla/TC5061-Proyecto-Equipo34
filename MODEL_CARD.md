# Model Card · Proyecto MiBici (Equipo 34)

Ficha de los modelos que el sistema sirve en producción, siguiendo la estructura de *Model Cards for Model Reporting* (Mitchell et al., 2019). Cubre los tres modelos registrados en el Model Registry de MLflow; P1 es el modelo principal del proyecto y el único con monitoreo de drift, reentrenamiento y A/B testing, por eso tiene la sección más detallada.

| | P1 · Demanda | P2 · Riesgo de desbalance | P3 · Grupos de estaciones |
|---|---|---|---|
| Modelo registrado | `mibici-equipo34-p1-demanda` | `mibici-equipo34-p2-riesgo` | `mibici-equipo34-p3-grupos` |
| Tarea | Regresión | Clasificación binaria | Clustering |
| `@champion` | v1 · HistGradientBoostingRegressor (`learning_rate=0.1`) | v1 · HistGradientBoostingClassifier (`learning_rate=0.1`) + umbral | v1 · KMeans (k = 3) |
| `@challenger` | v3 · el mismo HGB reentrenado hasta 2026-06 | v2 · RandomForestClassifier | v2 · Birch |
| Endpoint | `POST /predict/p1` | `POST /predict/p2` | `POST /predict/p3` |
| Métrica de selección | `valid_wape` (menor es mejor) | `valid_pr_auc` | silueta + estabilidad (ARI) |

## 1. Detalles del modelo

- **Desarrollado por:** Equipo 34 de TC5061 (Diego Falcón Costilla, Oscar Elton, Mariana Favarony Avila y Javier Alejandro Garcia Flores), noviembre de 2026.
- **Tipo:** modelos tabulares de scikit-learn 1.5.2 guardados con MLflow 3.17.2 (`mlflow.sklearn`), con firma de entrada y salida e `input_example`. P2 guarda el umbral de decisión dentro del modelo (`ThresholdClassifier`).
- **Código y entrenamiento:** MLflow Project (`MLproject`): `ingest`, `train`, `search`, `register`, `retrain` y `abtest`. Configuración y semilla (`SEED = 42`) en `mibici/config.py`.
- **Versionado y estado de producción:** aliases del Model Registry, no stages (MLflow los marcó como obsoletos). `@champion` es lo que sirve la API por defecto; `@challenger` es la alternativa en evaluación y se consulta con `?alias=challenger`. La imagen de la API no contiene modelos: los descarga del registry al primer uso, así que promover una versión es mover el alias y reiniciar el contenedor.
- **Licencia:** la del repositorio ([LICENSE](LICENSE)). Los datos de MiBici se usan con fines académicos (ver [Datos](README.md#datos)).
- **Contacto:** issues del repositorio `diegofalconcostilla/TC5061-Proyecto-Equipo34`.

### Linaje

Cada versión registrada lleva estas etiquetas, que permiten ir del modelo en producción al código y a los datos exactos con que se entrenó:

| Etiqueta | Qué liga |
|---|---|
| `source_run_id`, `source_run_name` | run de MLflow que produjo el modelo (parámetros, métricas, gráficas y `config/run_config.json`) |
| `git_commit`, `git_dirty` | commit del código; la corrida oficial tiene `git_dirty=false` |
| `data_version` | hash de los datos crudos (`efa3a16661ec`, ver `data/manifest.json`) |
| `train_data_hash` | hash del dataset de entrenamiento ya construido (detecta cambios en `features.py`) |
| `selection_metric` y su valor | por qué se eligió esa versión |
| `search_parent_run_id` | run padre si la versión salió de una búsqueda con nested runs |

```
Portal MiBici ─► ingest (data_version) ─► features.py (train_data_hash) ─► run de MLflow (git_commit)
      ─► register (alias @champion/@challenger) ─► API FastAPI (responde con "version" y "source": "registry")
```

La respuesta de la API incluye `model`, `alias` y `version`, así que cada predicción servida se puede rastrear hasta su run.

## 2. Uso previsto

| | Usuario | Decisión que apoya | Frecuencia |
|---|---|---|---|
| P1 | Operador del sistema | Planeación de flota: cuántas bicicletas tener disponibles por estación mañana | Diaria (tablero de ~480 estaciones) |
| P2 | Equipo de rebalanceo | Qué estaciones visitar antes del siguiente bloque de 3 h | Cada bloque (06:00-24:00), en línea |
| P3 | Planeación urbana | Segmentar estaciones por patrón de uso para ubicar nuevas estaciones o campañas | Trimestral o menos |

**Fuera de alcance:**

- Decisiones sobre personas: los modelos no ven usuarios y no deben usarse para perfilar, tarifar o sancionar a nadie.
- P2 como probabilidad: el score no está calibrado (los clasificadores se entrenan con `class_weight="balanced"`; probabilidad media predicha 0.17 contra prevalencia real 0.025). Sirve para **ordenar** estaciones por riesgo y para la decisión binaria con el umbral de validación, no para decir "X % de riesgo".
- P2 como inventario real: MiBici no publica ocupación por estación. La etiqueta aproxima el riesgo con el flujo neto del bloque (|llegadas − salidas| ≥ 0.5 × capacidad).
- Estaciones sin historial: P1 y P2 necesitan lags de hasta 14 días (P1) o 4 semanas (P2). Una estación recién abierta no tiene predicción confiable hasta acumular ese historial.
- Otros sistemas de bicicletas u otras ciudades sin reentrenar.
- Horizontes distintos a los formulados (P1: 1 día; P2: siguiente bloque de 3 h).

## 3. Datos

| Concepto | Valor |
|---|---|
| Fuente | Portal de datos abiertos de MiBici, viajes mensuales 2024-01 a 2026-08 y catálogo de estaciones |
| Volumen | 12,049,962 viajes crudos · 11,677,716 limpios · 473 estaciones con viajes |
| Limpieza | viajes de menos de 1 min (anclajes fallidos) o más de 180 min se descartan; duplicados fuera |
| Partición temporal | entrenamiento 2024-01 a 2025-12 · validación 2026-01 a 2026-04 · prueba 2026-05 a 2026-08 |
| Unidad | P1: estación-día · P2: estación-bloque de 3 h · P3: estación (perfil horario de entrenamiento) |
| Calidad | validada con DeepChecks (`make validate`): 0 % de nulos, sin columnas constantes, sin categorías nuevas ([reporte](reports/validacion_datos.md)) |

**Privacidad.** La ingesta elimina `Usuario_Id`, `Genero` y `Año_de_nacimiento` antes de guardar cualquier archivo (`PERSONAL_DATA_COLUMNS` en `config.py`, comprobado por `test_clean_trips_drops_personal_data_and_invalid_rows`). Ningún dataset, artefacto de MLflow ni petición a la API contiene datos personales: solo agregados por estación e identificadores públicos de estaciones. Los datos crudos no se versionan en Git; solo el manifiesto con hashes.

**Variables de entrada.** Construidas por `mibici/features.py`, con lags de al menos el horizonte (`shift` ≥ horizonte, validado en código y pruebas) para evitar fuga de información del futuro.

- P1 (16): salidas rezagadas 1, 2, 7 y 14 días, medias móviles de 7 y 28 días, llegadas rezagadas y su media de 7 días, día de la semana, mes, fin de semana, festivo, capacidad, latitud y longitud.
- P2 (17): bloque del día, calendario y capacidad de la estación; flujo neto y salidas del bloque anterior, de hace un día y de hace una semana; riesgo rezagado un día y una semana; y flujo neto absoluto medio del mismo bloque en las últimas 4 semanas.
- P3 (52): 48 fracciones horarias (día hábil y fin de semana), volumen, balance, proporción de fin de semana y duración típica.

## 4. Métricas y desempeño

Corrida oficial del experimento `mibici-equipo34` (64 runs, mismo commit, `git_dirty=false`). La selección se hizo solo con validación; prueba es mayo a agosto de 2026, que ningún modelo vio.

### P1 · Demanda (champion v1)

| Modelo | Validación WAPE | Prueba WAPE | Prueba MAE |
|---|---:|---:|---:|
| Mejor baseline (naive estacional 7 d) | 0.258 | 0.261 | 7.12 |
| Poisson lineal (mejor lineal) | 0.197 | 0.219 | 5.97 |
| Random forest | 0.180 | 0.198 | 5.40 |
| **HGB `learning_rate=0.1` (champion v1)** | **0.176** | **0.196** | **5.33** |

El champion reduce el WAPE de prueba 25 % frente al mejor baseline: en promedio se equivoca en ~5 viajes por estación-día.

### P2 · Riesgo (champion v1)

| Modelo | Validación PR-AUC | Prueba PR-AUC | Prueba F1 |
|---|---:|---:|---:|
| Baseline persistencia semanal | 0.397 | 0.388 | 0.350 |
| Logística | 0.405 | 0.405 | 0.414 |
| Random forest (challenger) | 0.452 | 0.443 | 0.431 |
| **HGB `learning_rate=0.1` (champion)** | **0.483** | **0.462** | **0.442** (recall 0.454) |

Prevalencia de riesgo en prueba: 2.46 %. Un PR-AUC de 0.46 es ~19 veces la prevalencia, pero el modelo detecta menos de la mitad de los eventos de riesgo: complementa, no sustituye, la supervisión del equipo de rebalanceo.

### P3 · Grupos (champion v1)

| Modelo | Silueta (train) | Estabilidad ARI (train vs. 2026) |
|---|---:|---:|
| Regla mañana/tarde (baseline) | 0.051 | 0.271 |
| Mezcla gaussiana diagonal | 0.160 | 0.305 |
| Birch (challenger) | 0.122 | 0.532 |
| **KMeans k = 3 (champion)** | **0.142** | **0.446** |

La mezcla gaussiana tiene mejor silueta pero su asignación cambia mucho entre periodos; se eligió KMeans k = 3 por ser más estable y fácil de interpretar (3 grupos de perfil horario), y Birch, el más estable, queda como challenger.

### Desempeño por segmento (P1, A/B julio-agosto 2026)

| Segmento | Estaciones | Viajes/día | WAPE v1 | WAPE v3 |
|---|---:|---:|---:|---:|
| Estaciones existentes | 373 | 30.4 | 0.174 | 0.172 |
| **Estaciones nuevas** (abiertas 2025-2026) | 98 | 9.2 | **0.287** | **0.293** |
| Día laborable | 471 | 29.6 | 0.167 | 0.165 |
| Fin de semana o festivo | 471 | 20.8 | 0.225 | 0.225 |

**El error relativo es 65 % mayor en estaciones nuevas que en las existentes, y 35 % mayor en fines de semana que en días laborables.** Las estaciones nuevas, de menor volumen, tienden a estar en zonas de expansión de la red; si P1 se usa para asignar flota, esas zonas reciben pronósticos menos precisos. Se recomienda revisar manualmente su asignación hasta que acumulen historial (ver [Consideraciones éticas](#7-consideraciones-éticas-y-riesgos)).

## 5. Monitoreo en producción

Reportes de [Evidently](reports/analisis_monitoreo_p1/analisis_drift_ab_testing_p1.md) (`python -m mibici.monitoring`) sobre el periodo que representa producción (mayo-agosto 2026, 50,561 estación-días), con distancia de Wasserstein (numéricas) y Jensen-Shannon (categóricas), umbral 0.1:

| Referencia | Columnas con drift |
|---|---:|
| Entrenamiento completo (2024-2025) | 12 de 18 (66.7 %) |
| Misma temporada del año anterior (mayo-agosto 2025) | 11 de 18 (61.1 %) |

Comparar contra la misma temporada descarta que el drift sea estacional. **Causa principal: expansión de la red**, de 371 a 471 estaciones activas; las nuevas promedian 9.5 salidas al día contra 29.2 de las existentes, lo que desplaza hacia abajo las distribuciones de lags y medias móviles. Calidad del periodo actual: 0 % de faltantes, sin columnas vacías ni constantes.

**Disparadores de acción (propuestos por el equipo; hoy el monitoreo se corre a mano con `python -m mibici.monitoring`):**

| Señal | Umbral | Acción |
|---|---|---|
| Columnas con drift contra la misma temporada | > 30 % | `mlflow run . -e retrain` y registrar como `@challenger` |
| WAPE de P1 en ventana móvil | > 0.22 (mejor baseline de validación × 0.85) | Igual que arriba |
| A/B del challenger | IC 95 % de ΔWAPE no incluye 0 y favorece a B | `python -m mibici.modeling.abtest --promote` y reiniciar la API |
| Latencia p95 de la API | > 300 ms sostenido | Escalar workers (`API_WORKERS`) o réplicas |

## 6. Ciclo de reentrenamiento y A/B testing (P1)

1. **Detección:** el drift de mayo-agosto 2026 supera el umbral.
2. **Reentrenamiento:** `mlflow run . -e retrain` entrena la misma familia (HGB, `learning_rate=0.1`) con datos hasta 2026-06-30 y se registra como **v3 `@challenger`** (etiqueta `stage_of_work=retrain`).
3. **A/B fuera de muestra** (`mlflow run . -e abtest`): julio-agosto 2026, 27,433 estación-días que ninguna versión vio.

| Métrica | A: champion v1 | B: challenger v3 | B − A | IC 95 % (bootstrap por estación, 2,000) |
|---|---:|---:|---:|---|
| WAPE | 0.1802 | 0.1782 | −0.0020 | [−0.0036, −0.0004] · P(B mejor) = 0.998 |
| MAE | 4.874 | 4.821 | −0.053 | [−0.100, −0.011] |
| RMSE | 8.665 | 8.384 | −0.281 | |
| R² | 0.9214 | 0.9265 | +0.0050 | |

4. **Decisión:** promover v3. La promoción no es automática a propósito: `make abtest` registra la decisión (run `abtest` y etiquetas en ambas versiones) y una persona la ejecuta con `--promote`, que mueve `@champion` a v3 y `@challenger` a v1. Si v3 se degrada, revertir es mover el alias de vuelta.

**Estado de producción a la fecha de esta ficha:** `@champion` = v1 y `@challenger` = v3, con la promoción aprobada por el A/B y pendiente de ejecutarse. P2 y P3 no tienen reentrenamiento: sus champions son los de la corrida oficial.

## 7. Consideraciones éticas y riesgos

| Riesgo | Mitigación |
|---|---|
| **Inequidad territorial:** menor precisión en estaciones nuevas y de bajo volumen (zonas de expansión) puede llevar a asignarles menos flota y reforzar su bajo uso (ciclo de retroalimentación, Sculley et al., 2015). | Reportar el desempeño por segmento en cada A/B; revisión manual de la asignación en estaciones con menos de 8 semanas de historial; incluir el segmento "nueva" como criterio de aceptación de un challenger. |
| **Ciclo de retroalimentación en P2:** el rebalanceo que el modelo dispara cambia los flujos que después se usan para entrenar. | La etiqueta usa flujos de viajes, no inventario; documentar las intervenciones de rebalanceo cuando estén disponibles para separarlas del comportamiento de usuarios. |
| **Mal uso del score de P2 como probabilidad.** | La API solo expone la clase (0/1) con el umbral de validación; la nota está en esta ficha y en el README. |
| **Privacidad.** | Columnas personales eliminadas en la ingesta, con prueba automatizada; ningún artefacto contiene datos de usuarios. |
| **Datos sin licencia explícita.** | Uso académico, cita de la fuente, los datos no se redistribuyen (cada quien los descarga con la ingesta). |
| **Eventos atípicos** (Mundial 2026 en Guadalajara, cierres viales, lluvias): la demanda puede salirse del patrón aprendido. | Monitoreo de drift y de error; el operador puede ignorar la predicción en días con eventos conocidos. |

## 8. Limitaciones y advertencias

- Los modelos reciben variables ya construidas: la API no calcula features a partir de viajes en tiempo real.
- P1 y P3 se basan en el catálogo de estaciones de 2026-09; estaciones nuevas posteriores necesitan reentrenar P3 y acumular historial para P1.
- La capacidad medida (71 RPS con 1 worker, 112 con 2, p95 ≤ 300 ms) viene de Docker Desktop con Locust en la misma máquina: sirve para comparar configuraciones, no como capacidad de producción ([resultados](reports/carga/resultados.md)).
- El costo por inferencia (USD 0.0518 por millón de peticiones con 2 workers en t3.small) es una estimación con tarifas On-Demand de AWS us-east-1 ([reporte FinOps](reports/TC5061_Operacion_FinOps_Equipo34.pdf)).

## 9. Cómo reproducir

```bash
make mlflow-server   # terminal 1
make all             # ingesta + experimentos + registro de champions
mlflow run . -e retrain && mlflow run . -e register -P question=p1 -P run_name=p1-hgb-reentrenado-hasta-2026-06 -P alias=challenger
mlflow run . -e abtest
docker compose up -d --build --wait && curl -X POST localhost:8000/predict/p1 -H "Content-Type: application/json" -d @examples/api/p1.json
```

## Referencias

- Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 220-229. https://doi.org/10.1145/3287560.3287596
- Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems, 28*.
- van den Goorbergh, R., van Smeden, M., Timmerman, D., & Van Calster, B. (2022). The harm of class imbalance corrections for risk prediction models. *Journal of the American Medical Informatics Association, 29*(9), 1525-1534.
