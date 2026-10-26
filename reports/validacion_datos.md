# Validación de datos con DeepChecks: hallazgos

Resultados de `make validate` sobre los datasets de la versión de datos `efa3a16661ec`. Cómo correrlo y qué revisa cada chequeo: sección [Validación de datos](../README.md#validación-de-datos) del README. Los reportes HTML completos se generan en `reports/deepchecks/` (no se versionan).

Train es 2024-01 a 2025-12 y test es 2026-05 a 2026-08. En P1 y P2 se revisa una muestra fija de 50,000 filas por conjunto; P3 tiene una fila por estación (373 en train y 468 en test), así que se revisa completo.

## Resumen

| Pregunta | Integridad | Train contra test |
|---|---|---|
| P1 | Pasa | Drift en `month` (esperado) |
| P2 | Pasa | Drift en `month` (esperado) |
| P3 | Pasa, con 6 columnas constantes conocidas | Drift en 4 horas de fin de semana |

`make validate` termina sin error: la integridad pasa en las tres preguntas y el drift se reporta como aviso.

## Integridad

| Chequeo | P1 | P2 | P3 |
|---|---|---|---|
| Nulos | 0 % | 0 % | 0 % |
| Tipos de nulo y de dato mezclados | ninguno | ninguno | ninguno |
| Filas duplicadas (train / test) | 0.21 % / 0.95 % | 0.07 % / 0.36 % | 0 % / 0 % |
| Columnas constantes | ninguna | ninguna | 6 en train, 1 en test (ver abajo) |
| Etiquetas en conflicto | no aplica (regresión) | 0 % | no aplica (sin etiqueta) |

**Duplicados en P1 y P2.** Los revisé: son filas de estaciones sin actividad, con etiqueta 0 y casi todos los lags en 0 (entre 86 % y 100 % de los duplicados según el conjunto), y con el mismo calendario. Son observaciones reales, no errores de la ingesta, y están muy por debajo del 5 %.

**Columnas constantes en P3.** En train, `dep_wd_h02`, `dep_wd_h03`, `dep_wd_h04`, `dep_we_h02`, `dep_we_h03` y `dep_we_h04` valen 0 en todas las estaciones: en 2024-2025 no hay ninguna salida entre las 2:00 y las 4:59. En test solo `dep_wd_h03` es constante; en 2026 sí hay algunas salidas a las 2:00 y a las 4:00.

No es un error de datos, así que estas columnas están declaradas como excepción en `KNOWN_CONSTANT_COLUMNS` (en `mibici/validation.py`). Si otra columna se vuelve constante, el chequeo falla. Hoy no afectan los grupos: el `StandardScaler` de P3 las deja en 0 para todas las estaciones, así que no cambian ninguna distancia. Aun así, **para quien lleve P3: conviene evaluar si se quitan de `p3_feature_columns`**, para que el perfil solo tenga columnas que aportan información.

## Train contra test

**P1 y P2: drift en `month`** (V de Cramer 0.70, el umbral es 0.2). Es por construcción: train tiene los 12 meses y test solo mayo a agosto. No es un problema de los datos; el modelo vio esos meses en 2024 y 2025. El resto de las variables tiene drift bajo: la más alta en P1 es `departures_roll28_mean` (KS 0.11) y en P2 `abs_net_flow_same_block_4w_mean` (KS 0.12).

**Drift de la etiqueta:** P1 0.08 (KS) y P2 0.07 (V de Cramer), los dos debajo del umbral de 0.15. La demanda y la prevalencia de riesgo de 2026 se parecen a las de entrenamiento.

**P3: drift en la proporción de salidas de fin de semana** a las 5:00 (KS 0.30), 16:00 (0.21), 17:00 (0.30) y 18:00 (0.26). Los perfiles de fin de semana de 2026 son distintos a los de 2024-2025. Es consistente con la estabilidad moderada de los grupos entre periodos (ARI 0.45 del champion) y conviene mencionarlo al interpretar los grupos.

**Tamaño de test:** P1 y P2 tienen 50,000 filas en la muestra. P3 tiene 468 estaciones en test contra 373 en train (las que alcanzan `P3_MIN_DEPARTURES` salidas en cada periodo).

## Capturas

| Integridad de P1 | Train contra test de P1 |
|---|---|
| ![Integridad de P1](figures/deepchecks/p1_integridad.png) | ![Drift de P1](figures/deepchecks/p1_train_contra_test.png) |
| **Integridad de P3** | **Train contra test de P3** |
| ![Integridad de P3](figures/deepchecks/p3_integridad.png) | ![Drift de P3](figures/deepchecks/p3_train_contra_test.png) |
