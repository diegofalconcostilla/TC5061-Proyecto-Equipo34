# Periodos del monitoreo de P1

Las fechas viven en `mibici/config.py` (sección de monitoreo). Los reportes se generan con
`python -m mibici.monitoring`, con el servidor de MLflow activo y P1 registrado como `@champion`.

| Conjunto | Periodo | Filas (estación-día) | Para qué sirve |
|---|---|---:|---|
| Referencia | 2024-01-01 a 2025-12-31 | 262,712 | Es el periodo con el que se entrenó el champion. |
| Referencia de la misma temporada | 2025-05-01 a 2025-08-31 | 45,580 | Son los mismos meses un año antes, para aislar la estacionalidad. |
| Actual | 2026-05-01 a 2026-08-31 | 50,561 | Representa producción: es posterior a todo el entrenamiento y a la selección del champion. |
| Ajuste de la versión alternativa | hasta 2026-06-30 | 330,600 | Incorpora la primera mitad del periodo actual. |
| Evaluación del A/B | 2026-07-01 a 2026-08-31 | 27,433 | Ninguna de las dos versiones la usó para entrenar. |

## Por qué estos periodos

- **Actual.** Mayo a agosto de 2026 ya está descargado y versionado (`data_version` `efa3a16661ec`).
  El champion se eligió con la validación de enero a abril de 2026, así que este periodo solo se
  había usado para reportar el desempeño final.
- **Dos referencias.** El entrenamiento tiene los 12 meses y el periodo actual solo cuatro, de modo
  que `month` cambia contra el entrenamiento aunque el comportamiento sea idéntico. Una variable
  se considera con drift no explicado por la temporada solo si cambia contra ambas referencias.
- **Evaluación separada.** Si la versión alternativa se evaluara con meses que usó para entrenar,
  su ventaja estaría inflada. Por eso el periodo actual se divide en dos.

## Cómo leer los reportes

- Evidently usa la distancia de Wasserstein normalizada en las variables numéricas y la distancia
  de Jensen-Shannon en las categóricas, con umbral de 0.1. Son sus valores por defecto para
  conjuntos de más de 1,000 filas.
- `DuplicatedRowCount` (486) no indica registros repetidos: el reporte no incluye `station_id` ni
  `date`, y en la tabla original no hay ninguna pareja estación-fecha duplicada.
- La red creció entre los dos veranos: pasó de 371 a 471 estaciones con filas. Las 100 estaciones
  nuevas promedian 9.5 salidas al día y las que ya existían, 29.2, frente a 31.9 del verano de 2025.
  Parte de la caída en los lags y promedios móviles se debe a esta composición, y parte a menos
  demanda en las estaciones existentes.
