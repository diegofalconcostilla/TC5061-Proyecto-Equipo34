# A/B testing de P1

- **A** = `champion` v1 (p1-hgb-regressor-lags-calendario[learning_rate=0.1]).
- **B** = `challenger` v3 (p1-hgb-reentrenado-hasta-2026-06).
- Evaluación: 2026-07-01 a 2026-08-31, 27,433 estación-días de 471 estaciones. Ninguna versión entrenó con estas fechas.

## Métricas globales

| Métrica | A | B | B - A |
|---|---:|---:|---:|
| MAE | 4.8738 | 4.8207 | -0.0530 |
| RMSE | 8.6650 | 8.3843 | -0.2807 |
| WAPE | 0.1802 | 0.1782 | -0.0020 |
| R2 | 0.9214 | 0.9265 | +0.0050 |

## Diferencia B - A con bootstrap por estación (2000 remuestreos)

| Métrica | B - A | IC 95 % | P(B mejor) |
|---|---:|---|---:|
| WAPE | -0.0020 | [-0.0036, -0.0004] | 0.998 |
| MAE | -0.053 | [-0.100, -0.011] | 0.998 |

B tiene menor error en 49.9% de las estación-días y A en 50.1% (empate en 0.0%).

## Por segmento

| segmento | grupo | filas | estaciones | viajes_promedio | wape_A | wape_B | mae_A | mae_B | cambio_wape_pct |
|---|---|---|---|---|---|---|---|---|---|
| Estación | existente | 23126 | 373 | 30.37 | 0.1741 | 0.1717 | 5.29 | 5.22 | -1.4 % |
| Estación | nueva | 4307 | 98 | 9.22 | 0.2868 | 0.2925 | 2.64 | 2.70 | +2.0 % |
| Tipo de día | fin de semana o festivo | 7981 | 471 | 20.84 | 0.2254 | 0.2248 | 4.70 | 4.68 | -0.3 % |
| Tipo de día | laborable | 19452 | 471 | 29.60 | 0.1671 | 0.1647 | 4.95 | 4.88 | -1.4 % |
| Mes | agosto | 14179 | 471 | 26.40 | 0.1734 | 0.1723 | 4.58 | 4.55 | -0.6 % |
| Mes | julio | 13254 | 440 | 27.75 | 0.1870 | 0.1842 | 5.19 | 5.11 | -1.5 % |

## Decisión

**Promover B: su WAPE es menor y el intervalo del 95 % no incluye 0.**
