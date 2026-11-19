# Resultados de las pruebas de carga

Generado el 2026-11-19 11:28 con `make load-report`. Contenedor de la API con límite de 2 vCPU y 2048 MiB (docker-compose.yml); escenario de tráfico en `load/locustfile.py`.

## Resumen por escenario

| Escenario | Workers | Peticiones | Errores (%) | RPS | p50 (ms) | p95 (ms) | p99 (ms) | CPU API prom. (%) | CPU API máx. (%) | Memoria API máx. (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 1 | 1868 | 0.00 | 15.6 | 25 | 54 | 80 | 30 | 48 | 239 |
| nominal | 1 | 10804 | 0.00 | 60.2 | 35 | 150 | 250 | 100 | 137 | 242 |
| estres | 1 | 18580 | 0.00 | 61.8 | 280 | 1500 | 1700 | 119 | 169 | 256 |
| nominal_2workers | 2 | 10757 | 0.00 | 60.0 | 23 | 53 | 90 | 90 | 114 | 444 |
| estres_2workers | 2 | 42080 | 0.00 | 93.3 | 240 | 1300 | 1600 | 155 | 208 | 515 |

CPU en porcentaje de un vCPU (100 % = un núcleo completo). En los escenarios de estrés el promedio mezcla todos los escalones; la tabla por escalón es la que importa.

## Estrés por escalones: 1 worker(s)

| Usuarios | RPS | p50 (ms) | p95 (ms) | Errores (%) | CPU API (%) | Memoria API (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 13.0 | 27 | 130 | 0.00 | 33 | 230 |
| 40 | 27.8 | 27 | 90 | 0.00 | 56 | 230 |
| 60 | 43.3 | 27 | 92 | 0.00 | 75 | 239 |
| 80 | 58.1 | 29 | 130 | 0.00 | 104 | 230 |
| 100 | 71.0 | 33 | 220 | 0.00 | 133 | 231 |
| 120 | 78.5 | 41 | 370 | 0.00 | 153 | 233 |
| 140 | 78.1 | 54 | 690 | 0.00 | 159 | 246 |
| 160 | 78.7 | 85 | 870 | 0.00 | 158 | 234 |
| 180 | 74.6 | 140 | 1300 | 0.00 | 159 | 256 |
| 200 | 78.6 | 230 | 1500 | 0.00 | 160 | 256 |

## Estrés por escalones: 2 worker(s)

| Usuarios | RPS | p50 (ms) | p95 (ms) | Errores (%) | CPU API (%) | Memoria API (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 14.4 | 26 | 90 | 0.00 | 28 | 447 |
| 40 | 28.2 | 22 | 55 | 0.00 | 43 | 438 |
| 60 | 43.9 | 23 | 55 | 0.00 | 81 | 438 |
| 80 | 58.8 | 24 | 61 | 0.00 | 103 | 441 |
| 100 | 74.4 | 26 | 72 | 0.00 | 129 | 442 |
| 120 | 89.7 | 27 | 85 | 0.00 | 154 | 454 |
| 140 | 102.9 | 29 | 140 | 0.00 | 180 | 453 |
| 160 | 111.6 | 32 | 210 | 0.00 | 193 | 472 |
| 180 | 116.7 | 37 | 430 | 0.00 | 203 | 485 |
| 200 | 121.9 | 43 | 600 | 0.00 | 201 | 482 |
| 220 | 120.3 | 53 | 770 | 0.00 | 203 | 505 |
| 240 | 113.1 | 67 | 960 | 0.00 | 204 | 497 |
| 260 | 125.7 | 92 | 1100 | 0.00 | 203 | 507 |
| 280 | 126.2 | 140 | 1200 | 0.00 | 204 | 502 |
| 300 | 127.2 | 200 | 1300 | 0.00 | 203 | 515 |

## Capacidad (insumo para FinOps)

Capacidad: el escalón con más RPS cuyo p95 <= 300 ms.

| Workers | Capacidad (peticiones/s) | Usuarios | p95 (ms) | Peticiones/hora | Peticiones/s por vCPU | CPU API (%) | Memoria API (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | **71** | 100 | 220 | 255,662 | 36 | 133 | 231 |
| 2 | **112** | 160 | 210 | 401,723 | 56 | 193 | 472 |

Con 2 workers la capacidad cambia +57 % respecto a 1 worker, con el mismo límite de 2 vCPU.

## Figuras

- `reports/figures/carga/carga_series.png`: RPS, latencia y errores por escenario.
- `reports/figures/carga/carga_saturacion.png`: curva de saturación del estrés.
- `reports/figures/carga/carga_recursos.png`: CPU y memoria de la API.
- `reports/figures/carga/carga_workers.png`: estrés con 1 y 2 workers.
