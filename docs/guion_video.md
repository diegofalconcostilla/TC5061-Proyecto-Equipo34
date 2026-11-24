# Guion del video · Proyecto final (Equipo 34)

Video **continuo, sin cortes**, de **4:00 a 4:30 min** (límite 2-5 min). Una persona maneja el teclado y las cuatro narran por turnos. Archivo: `TC5061_Proyecto_Final_Equipo34.mp4`.

## Antes de grabar (no sale en el video)

```bash
git pull origin main
docker compose up -d --build --wait     # MLflow (5001) + API (8000), ambos healthy
docker compose ps
# Si el registry está vacío: make all  (tarda ~10 min; hacerlo antes, no en el video)
curl -s localhost:8000/health           # {"status": "ok"}
curl -s -X POST localhost:8000/predict/p1 -H "Content-Type: application/json" -d @examples/api/p1.json   # calentar
```

Dejar abiertos en pestañas, en este orden:

1. `docs/TC5061_Proyecto_Final_Equipo34.pdf` en pantalla completa.
2. MLflow UI: <http://localhost:5001> (experimento `mibici-equipo34`, ordenado por `valid_wape`).
3. MLflow UI: Models → `mibici-equipo34-p1-demanda`.
4. Swagger: <http://localhost:8000/docs>.
5. Una terminal en la raíz del repo con letra grande.
6. GitHub: pestaña **Actions** con la última corrida verde de `main`.
7. `reports/analisis_monitoreo_p1/evidencia/evidently/p1_drift_misma_temporada.html`.

Grabar con OBS o la grabadora de Zoom/Teams a 1080p. Cerrar notificaciones.

## Guion

| Tiempo | Quién | Pantalla | Qué decir (idea, no leer palabra por palabra) |
|---|---|---|---|
| 0:00-0:20 | Diego | Diapositiva 1-2 | Somos el equipo 34. MiBici tiene 470 estaciones y 11.7 millones de viajes en datos abiertos; el reto es anticipar la demanda para mover bicicletas a tiempo. Tres preguntas: demanda del día siguiente, riesgo de que una estación quede vacía o llena en 3 horas, y grupos de estaciones. |
| 0:20-0:45 | Diego | Diapositiva 3 | La arquitectura: experimentación con MLflow, operación con contenedores, CI/CD y Locust, y monitoreo con Evidently, A/B testing y Model Card. Todo se reproduce con `make all` y `docker compose up`. |
| 0:45-1:15 | Oscar | Pestaña MLflow (runs) | Estructura Cookiecutter, features con lags sin fuga del futuro y el código como MLflow Project. Aquí los 64 runs de la corrida oficial: cada uno guarda `git_commit`, `data_version` y el hash del dataset. Ordenado por WAPE de validación, el mejor es el HistGradientBoosting: 25 % menos error que el mejor baseline. |
| 1:15-1:40 | Oscar | Pestaña Models | En el Model Registry usamos aliases: `@champion` es lo que sirve la API y `@challenger` lo que evaluamos. Cada versión tiene su linaje: run, commit y versión de datos. |
| 1:40-2:40 | Mariana | Terminal + Swagger | **Demo.** `docker compose ps`: MLflow y la API en healthy. (Ejecutar los tres `curl` de abajo.) P1 responde 54.6 y 49.3 viajes, ese día fueron 56 y 43, y la respuesta dice versión 1 y `source: registry`: la imagen no lleva modelos, los descarga del registry. Con `?alias=challenger` responde la versión 3. Una petición con `dow = 9` devuelve 422. En Swagger se ven los esquemas de las tres preguntas. |
| 2:40-3:05 | Mariana | Pestaña Actions | Cada Pull Request corre lint, pruebas, bandit y pip-audit, construye la imagen y levanta el sistema con Compose para probarlo de punta a punta; en `main` publica la imagen en GHCR. |
| 3:05-3:25 | Mariana | Diapositiva 9 | Con Locust: 71 peticiones por segundo con un worker y 112 con dos, sin errores; el cuello es la CPU. |
| 3:25-3:55 | Javier | Reporte de Evidently y diapositiva 10 | Evidently detectó drift en 11 de 18 variables incluso contra la misma temporada de 2025: la red creció de 371 a 471 estaciones. Reentrenamos con `-e retrain` y el A/B en julio-agosto da a la versión nueva menor WAPE con 99.8 % de probabilidad, así que recomienda promoverla; la promoción es un paso explícito. |
| 3:55-4:15 | Javier | Diapositivas 11-12 | FinOps: 5 centavos de dólar por millón de peticiones con dos workers en una t3.small. La Model Card documenta el uso previsto y un riesgo: las estaciones nuevas tienen 65 % más error, así que su asignación se revisa a mano. |
| 4:15-4:30 | Diego | Diapositiva 13-14 | Las decisiones clave: partición temporal, registry como fuente de verdad, promoción con A/B y capacidad medida antes de dimensionar. El repositorio está en GitHub. Gracias. |

### Comandos de la demo (copiar a la terminal antes de grabar)

```bash
docker compose ps
curl -s -X POST localhost:8000/predict/p1 -H "Content-Type: application/json" -d @examples/api/p1.json | python -m json.tool
curl -s -X POST "localhost:8000/predict/p2?alias=challenger" -H "Content-Type: application/json" -d @examples/api/p2.json | python -m json.tool
curl -s -X POST localhost:8000/predict/p1 -H "Content-Type: application/json" -d @examples/api/p1_invalido.json | python -m json.tool
```

Nota: P1 solo tiene `@challenger` v3 en el registry donde se corrió el reentrenamiento. Si el registry de la grabación viene de `make all` sin `retrain`, usar P2 para mostrar `?alias=challenger` (responde versión 2), como arriba.

## Checklist de entrega en Canvas

- [ ] Enlace al repositorio: <https://github.com/diegofalconcostilla/TC5061-Proyecto-Equipo34>
- [ ] `TC5061_Proyecto_Final_Equipo34.pdf` (está en `docs/`)
- [ ] `TC5061_Proyecto_Final_Equipo34.mp4` entre 2:00 y 5:00, sin cortes (o el enlace dentro de la presentación)
- [ ] Revisar la duración final antes de subir
