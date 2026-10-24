# Machine Learning Canvas - Proyecto MiBici (Equipo 34)

Basado en la plantilla de Dorard (2018). Versión gráfica: [`ml_canvas.png`](ml_canvas.png).

| Bloque | Contenido |
|---|---|
| **Propuesta de valor** | Usuarios: operador del sistema (planeación de flota), equipo de rebalanceo y planeación urbana. Problema: estaciones vacías o llenas generan viajes perdidos; el rebalanceo es reactivo y la expansión se decide sin datos de uso. Valor: anticipar la demanda diaria, priorizar estaciones con riesgo de desabasto y entender qué zonas comparten patrones para crecer el sistema. |
| **Decisiones** | P1: cuántas bicicletas asignar por estación o zona para el día siguiente. P2: qué estaciones visitar primero en la siguiente ronda de rebalanceo (3 h). P3: dónde abrir estaciones nuevas y qué perfil de servicio dar a cada grupo. |
| **Tarea de ML** | P1 regresión: viajes iniciados por estación y día, horizonte 1 día. P2 clasificación binaria: estación en riesgo en el siguiente bloque de 3 h, con riesgo = \|llegadas - salidas\| >= 0.5 x capacidad. P3 clustering: grupos de estaciones por perfil horario de uso. |
| **Fuentes de datos** | Viajes mensuales `datos_abiertos_AAAA_MM.csv` y nomenclatura de estaciones del portal de datos abiertos de MiBici; calendario de festivos de México (derivado). Futuro: clima y eventos. |
| **Recolección de datos** | `python -m mibici.dataset ingest`: 32 meses (2024-01 a 2026-08), 11,677,716 viajes limpios, 484 estaciones. Manifiesto con SHA-256 y `data_version` versionado en Git. Las etiquetas se derivan de los viajes. |
| **Variables** | Lags y medias móviles de salidas y llegadas; calendario (día, mes, fin de semana, festivo, bloque); capacidad, coordenadas y municipio; para P3, 48 proporciones horarias, volumen, balance de flujos y duración mediana. |
| **Construir modelos** | Partición temporal: train 2024-01 a 2025-12, validación 2026-01 a 2026-04, prueba 2026-05 a 2026-08. Baselines: naive estacional de 7 días (P1), persistencia semanal (P2), KMeans geográfico (P3). Candidatos: HistGradientBoosting (P1, P2) y KMeans sobre perfiles de uso (P3). Reentrenamiento mensual. |
| **Hacer predicciones** | P1 en lote diario al cierre del día; P2 en lote al inicio de cada bloque de 3 h; P3 trimestral o al abrir estaciones. Cómputo local de segundos por lote. |
| **Simulación de impacto** | Métricas en prueba: P1 MAE y WAPE; P2 PR-AUC, recall y precisión; P3 silueta y estabilidad temporal (ARI). En P2 un falso negativo deja una estación sin servicio y un falso positivo cuesta una visita del camión. Los scores de P2 sirven para priorizar estaciones, no como probabilidades: con `class_weight="balanced"` sobreestiman el riesgo (media de 0.17 frente a una prevalencia de 0.025 en prueba), así que una regla de decisión por costos requiere calibrarlos antes. Criterio de aceptación: superar al baseline en prueba. |
| **Monitoreo** | Comparar predicción contra viajes reales con cada nuevo archivo mensual; vigilar deriva (estaciones nuevas, cambio en la prevalencia del riesgo); registrar cada reentrenamiento en MLflow con su `data_version`. |
