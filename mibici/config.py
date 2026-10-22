"""Configuración central del proyecto MiBici.

Todas las rutas, fechas, semillas y parámetros compartidos viven aquí. Ningún otro
módulo define rutas, fechas ni semillas por su cuenta.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

REFERENCES_DIR = PROJ_ROOT / "references"

# Manifiesto de versionado de datos (es el único archivo de data/ que se versiona en Git).
DATA_MANIFEST_PATH = DATA_DIR / "manifest.json"

# Datos limpios en Parquet.
TRIPS_INTERIM_DIR = INTERIM_DATA_DIR / "trips"  # un Parquet por mes
STATIONS_INTERIM_PATH = INTERIM_DATA_DIR / "stations.parquet"

# ---------------------------------------------------------------------------
# Fuente de datos (portal de datos abiertos de MiBici)
# ---------------------------------------------------------------------------
MIBICI_BASE_URL = "https://www.mibici.net"
MIBICI_OPEN_DATA_URL = f"{MIBICI_BASE_URL}/es/datos-abiertos/"
STATIONS_FILE_NAME = "nomenclatura_2026_09.csv"
HTTP_TIMEOUT_SECONDS = 300

# ---------------------------------------------------------------------------
# Rango de fechas y partición temporal (sin fuga de información del futuro)
# ---------------------------------------------------------------------------
DATE_START = "2024-01"  # primer mes incluido (AAAA-MM)
DATE_END = "2026-08"  # último mes incluido (AAAA-MM)

# Fechas de corte: train < VALID_START <= validación < TEST_START <= test
VALID_START = "2026-01-01"
TEST_START = "2026-05-01"

# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Limpieza de viajes
# ---------------------------------------------------------------------------
MIN_TRIP_MINUTES = 1  # viajes más cortos se consideran anclajes fallidos
MAX_TRIP_MINUTES = 180  # viajes más largos se consideran anómalos

# Columnas con datos personales que se eliminan durante la ingesta.
PERSONAL_DATA_COLUMNS = ["usuario_id", "genero", "anio_nacimiento"]

# ---------------------------------------------------------------------------
# Formulación de las preguntas
# ---------------------------------------------------------------------------
# Datasets de modelado (Parquet en data/processed/).
P1_DATASET_PATH = PROCESSED_DATA_DIR / "p1_station_day.parquet"
P2_DATASET_PATH = PROCESSED_DATA_DIR / "p2_station_block.parquet"
P3_DATASET_PATH = PROCESSED_DATA_DIR / "p3_station_profiles_train.parquet"
P3_TEST_DATASET_PATH = PROCESSED_DATA_DIR / "p3_station_profiles_test.parquet"

# P1 (regresión): viajes iniciados por estación y día, horizonte de 1 día.
P1_LAGS_DAYS = [1, 2, 7, 14]
P1_ROLLING_DAYS = [7, 28]

# P2 (clasificación): bloques de 3 horas a partir de las 06:00, horizonte de 1 bloque.
P2_BLOCK_HOURS = 3
P2_FIRST_HOUR = 6
P2_BLOCKS_PER_DAY = (24 - P2_FIRST_HOUR) // P2_BLOCK_HOURS
# Riesgo: |llegadas - salidas| del bloque >= fracción de la capacidad de la estación.
P2_RISK_CAPACITY_FRACTION = 0.5
P2_ROLLING_WEEKS = 4

# P3 (clustering): perfiles de uso calculados solo con el periodo de entrenamiento.
P3_K_RANGE = list(range(3, 9))
P3_MIN_DEPARTURES = 30  # estaciones con menos salidas en train tienen perfiles ruidosos
P3_BASELINE_K = 6
# Regla de mañana y tarde: fracción de las salidas pico de días hábiles que ocurren en la mañana.
P3_RULE_MORNING_HOURS = range(6, 10)
P3_RULE_EVENING_HOURS = range(16, 20)
P3_RULE_THRESHOLDS = (0.4, 0.6)  # < 0.4 destino matutino, > 0.6 origen matutino, resto mixta

# ---------------------------------------------------------------------------
# Hiperparámetros por defecto de cada modelo (se sobrescriben con -P params="k=v,...")
# ---------------------------------------------------------------------------
# La clave es el nombre del modelo en P1_MODELS, P2_MODELS o P3_MODELS. Los modelos sin
# hiperparámetros (reglas y promedios) no aparecen aquí.
P1_MODEL_PARAMS = {
    "poisson": {"alpha": 1e-4, "max_iter": 1000},
    "hgb": {
        "loss": "poisson",
        "learning_rate": 0.05,
        "max_iter": 500,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 30,
    },
}
P2_MODEL_PARAMS = {
    "logistica": {"C": 1.0, "class_weight": "balanced", "max_iter": 1000},
    "hgb": {
        "learning_rate": 0.05,
        "max_iter": 400,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
        "class_weight": "balanced",
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 30,
    },
}
P3_MODEL_PARAMS = {
    "baseline": {"n_clusters": P3_BASELINE_K, "n_init": 20},
    # n_clusters = 0: se elige k en P3_K_RANGE por silueta.
    "kmeans": {"n_clusters": 0, "n_init": 20},
}

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
TEAM = "equipo34"
MLFLOW_EXPERIMENT_NAME = f"mibici-{TEAM}"
# Servidor local: mlflow server --backend-store-uri sqlite:///mlflow.db --port 5001
# (el puerto 5000 lo ocupa AirPlay en macOS). mlflow run exporta esta variable al proceso.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
