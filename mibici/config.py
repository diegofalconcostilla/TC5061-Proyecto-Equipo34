"""Configuración central del proyecto MiBici.

Todas las rutas, fechas, semillas y parámetros compartidos viven aquí. Ningún otro
módulo define rutas, fechas ni semillas por su cuenta.
"""

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
# MLflow
# ---------------------------------------------------------------------------
TEAM = "equipo34"
MLFLOW_EXPERIMENT_NAME = f"mibici-{TEAM}"
MLFLOW_TRACKING_URI = f"sqlite:///{(PROJ_ROOT / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACTS_DIR = PROJ_ROOT / "mlartifacts"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
