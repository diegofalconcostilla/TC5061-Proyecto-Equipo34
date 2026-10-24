"""Entrena un modelo de P1, P2 o P3 y lo registra en el experimento de MLflow.

Se lanza desde el MLflow Project (un run por invocación):

    mlflow run . -e train -P question=p1 -P model=hgb
    mlflow run . -e train -P question=p1 -P model=hgb -P params="learning_rate=0.1,max_iter=300"

o, durante el desarrollo, directamente:

    python -m mibici.modeling.train --question p2 --model logistica --params "C=0.5"

Cada pregunta tiene su tabla de modelos (``P1_MODELS`` en ``p1.py``, etc.) y los
hiperparámetros por defecto viven en ``config.py`` (``P1_MODEL_PARAMS``, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import MiniBatchKMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import typer

from mibici.modeling.p1 import P1_MODELS, run_p1
from mibici.modeling.p2 import P2_MODELS, run_p2
from mibici.modeling.p3 import P3_MODELS, run_p3

from mibici import plots
from mibici.config import (
    DATE_END,
    DATE_START,
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    P1_DATASET_PATH,
    P2_BLOCK_HOURS,
    P2_DATASET_PATH,
    P2_RISK_CAPACITY_FRACTION,
    P3_BASELINE_K,
    P3_DATASET_PATH,
    P3_K_RANGE,
    P3_TEST_DATASET_PATH,
    PROJ_ROOT,
    SEED,
    TEAM,
    TEST_START,
    VALID_START,
)
from mibici.dataset import get_data_version
from mibici.features import p3_feature_columns
from mibici.modeling.tracking import STAGES

app = typer.Typer(help="Experimento base de MiBici con MLflow Tracking.")

P1_FEATURES = [
    "departures_lag1",
    "departures_lag2",
    "departures_lag7",
    "departures_lag14",
    "departures_roll7_mean",
    "departures_roll28_mean",
    "arrivals_lag1",
    "arrivals_lag7",
    "arrivals_roll7_mean",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
    "capacity",
    "latitude",
    "longitude",
]
P2_FEATURES = [
    "block",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
    "capacity",
    "latitude",
    "longitude",
    "net_flow_lag1",
    "net_flow_lag6",
    "net_flow_lag42",
    "departures_lag1",
    "departures_lag6",
    "departures_lag42",
    "risk_lag6",
    "risk_lag42",
    "abs_net_flow_same_block_4w_mean",
]


# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------
def git_info() -> dict[str, str]:
    """Commit, rama y estado del árbol de trabajo de Git para ligar el run al código."""

    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=PROJ_ROOT, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": str(bool(status) and status != "unknown").lower(),
    }


def setup_mlflow() -> None:
    """Configura el tracking local (SQLite) y el experimento único del proyecto."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME,
            artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri(),
            tags={"team": TEAM, "project": "MiBici TC5061"},
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def common_params(granularity: str, horizon: str) -> dict:
    """Parámetros de reproducibilidad compartidos por todos los runs."""
    return {
        "date_start": DATE_START,
        "date_end": DATE_END,
        "data_version": get_data_version(),
        "valid_start": VALID_START,
        "test_start": TEST_START,
        "seed": SEED,
        "granularity": granularity,
        "horizon": horizon,
    }


def start_run(question: str, role: str, run_name: str, description: str):
    """Abre un run con nombre descriptivo y las etiquetas obligatorias."""
    tags = {
        "team": TEAM,
        "question": question,
        "model_role": role,
        "mlflow.note.content": description,
        **git_info(),
    }
    return mlflow.start_run(run_name=run_name, tags=tags)


def log_json(data: dict, name: str) -> None:
    """Registra un diccionario como artefacto JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(path))


def log_figure(builder, name: str, *args, **kwargs) -> None:
    """Genera una figura con ``builder`` en un directorio temporal y la registra."""
    with tempfile.TemporaryDirectory() as tmp:
        path = builder(*args, path=Path(tmp) / name, **kwargs)
        mlflow.log_artifact(str(path), artifact_path="figures")


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {s: df[df["split"] == s] for s in ("train", "valid", "test")}


# ---------------------------------------------------------------------------
# P1: regresión de viajes por estación y día
# ---------------------------------------------------------------------------
def regression_metrics(y_true, y_pred, prefix: str) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}_rmse": root_mean_squared_error(y_true, y_pred),
        f"{prefix}_wape": np.abs(y_true - y_pred).sum() / y_true.sum(),
        f"{prefix}_r2": r2_score(y_true, y_pred),
    }




# ---------------------------------------------------------------------------
# P2: clasificación del riesgo por estación y bloque de 3 horas
# ---------------------------------------------------------------------------
def classification_metrics(y_true, y_score, y_pred, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}_pr_auc": average_precision_score(y_true, y_score),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_score),
        f"{prefix}_positive_rate": float(np.mean(y_true)),
    }


def best_f1_threshold(y_true, y_score) -> float:
    """Umbral que maximiza F1 en validación (nunca se elige con test)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


# ---------------------------------------------------------------------------
# P3: clustering de estaciones por patrón de uso
# ---------------------------------------------------------------------------
def cluster_metrics(x, labels, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_silhouette": silhouette_score(x, labels, random_state=SEED),
        f"{prefix}_davies_bouldin": davies_bouldin_score(x, labels),
        f"{prefix}_calinski_harabasz": calinski_harabasz_score(x, labels),
    }

app = typer.Typer(help="Entrena un modelo de MiBici y lo registra en MLflow.")

RUNNERS = {"p1": run_p1, "p2": run_p2, "p3": run_p3}
MODELS = {"p1": P1_MODELS, "p2": P2_MODELS, "p3": P3_MODELS}


@app.command()
def main(
    question: str = typer.Option(..., help="p1, p2 o p3."),
    model: str = typer.Option(
        "baseline",
        help="; ".join(f"{q.upper()}: {', '.join(models)}" for q, models in MODELS.items()),
    ),
    params: str = typer.Option("", help='Cambios a los defaults de config.py: "k=v,k2=v2".'),
    stage: str = typer.Option(
        "", help=f"stage_of_work ({', '.join(STAGES)}); por defecto, el del modelo."
    ),
):
    """Entrena un modelo y registra parámetros, métricas y artefactos en un run."""
    question = question.lower()
    if question not in RUNNERS:
        raise typer.BadParameter(f"question debe ser p1, p2 o p3, no {question!r}")
    setup_mlflow()
    RUNNERS[question](model, params, stage)


if __name__ == "__main__":
    app()
