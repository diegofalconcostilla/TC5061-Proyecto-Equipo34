"""Conexión con MLflow: servidor, experimento, etiquetas y registro de artefactos."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile

import mlflow

from mibici.config import (
    DATE_END,
    DATE_START,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROJ_ROOT,
    SEED,
    TEAM,
    TEST_START,
    VALID_START,
)
from mibici.dataset import get_data_version


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
    """Apunta al servidor de tracking y al experimento único del proyecto."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME, tags={"team": TEAM, "project": "MiBici TC5061"}
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
