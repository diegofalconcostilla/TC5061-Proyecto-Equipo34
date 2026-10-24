"""Conexión con MLflow: servidor, experimento, parámetros comunes y etiquetas de cada run."""

from __future__ import annotations

import subprocess

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


def common_params(question: str, role: str, granularity: str, horizon: str) -> dict:
    """Parámetros de reproducibilidad compartidos por todos los runs.

    ``question`` y ``model`` coinciden con los que ``mlflow run`` registra del MLproject, para
    que los runs lanzados con ``python -m`` tengan los mismos parámetros.
    """
    return {
        "question": question,
        "model": role,
        "date_start": DATE_START,
        "date_end": DATE_END,
        "data_version": get_data_version(),
        "valid_start": VALID_START,
        "test_start": TEST_START,
        "seed": SEED,
        "granularity": granularity,
        "horizon": horizon,
    }


STAGES = ("baseline", "search", "candidate")


def start_run(
    question: str,
    role: str,
    family: str,
    stage: str,
    run_name: str,
    description: str,
    nested: bool = False,
):
    """Abre un run con nombre descriptivo y las etiquetas para filtrar en la interfaz.

    Bajo ``mlflow run`` el run ya existe (``MLFLOW_RUN_ID``) y ``start_run`` lo retoma ignorando
    ``run_name`` y ``tags``; por eso se fijan después con ``set_tags``.
    """
    if stage not in STAGES:
        raise ValueError(f"stage_of_work debe ser uno de {STAGES}, no {stage!r}")
    tags = {
        "team": TEAM,
        "question": question,
        "model_role": role,
        "model_family": family,
        "stage_of_work": stage,
        "data_version": get_data_version(),
        "mlflow.runName": run_name,
        "mlflow.note.content": description,
        **git_info(),
    }
    run = mlflow.start_run(run_name=run_name, nested=nested)
    mlflow.set_tags(tags)
    return run
