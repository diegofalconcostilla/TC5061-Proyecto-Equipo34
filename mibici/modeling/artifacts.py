"""Artefactos de cada run, organizados en subcarpetas.

- ``model``: el modelo con firma y ejemplo de entrada (``mlflow.sklearn.log_model``).
- ``data/``: manifiesto de la ingesta y huella del conjunto de entrenamiento (más ``log_input``).
- ``plots/``: gráficas de desempeño de la tarea.
- ``config/``: configuración efectiva de la corrida y copia de ``config.py``.
- ``metrics/``: métricas y salidas tabulares.

Solo se registran agregados por estación e identificadores públicos de estaciones; ningún dato
personal de los usuarios de MiBici.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile

import mlflow
from mlflow.models import infer_signature
import pandas as pd

from mibici.config import DATA_MANIFEST_PATH, PROJ_ROOT

CONFIG_FILE = PROJ_ROOT / "mibici" / "config.py"
MODEL_NAME = "model"


def log_json(data: dict, name: str, folder: str = "metrics") -> None:
    mlflow.log_dict(data, f"{folder}/{name}")


def log_figure(builder, name: str, *args, **kwargs) -> None:
    """Genera una figura con ``builder`` en un directorio temporal y la registra en plots/."""
    with tempfile.TemporaryDirectory() as tmp:
        path = builder(*args, path=Path(tmp) / name, **kwargs)
        mlflow.log_artifact(str(path), artifact_path="plots")


def log_table(df: pd.DataFrame, name: str, folder: str = "metrics") -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        df.to_csv(path, index=False)
        mlflow.log_artifact(str(path), artifact_path=folder)


def content_hash(df: pd.DataFrame) -> str:
    """Hash del contenido completo (el digest de MLflow solo usa las primeras filas)."""
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()[:16]


def log_data(train: pd.DataFrame, question: str, source: Path, target: str | None) -> None:
    """Versión de los datos: manifiesto de la ingesta, huella del dataset y ``log_input``.

    ``data_version`` identifica los CSV crudos y la limpieza; ``train_data_hash`` además cambia
    si cambia el feature engineering con los mismos datos crudos.
    """
    mlflow.log_artifact(str(DATA_MANIFEST_PATH), artifact_path="data")
    relpath = source.relative_to(PROJ_ROOT).as_posix()
    data_hash = content_hash(train)
    mlflow.log_param("train_data_hash", data_hash)
    log_json(
        {
            "question": question,
            "source": relpath,
            "rows": len(train),
            "columns": list(train.columns),
            "content_hash": data_hash,
        },
        "dataset.json",
        folder="data",
    )
    dataset = mlflow.data.from_pandas(
        train,
        source=relpath,
        name=f"{question}-train",
        targets=target,
    )
    mlflow.log_input(dataset, context="training")


def log_config(run_config: dict) -> None:
    log_json(run_config, "run_config.json", folder="config")
    mlflow.log_artifact(str(CONFIG_FILE), artifact_path="config")


def log_model(estimator, X_example: pd.DataFrame) -> None:
    """Modelo con firma inferida de un ejemplo de entrada y su predicción."""
    X_example = X_example.reset_index(drop=True)
    signature = infer_signature(X_example, estimator.predict(X_example))
    mlflow.sklearn.log_model(
        estimator,
        name=MODEL_NAME,
        signature=signature,
        input_example=X_example,
        serialization_format="cloudpickle",
        code_paths=[str(PROJ_ROOT / "mibici")],
    )
