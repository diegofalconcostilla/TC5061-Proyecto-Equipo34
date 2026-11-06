"""Carga de los modelos de la API desde el Model Registry de MLflow.

Hay dos fuentes, en este orden:

1. ``packaged``: copia local en ``API_MODELS_DIR/<pregunta>/<alias>/`` (en la imagen,
   ``/app/models/api``), descargada antes del ``podman build`` con::

       python -m mibici.api.registry --alias champion      # o: make api-models

   La copia guarda en ``registry.json`` el nombre, la versión y el run de origen.
2. ``registry``: ``models:/<nombre>/<versión>`` desde ``MLFLOW_TRACKING_URI``, resolviendo
   primero qué versión tiene el alias.

La primera hace que la imagen funcione igual en cualquier máquina, sin servidor de MLflow ni
volúmenes con la ruta absoluta de ``mlartifacts``; la segunda permite cambiar de champion sin
reconstruir la imagen.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import shutil
from typing import Any, NamedTuple

from loguru import logger
import mlflow
from mlflow import MlflowClient
import typer

from mibici.config import API_MODELS_DIR, MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAMES

METADATA_FILE = "registry.json"


class LoadedModel(NamedTuple):
    model: Any  # mlflow.pyfunc.PyFuncModel
    name: str
    alias: str
    version: str
    source: str  # "packaged" o "registry"


def packaged_dir(question: str, alias: str) -> Path:
    return API_MODELS_DIR / question / alias


@lru_cache(maxsize=12)
def load_model(question: str, alias: str) -> LoadedModel:
    """Modelo de ``question`` con ``alias``: copia empaquetada o, si no existe, el registry."""
    name = REGISTERED_MODEL_NAMES[question]
    local = packaged_dir(question, alias)
    if (local / "MLmodel").exists():
        meta = json.loads((local / METADATA_FILE).read_text(encoding="utf-8"))
        logger.info(f"{name}@{alias}: copia empaquetada v{meta['version']} ({local})")
        return LoadedModel(
            mlflow.pyfunc.load_model(str(local)), name, alias, meta["version"], "packaged"
        )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    version = MlflowClient().get_model_version_by_alias(name, alias).version
    logger.info(f"{name}@{alias}: v{version} desde {MLFLOW_TRACKING_URI}")
    model = mlflow.pyfunc.load_model(f"models:/{name}/{version}")
    return LoadedModel(model, name, alias, str(version), "registry")


def export_model(question: str, alias: str) -> Path:
    """Descarga ``models:/<nombre>@<alias>`` a ``API_MODELS_DIR`` junto con su linaje."""
    name = REGISTERED_MODEL_NAMES[question]
    mv = MlflowClient().get_model_version_by_alias(name, alias)
    dst = packaged_dir(question, alias)
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{name}/{mv.version}", dst_path=str(dst)
    )
    meta = {
        "question": question,
        "name": name,
        "alias": alias,
        "version": str(mv.version),
        "run_id": mv.run_id,
        "source_run_name": mv.tags.get("source_run_name", ""),
        "tracking_uri": MLFLOW_TRACKING_URI,
    }
    (dst / METADATA_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.success(f"{name} v{mv.version} @{alias} -> {dst}")
    return dst


app = typer.Typer(help="Descarga del Model Registry los modelos que sirve la API.")


@app.command()
def main(
    alias: str = typer.Option("champion", help="Alias de la versión registrada."),
    question: str = typer.Option("all", help="p1, p2, p3 o all."),
):
    """Copia los modelos a models/api/ para empaquetarlos en la imagen de la API."""
    question = question.lower()
    if question != "all" and question not in REGISTERED_MODEL_NAMES:
        raise typer.BadParameter(f"question debe ser p1, p2, p3 o all, no {question!r}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    for q in REGISTERED_MODEL_NAMES if question == "all" else [question]:
        export_model(q, alias)


if __name__ == "__main__":
    app()
