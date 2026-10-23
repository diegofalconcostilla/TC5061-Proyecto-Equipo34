"""Registra en MLflow la ingesta que acaba de correr: manifiesto, totales y versión de datos.

Es el último paso del punto de entrada ``ingest`` del MLproject; deja en el experimento la
evidencia de qué datos produjo cada ``data_version``.
"""

from __future__ import annotations

from loguru import logger
import mlflow

from mibici.config import DATA_MANIFEST_PATH, DATE_END, DATE_START, TEAM
from mibici.dataset import load_manifest
from mibici.modeling.tracking import git_info, setup_mlflow


def main() -> None:
    manifest = load_manifest()
    setup_mlflow()
    with mlflow.start_run():
        mlflow.set_tags(
            {
                "mlflow.runName": f"ingest-{manifest['data_version']}",
                "team": TEAM,
                "stage_of_work": "ingest",
                "data_version": manifest["data_version"],
                **git_info(),
            }
        )
        mlflow.log_params(
            {
                "date_start": DATE_START,
                "date_end": DATE_END,
                "data_version": manifest["data_version"],
                "pipeline_version": manifest["pipeline_version"],
            }
        )
        mlflow.log_metrics({k: float(v) for k, v in manifest["totals"].items()})
        mlflow.log_artifact(str(DATA_MANIFEST_PATH), artifact_path="data")
    logger.success(f"Ingesta registrada en MLflow: data_version={manifest['data_version']}")


if __name__ == "__main__":
    main()
