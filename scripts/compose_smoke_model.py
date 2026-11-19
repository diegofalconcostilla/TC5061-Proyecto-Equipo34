"""Registra un modelo de humo de P1 para probar el sistema de Compose en CI.

    docker compose exec -T mlflow python scripts/compose_smoke_model.py

Corre dentro del servicio ``mlflow`` (que tiene MLflow y scikit-learn) contra un registry
vacío: entrena un ``DummyRegressor`` con la firma de ``examples/api/p1.json``, lo registra como
``mibici-equipo34-p1-demanda`` y le pone el alias ``champion``. Así el workflow comprueba que la
API descarga un modelo del servicio ``mlflow`` por la red de Compose y responde 200, sin datos
de MiBici. Se niega a correr si el modelo ya existe, para no tocar un registry real.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
import pandas as pd
from sklearn.dummy import DummyRegressor

MODEL_NAME = "mibici-equipo34-p1-demanda"
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "api" / "p1.json"


def main() -> None:
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
    client = MlflowClient()
    try:
        client.get_registered_model(MODEL_NAME)
    except MlflowException:
        pass
    else:
        raise SystemExit(f"{MODEL_NAME} ya existe: este script es solo para un registry vacío.")

    records = pd.DataFrame(json.loads(EXAMPLE.read_text(encoding="utf-8"))["records"])
    target = records["departures_lag1"].to_numpy()
    model = DummyRegressor(strategy="mean").fit(records, target)

    mlflow.set_experiment("compose-smoke-test")
    with mlflow.start_run(run_name="p1-dummy-smoke"):
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=infer_signature(records, model.predict(records)),
            registered_model_name=MODEL_NAME,
        )
    version = info.registered_model_version
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    print(f"{MODEL_NAME} v{version} con alias champion ({info.model_uri})")


if __name__ == "__main__":
    main()
