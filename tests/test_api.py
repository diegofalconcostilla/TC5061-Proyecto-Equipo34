"""Pruebas de la API con modelos simulados: no necesitan servidor de MLflow ni datos de MiBici.

Las peticiones válidas son los ejemplos de ``examples/api/`` (los mismos del README).
"""

import copy
import json
from pathlib import Path

from fastapi.testclient import TestClient
import mlflow
from mlflow.models import infer_signature
from mlflow.types import ColSpec, Schema
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mibici.api import registry
from mibici.api.main import app
from mibici.api.schemas import P1Record, P2Record, P3Record
from mibici.config import REGISTERED_MODEL_NAMES

client = TestClient(app)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "api"
RECORDS = {"p1": P1Record, "p2": P2Record, "p3": P3Record}
OUTPUTS = {"p1": 1.5, "p2": 1, "p3": 2}


def example(question: str) -> dict:
    return json.loads((EXAMPLES / f"{question}.json").read_text(encoding="utf-8"))


class FakeMetadata:
    def __init__(self, question):
        fields = RECORDS[question].model_fields
        self.schema = Schema(
            [
                ColSpec("long" if f.annotation is int else "double", name)
                for name, f in fields.items()
            ]
        )

    def get_input_schema(self):
        return self.schema


class FakeModel:
    def __init__(self, question):
        self.question = question
        self.metadata = FakeMetadata(question)
        self.seen = None

    def predict(self, frame):
        self.seen = frame
        return np.array([OUTPUTS[self.question]] * len(frame))


def fake_loader(question, alias):
    model = FakeModel(question)
    return registry.LoadedModel(model, REGISTERED_MODEL_NAMES[question], alias, "7", "registry")


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr("mibici.api.main.load_model", fake_loader)


def test_root_returns_service_information():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["service"] == "MiBici API"
    assert response.json()["status"] == "running"
    assert response.json()["available_models"] == ["p1", "p2", "p3"]


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_models_returns_registered_models():
    models = client.get("/models").json()

    assert set(models) == {"p1", "p2", "p3"}
    assert all(m["alias"] == "champion" for m in models.values())
    assert models["p1"]["name"] == REGISTERED_MODEL_NAMES["p1"]


@pytest.mark.parametrize("question", ["p1", "p2", "p3"])
def test_examples_match_the_api_schema(question):
    """Los ejemplos del README son peticiones válidas para el esquema de cada pregunta."""
    for record in example(question)["records"]:
        RECORDS[question].model_validate(record)


@pytest.mark.parametrize("question", ["p1", "p2", "p3"])
def test_prediction_returns_model_results(fake_models, question):
    payload = example(question)
    response = client.post(f"/predict/{question}", json=payload)

    assert response.status_code == 200
    result = response.json()
    n = len(payload["records"])
    assert result["question"] == question
    assert result["model"] == REGISTERED_MODEL_NAMES[question]
    assert result["alias"] == "champion"
    assert result["version"] == "7"
    assert result["total_predictions"] == n
    assert result["predictions"] == [OUTPUTS[question]] * n


def test_prediction_accepts_challenger_alias(fake_models):
    response = client.post("/predict/p1?alias=challenger", json=example("p1"))

    assert response.status_code == 200
    assert response.json()["alias"] == "challenger"


def test_prediction_rejects_unknown_model():
    response = client.post("/predict/p4", json=example("p1"))

    assert response.status_code == 404


def test_prediction_rejects_missing_features(fake_models):
    payload = example("p1")
    del payload["records"][0]["departures_lag7"]

    response = client.post("/predict/p1", json=payload)

    assert response.status_code == 422
    errors = response.json()["detail"]
    assert errors[0]["type"] == "missing"
    assert errors[0]["loc"][-1] == "departures_lag7"


def test_prediction_rejects_unknown_features(fake_models):
    payload = example("p1")
    payload["records"][0]["departures_lag_7"] = 3.0  # error de escritura

    response = client.post("/predict/p1", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


@pytest.mark.parametrize(
    ("question", "field", "value"),
    [
        ("p1", "dow", 7),
        ("p1", "month", 0),
        ("p1", "is_holiday", 2),
        ("p1", "departures_lag1", -1),
        ("p1", "dow", "lunes"),
        ("p2", "block", 6),
        ("p2", "risk_lag6", 1.5),
        ("p3", "weekend_share", 1.2),
    ],
)
def test_prediction_rejects_out_of_range_values(fake_models, question, field, value):
    payload = example(question)
    payload["records"][0][field] = value

    response = client.post(f"/predict/{question}", json=payload)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"][-1] == field


def test_p3_profile_must_sum_to_one(fake_models):
    payload = example("p3")
    payload["records"][0]["dep_wd_h08"] += 0.2

    response = client.post("/predict/p3", json=payload)

    assert response.status_code == 422
    assert "dep_wd_h*" in response.json()["detail"][0]["msg"]


def test_prediction_returns_503_when_model_cannot_be_loaded(monkeypatch):
    def fail_loading(question, alias):
        raise RuntimeError("MLflow unavailable")

    monkeypatch.setattr("mibici.api.main.load_model", fail_loading)

    response = client.post("/predict/p1", json=example("p1"))

    assert response.status_code == 503
    assert "MLflow" in response.json()["detail"]


def test_prediction_rejects_empty_records():
    response = client.post("/predict/p1", json={"records": []})

    assert response.status_code == 422


def test_prediction_uses_model_column_order_and_types(monkeypatch):
    model = FakeModel("p1")
    monkeypatch.setattr(
        "mibici.api.main.load_model",
        lambda q, a: registry.LoadedModel(model, "m", a, "1", "registry"),
    )
    payload = example("p1")
    payload["records"] = [dict(reversed(r.items())) for r in payload["records"]]

    assert client.post("/predict/p1", json=payload).status_code == 200
    assert list(model.seen.columns) == model.metadata.schema.input_names()
    assert model.seen["dow"].dtype == "int64"


def test_invalid_response_is_not_returned(monkeypatch):
    """La salida también se valida: P2 solo puede devolver 0 o 1."""

    class BrokenModel(FakeModel):
        def predict(self, frame):
            return np.array([0.73] * len(frame))

    monkeypatch.setattr(
        "mibici.api.main.load_model",
        lambda q, a: registry.LoadedModel(BrokenModel("p2"), "m", a, "1", "registry"),
    )
    no_raise = TestClient(app, raise_server_exceptions=False)

    assert no_raise.post("/predict/p2", json=example("p2")).status_code == 500


def test_packaged_model_is_loaded_without_mlflow_server(monkeypatch, tmp_path):
    """Un modelo guardado en API_MODELS_DIR se sirve sin consultar el Model Registry."""
    records = example("p1")["records"]
    X = pd.DataFrame(copy.deepcopy(records * 5))
    y = np.arange(len(X), dtype=float)
    estimator = LinearRegression().fit(X, y)
    dst = tmp_path / "p1" / "champion"
    mlflow.sklearn.save_model(
        estimator, str(dst), signature=infer_signature(X, estimator.predict(X))
    )
    (dst / registry.METADATA_FILE).write_text(json.dumps({"version": "3"}), encoding="utf-8")
    monkeypatch.setattr(registry, "API_MODELS_DIR", tmp_path)
    monkeypatch.setattr(registry, "MLFLOW_TRACKING_URI", "http://127.0.0.1:9")  # inaccesible
    registry.load_model.cache_clear()

    try:
        response = client.post("/predict/p1", json={"records": records})
    finally:
        registry.load_model.cache_clear()

    assert response.status_code == 200
    result = response.json()
    assert result["source"] == "packaged"
    assert result["version"] == "3"
    expected = estimator.predict(pd.DataFrame(records)[X.columns])
    np.testing.assert_allclose(result["predictions"], expected, rtol=1e-6)


def test_model_is_loaded_from_registry_by_alias(monkeypatch, tmp_path):
    """Sin copia local (como en el contenedor), el alias se resuelve en el registry."""
    loaded_uris = []

    class FakeClient:
        def get_model_version_by_alias(self, name, alias):
            assert (name, alias) == (REGISTERED_MODEL_NAMES["p1"], "champion")
            return type("ModelVersion", (), {"version": "5"})()

    def fake_load(uri):
        loaded_uris.append(uri)
        return FakeModel("p1")

    monkeypatch.setattr(registry, "API_MODELS_DIR", tmp_path)
    monkeypatch.setattr(registry, "MlflowClient", FakeClient)
    monkeypatch.setattr(registry.mlflow.pyfunc, "load_model", fake_load)
    registry.load_model.cache_clear()

    try:
        response = client.post("/predict/p1", json=example("p1"))
    finally:
        registry.load_model.cache_clear()

    assert response.status_code == 200
    assert loaded_uris == [f"models:/{REGISTERED_MODEL_NAMES['p1']}/5"]
    assert response.json()["source"] == "registry"
    assert response.json()["version"] == "5"
