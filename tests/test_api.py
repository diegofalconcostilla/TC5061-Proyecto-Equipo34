import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mibici.api.main import app, load_model

client = TestClient(app)


class FakeInputSchema:
    def input_names(self):
        return ["feature_1", "feature_2"]

    @property
    def inputs(self):
        class Column:
            def __init__(self, name):
                self.name = name
                self.type = "double"

        return [Column("feature_1"), Column("feature_2")]


class FakeMetadata:
    def get_input_schema(self):
        return FakeInputSchema()


class FakeModel:
    metadata = FakeMetadata()

    def predict(self, frame):
        return np.array([1.5] * len(frame))


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
    response = client.get("/models")

    assert response.status_code == 200

    models = response.json()

    assert set(models) == {"p1", "p2", "p3"}
    assert models["p1"]["alias"] == "champion"
    assert models["p2"]["alias"] == "champion"
    assert models["p3"]["alias"] == "champion"


@pytest.mark.parametrize("question", ["p1", "p2", "p3"])
def test_prediction_returns_model_results(monkeypatch, question):
    monkeypatch.setattr(
        "mibici.api.main.load_model",
        lambda question, alias: FakeModel(),
    )

    response = client.post(
        f"/predict/{question}",
        json={
            "records": [
                {"feature_1": 10.0, "feature_2": 20.0},
                {"feature_1": 30.0, "feature_2": 40.0},
            ]
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["question"] == question
    assert result["alias"] == "champion"
    assert result["total_predictions"] == 2
    assert result["predictions"] == [1.5, 1.5]


def test_prediction_rejects_unknown_model():
    response = client.post(
        "/predict/p4",
        json={"records": [{"feature_1": 10.0}]},
    )

    assert response.status_code == 404


def test_prediction_rejects_missing_features(monkeypatch):
    monkeypatch.setattr(
        "mibici.api.main.load_model",
        lambda question, alias: FakeModel(),
    )

    response = client.post(
        "/predict/p1",
        json={"records": [{"feature_1": 10.0}]},
    )

    assert response.status_code == 422
    assert "missing_columns" in response.json()["detail"]


def test_prediction_returns_503_when_model_cannot_be_loaded(monkeypatch):
    def fail_loading(question, alias):
        raise RuntimeError("MLflow unavailable")

    monkeypatch.setattr(
        "mibici.api.main.load_model",
        fail_loading,
    )

    response = client.post(
        "/predict/p1",
        json={"records": [{"feature_1": 10.0}]},
    )

    assert response.status_code == 503
    assert "MLflow" in response.json()["detail"]


def test_prediction_rejects_empty_records():
    response = client.post(
        "/predict/p1",
        json={"records": []},
    )

    assert response.status_code == 422
