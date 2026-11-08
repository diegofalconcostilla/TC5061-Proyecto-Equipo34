"""API REST de inferencia de los modelos P1, P2 y P3 de MiBici.

    uvicorn mibici.api.main:app --port 8000      # documentación interactiva en /docs

Cada pregunta tiene su endpoint y su esquema de Pydantic (``mibici/api/schemas.py``): la API
rechaza con 422 registros sin todas las variables, con variables de más o fuera de rango, y
valida también la respuesta (``response_model``).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
import pandas as pd
from pydantic import BaseModel

from mibici.api.registry import LoadedModel, load_model
from mibici.api.schemas import (
    P1Request,
    P1Response,
    P2Request,
    P2Response,
    P3Request,
    P3Response,
)
from mibici.config import REGISTERED_MODEL_DESCRIPTIONS, REGISTERED_MODEL_NAMES

app = FastAPI(
    title="MiBici API",
    description="API REST para inferencia de los modelos P1, P2 y P3 del Equipo 34.",
    version="1.2.0",
)

ALIAS = Query(
    "champion",
    description="Alias de la versión en el Model Registry (champion o challenger).",
)

# Tipos de la firma de MLflow -> dtype de pandas con el que se entrenó el modelo.
DTYPES = {
    "integer": "int32",
    "long": "int64",
    "float": "float32",
    "double": "float64",
    "boolean": "bool",
    "string": "string",
}


@app.get("/")
def root():
    return {
        "service": "MiBici API",
        "status": "running",
        "documentation": "/docs",
        "available_models": list(REGISTERED_MODEL_NAMES.keys()),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    return {
        question: {
            "name": model_name,
            "alias": "champion",
            "description": REGISTERED_MODEL_DESCRIPTIONS[question],
        }
        for question, model_name in REGISTERED_MODEL_NAMES.items()
    }


def get_model(question: str, alias: str) -> LoadedModel:
    try:
        return load_model(question, alias)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"No fue posible cargar el modelo desde MLflow: {exc}",
        ) from exc


def to_frame(loaded: LoadedModel, records: list[BaseModel]) -> pd.DataFrame:
    """Registros validados -> DataFrame con las columnas y tipos de la firma del modelo."""
    frame = pd.DataFrame([r.model_dump() for r in records])
    schema = loaded.model.metadata.get_input_schema()
    missing = [c for c in schema.input_names() if c not in frame.columns]
    if missing:
        # El esquema de la API ya exige todas sus variables: esto es un champion nuevo con otras.
        raise HTTPException(
            status_code=422,
            detail={
                "message": "El esquema de la API no coincide con la firma del modelo.",
                "missing_columns": missing,
            },
        )
    for column in schema.inputs:
        dtype = DTYPES.get(getattr(column.type, "name", str(column.type)).lower())
        if dtype:
            frame[column.name] = frame[column.name].astype(dtype)
    return frame[schema.input_names()]


def predict(question: str, records: list[BaseModel], alias: str) -> dict:
    loaded = get_model(question, alias)
    frame = to_frame(loaded, records)
    try:
        values = loaded.model.predict(frame)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Error al generar predicciones: {exc}"
        ) from exc
    predictions = [v.item() if hasattr(v, "item") else v for v in values.tolist()]
    return {
        "question": question,
        "model": loaded.name,
        "alias": loaded.alias,
        "version": loaded.version,
        "source": loaded.source,
        "total_predictions": len(predictions),
        "predictions": predictions,
    }


@app.post("/predict/p1", response_model=P1Response, summary="P1: viajes del día siguiente")
def predict_p1(request: P1Request, alias: str = ALIAS):
    return predict("p1", request.records, alias)


@app.post("/predict/p2", response_model=P2Response, summary="P2: riesgo de desbalance")
def predict_p2(request: P2Request, alias: str = ALIAS):
    return predict("p2", request.records, alias)


@app.post("/predict/p3", response_model=P3Response, summary="P3: grupo de uso de la estación")
def predict_p3(request: P3Request, alias: str = ALIAS):
    return predict("p3", request.records, alias)
