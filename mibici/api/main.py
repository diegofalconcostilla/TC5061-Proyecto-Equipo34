from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
from pydantic import BaseModel, Field

from mibici.config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAMES

app = FastAPI(
    title="MiBici API",
    description="API REST para inferencia de los modelos P1, P2 y P3 del Equipo 34.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1)


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
        }
        for question, model_name in REGISTERED_MODEL_NAMES.items()
    }


@lru_cache(maxsize=6)
def load_model(question: str, alias: str):
    if question not in REGISTERED_MODEL_NAMES:
        raise ValueError(f"Pregunta no válida: {question}. Usa p1, p2 o p3.")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_name = REGISTERED_MODEL_NAMES[question]
    model_uri = f"models:/{model_name}@{alias}"

    return mlflow.pyfunc.load_model(model_uri)


@app.post("/predict/{question}")
def predict(question: str, request: PredictionRequest, alias: str = "champion"):
    if question not in REGISTERED_MODEL_NAMES:
        raise HTTPException(
            status_code=404,
            detail="Modelo no disponible. Usa p1, p2 o p3.",
        )

    try:
        model = load_model(question, alias)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"No fue posible cargar el modelo desde MLflow: {exc}",
        ) from exc

    frame = pd.DataFrame(request.records)

    input_schema = model.metadata.get_input_schema()
    features = input_schema.input_names()

    # Validar que estén presentes todas las columnas requeridas.
    missing = [column for column in features if column not in frame.columns]

    if missing:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Faltan variables requeridas por el modelo.",
                "missing_columns": missing,
            },
        )

    # Convertir los tipos de datos según la firma registrada en MLflow.
    dtype_map = {
        "integer": "int32",
        "long": "int64",
        "float": "float32",
        "double": "float64",
        "boolean": "bool",
        "string": "string",
    }

    try:
        for column in input_schema.inputs:
            dtype_name = getattr(column.type, "name", str(column.type)).lower()

            if dtype_name in dtype_map:
                frame[column.name] = frame[column.name].astype(dtype_map[dtype_name])

        # Generar predicciones utilizando las columnas en el orden esperado.
        predictions = model.predict(frame[features])
        values = predictions.tolist()

        results = [
            None if pd.isna(value) else value.item() if hasattr(value, "item") else value
            for value in values
        ]

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Error al generar predicciones: {exc}",
        ) from exc

    return {
        "question": question,
        "model": REGISTERED_MODEL_NAMES[question],
        "alias": alias,
        "total_predictions": len(results),
        "predictions": results,
    }
