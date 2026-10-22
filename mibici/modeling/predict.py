"""Predice el periodo de test con el modelo registrado bajo un alias (``champion`` por defecto).

    python -m mibici.modeling.predict --question p1
    python -m mibici.modeling.predict --question p2 --alias challenger

Las columnas de entrada salen de la firma del modelo; las predicciones se guardan en
``data/processed/predictions_<pregunta>_<alias>.csv`` (no se versiona).
"""

from __future__ import annotations

from loguru import logger
import mlflow
import pandas as pd
import typer

from mibici.config import (
    P1_DATASET_PATH,
    P2_DATASET_PATH,
    P3_TEST_DATASET_PATH,
    PROCESSED_DATA_DIR,
    REGISTERED_MODEL_NAMES,
)
from mibici.modeling.tracking import setup_mlflow

app = typer.Typer(help="Predicciones de test con el modelo registrado bajo un alias.")

# Columnas que identifican cada fila en el archivo de predicciones.
TEST_DATA = {
    "p1": (P1_DATASET_PATH, ["station_id", "date"]),
    "p2": (P2_DATASET_PATH, ["station_id", "block_ts"]),
    "p3": (P3_TEST_DATASET_PATH, ["station_id"]),
}


def load_test(question: str) -> pd.DataFrame:
    path, _ = TEST_DATA[question]
    df = pd.read_parquet(path)
    return df[df["split"] == "test"] if "split" in df else df


@app.command()
def main(
    question: str = typer.Option(..., help="p1, p2 o p3."),
    alias: str = typer.Option("champion", help="Alias de la versión registrada."),
):
    """Carga ``models:/<nombre>@<alias>`` y predice el periodo de test."""
    question = question.lower()
    if question not in REGISTERED_MODEL_NAMES:
        raise typer.BadParameter(f"question debe ser p1, p2 o p3, no {question!r}")
    setup_mlflow()
    uri = f"models:/{REGISTERED_MODEL_NAMES[question]}@{alias}"
    model = mlflow.pyfunc.load_model(uri)
    features = model.metadata.get_input_schema().input_names()
    test = load_test(question)
    _, keys = TEST_DATA[question]
    out = test[keys].assign(prediction=model.predict(test[features]))
    path = PROCESSED_DATA_DIR / f"predictions_{question}_{alias}.csv"
    out.to_csv(path, index=False)
    logger.success(
        f"{uri}: {len(out)} predicciones en {path.relative_to(PROCESSED_DATA_DIR.parents[1])}"
    )


if __name__ == "__main__":
    app()
