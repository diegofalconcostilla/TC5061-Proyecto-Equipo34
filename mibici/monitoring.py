"""Reportes de data drift y calidad de datos de P1 con Evidently.

    python -m mibici.monitoring            # requiere el servidor de MLflow y P1 @champion

Compara el periodo actual contra dos referencias y escribe en ``reports/monitoreo/``:

- ``p1_drift_entrenamiento.html``: contra todo el entrenamiento (la comparación pedida).
- ``p1_drift_misma_temporada.html``: contra los mismos meses del año anterior.
- ``p1_drift_por_variable.csv``: ambas comparaciones por variable y su lectura.
- ``p1_drift_resumen.json``: periodos, filas y calidad de los datos.

El entrenamiento cubre los 12 meses y el periodo actual solo de mayo a agosto, así que una
variable puede cambiar contra el entrenamiento solo por la época del año. La segunda referencia
separa ese efecto: lo que cambia contra ambas referencias es drift que la estacionalidad no
explica.
"""

from __future__ import annotations

import json

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from loguru import logger
import mlflow
import pandas as pd
import typer

from mibici.config import (
    MONITORING_CURRENT,
    MONITORING_REFERENCE,
    MONITORING_REPORTS_DIR,
    MONITORING_SEASONAL_REFERENCE,
    P1_DATASET_PATH,
    REGISTERED_MODEL_NAMES,
)
from mibici.modeling.p1 import P1_FEATURES
from mibici.modeling.tracking import setup_mlflow

app = typer.Typer(help="Reportes de drift y calidad de datos de P1 con Evidently.")

TARGET = "departures"
PREDICTION = "prediction"
CATEGORICAL = ["dow", "month", "is_weekend", "is_holiday"]
NUMERICAL = [c for c in P1_FEATURES if c not in CATEGORICAL] + [TARGET, PREDICTION]
COLUMNS = NUMERICAL + CATEGORICAL
QUALITY_METRICS = (
    "RowCount",
    "DuplicatedRowCount",
    "DatasetMissingValueCount",
    "ConstantColumnsCount",
    "EmptyColumnsCount",
)


def in_period(df: pd.DataFrame, period: tuple[str, str]) -> pd.DataFrame:
    dates = pd.to_datetime(df["date"])
    return df[(dates >= pd.Timestamp(period[0])) & (dates <= pd.Timestamp(period[1]))]


def with_predictions(df: pd.DataFrame, model) -> pd.DataFrame:
    return df.assign(**{PREDICTION: model.predict(df[P1_FEATURES])})


def to_dataset(df: pd.DataFrame) -> Dataset:
    """Dataset de Evidently; sus etiquetas categóricas no aceptan enteros de numpy."""
    definition = DataDefinition(numerical_columns=NUMERICAL, categorical_columns=CATEGORICAL)
    frame = df[COLUMNS].astype({c: str for c in CATEGORICAL})
    return Dataset.from_pandas(frame, data_definition=definition)


def run_report(current: pd.DataFrame, reference: pd.DataFrame, html_name: str) -> dict:
    """Genera el HTML de drift y calidad y devuelve sus métricas como diccionario."""
    report = Report([DataDriftPreset(), DataSummaryPreset()])
    snapshot = report.run(current_data=to_dataset(current), reference_data=to_dataset(reference))
    snapshot.save_html(str(MONITORING_REPORTS_DIR / html_name))
    return snapshot.dict()


def drift_by_column(result: dict) -> pd.DataFrame:
    """Puntaje de drift por columna; hay drift si el puntaje alcanza el umbral de su método."""
    rows = [
        {
            "variable": m["config"]["column"],
            "metodo": m["config"]["method"],
            "umbral": m["config"]["threshold"],
            "puntaje": m["value"],
        }
        for m in result["metrics"]
        if m["metric_name"].startswith("ValueDrift(")
    ]
    table = pd.DataFrame(rows).set_index("variable")
    table["drift"] = table["puntaje"] >= table["umbral"]
    return table


def classify(against_training: bool, against_season: bool) -> str:
    if against_training and against_season:
        return "drift no explicado por la temporada"
    if against_training:
        return "compatible con estacionalidad"
    if against_season:
        return "cambio frente al año anterior"
    return "estable"


def variable_role(column: str) -> str:
    if column == TARGET:
        return "objetivo"
    return "prediccion" if column == PREDICTION else "variable de entrada"


def compare(training: pd.DataFrame, season: pd.DataFrame, frames: dict) -> pd.DataFrame:
    """Une las dos comparaciones y agrega las medias de las variables numéricas."""
    table = pd.DataFrame(
        {
            "rol": [variable_role(c) for c in training.index],
            "metodo": training["metodo"],
            "umbral": training["umbral"],
            "puntaje_vs_entrenamiento": training["puntaje"],
            "drift_vs_entrenamiento": training["drift"],
            "puntaje_vs_misma_temporada": season["puntaje"].reindex(training.index),
            "drift_vs_misma_temporada": season["drift"].reindex(training.index),
        },
        index=training.index,
    )
    table["lectura"] = [
        classify(a, b)
        for a, b in zip(table["drift_vs_entrenamiento"], table["drift_vs_misma_temporada"])
    ]
    for name, frame in frames.items():
        table[f"media_{name}"] = frame[NUMERICAL].mean().reindex(table.index)
    return table.sort_values("puntaje_vs_misma_temporada", ascending=False)


def quality(result: dict) -> dict:
    """Métricas generales de calidad de los datos del periodo actual."""
    found = {}
    for metric in result["metrics"]:
        name = metric["metric_name"].split("(")[0]
        if name in QUALITY_METRICS and metric["metric_name"].endswith("()"):
            found[name] = metric["value"]
    return found


def drifted_share(result: dict) -> dict:
    metric = next(m for m in result["metrics"] if m["metric_name"].startswith("DriftedColumns"))
    return metric["value"]


@app.command()
def main():
    """Genera los reportes contra el entrenamiento y contra la misma temporada."""
    setup_mlflow()
    MONITORING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    name = REGISTERED_MODEL_NAMES["p1"]
    uri = f"models:/{name}@champion"
    version = mlflow.MlflowClient().get_model_version_by_alias(name, "champion").version
    model = mlflow.pyfunc.load_model(uri)

    data = pd.read_parquet(P1_DATASET_PATH)
    periods = {
        "entrenamiento": MONITORING_REFERENCE,
        "misma_temporada": MONITORING_SEASONAL_REFERENCE,
        "actual": MONITORING_CURRENT,
    }
    frames = {name: with_predictions(in_period(data, p), model) for name, p in periods.items()}

    training = run_report(frames["actual"], frames["entrenamiento"], "p1_drift_entrenamiento.html")
    season = run_report(
        frames["actual"], frames["misma_temporada"], "p1_drift_misma_temporada.html"
    )
    by_column = compare(drift_by_column(training), drift_by_column(season), frames)
    by_column.to_csv(MONITORING_REPORTS_DIR / "p1_drift_por_variable.csv")

    summary = {
        "modelo": uri,
        "version": int(version),
        "periodos": {k: list(v) for k, v in periods.items()},
        "filas": {k: len(v) for k, v in frames.items()},
        "columnas_con_drift": {
            "vs_entrenamiento": drifted_share(training),
            "vs_misma_temporada": drifted_share(season),
        },
        "calidad_periodo_actual": quality(training),
    }
    path = MONITORING_REPORTS_DIR / "p1_drift_resumen.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"Reportes de drift en {MONITORING_REPORTS_DIR}")


if __name__ == "__main__":
    app()
