"""Reentrena P1 con datos del periodo actual, sin tocar el periodo reservado para el A/B.

    mlflow run . -e retrain --experiment-name mibici-equipo34
    mlflow run . -e register -P question=p1 -P run_name=p1-hgb-reentrenado-hasta-2026-06 \
        -P alias=challenger

El modelo y sus hiperparámetros son los del champion, para que la diferencia entre versiones
venga de los datos. Hay dos ajustes: uno hasta antes de ``RETRAIN_VALID_START``, que se valida
hasta ``RETRAIN_END`` y confirma que los hiperparámetros siguen sirviendo, y el definitivo hasta
``RETRAIN_END``, que es el que se registra. Las fechas desde ``AB_EVAL_START`` no se leen aquí.
"""

from __future__ import annotations

from loguru import logger
import mlflow
import pandas as pd
import typer

from mibici.config import AB_EVAL_START, P1_DATASET_PATH, RETRAIN_END, RETRAIN_VALID_START
from mibici.modeling.artifacts import log_config, log_data, log_json, log_model
from mibici.modeling.common import get_spec
from mibici.modeling.p1 import P1_MODELS, regression_metrics
from mibici.modeling.params import cli_text, resolve_params
from mibici.modeling.tracking import common_params, setup_mlflow, start_run

app = typer.Typer(help="Reentrena P1 con datos del periodo actual.")

TARGET = "departures"


def retrain_windows(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Ventanas de ajuste; ninguna contiene fechas posteriores a ``RETRAIN_END``."""
    dates = pd.to_datetime(df["date"])
    window = df[dates <= pd.Timestamp(RETRAIN_END)]
    window_dates = pd.to_datetime(window["date"])
    is_check = window_dates >= pd.Timestamp(RETRAIN_VALID_START)
    return {"full": window, "fit": window[~is_check], "check": window[is_check]}


def run_retrain(model: str, params: str) -> str:
    spec = get_spec(P1_MODELS, "p1", model)
    hp = resolve_params(spec.defaults, params)
    windows = retrain_windows(pd.read_parquet(P1_DATASET_PATH))
    full, check = windows["full"], windows["check"]

    # p1_hgb y compañía predicen valid y test: aquí ambos son la ventana de chequeo.
    checked = spec.fit({"train": windows["fit"], "valid": check, "test": check}, hp)
    final = spec.fit({"train": full, "valid": check, "test": check}, hp)
    metrics = regression_metrics(check[TARGET], checked.scores["valid"], "valid")

    run_name = f"p1-{model}-reentrenado-hasta-{RETRAIN_END[:7]}"
    description = (
        f"P1 reentrenado con datos hasta {RETRAIN_END} y los hiperparámetros del champion. "
        f"Las métricas valid_* vienen del ajuste de chequeo; desde {AB_EVAL_START} es el A/B."
    )
    run_params = common_params("p1", model, "estacion-dia", "1 dia")
    run_params |= {
        "target": TARGET,
        "n_train": len(full),
        "n_valid": len(check),
        "train_end": RETRAIN_END,
        "valid_start": RETRAIN_VALID_START,
        "test_start": AB_EVAL_START,
        **final.params,
    }
    with start_run("P1", model, spec.family, "retrain", run_name, description):
        mlflow.log_params(run_params)
        mlflow.log_metrics(metrics)
        log_data(full, "p1", P1_DATASET_PATH, target=TARGET)
        log_config({"question": "p1", "model": model, "hyperparameters": hp, "params": run_params})
        log_model(final.estimator, full[final.features].head(5))
        log_json(metrics, "metrics.json")
    logger.success(f"{run_name}: valid_wape={metrics['valid_wape']:.4f} (n_train={len(full)})")
    return run_name


@app.command()
def main(
    model: str = typer.Option("hgb", help=f"Uno de: {', '.join(P1_MODELS)}."),
    params: str = typer.Option("learning_rate=0.1", help='Hiperparámetros: "k=v,k2=v2".'),
):
    """Entrena la versión alternativa y la deja lista para registrarse."""
    setup_mlflow()
    run_retrain(model, cli_text(params))


if __name__ == "__main__":
    app()
