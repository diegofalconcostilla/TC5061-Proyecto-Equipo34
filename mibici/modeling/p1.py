"""P1: regresión de viajes iniciados por estación y día (horizonte de 1 día)."""

from __future__ import annotations

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import make_pipeline

from mibici import plots
from mibici.config import (
    P1_DATASET_PATH,
    P1_MODEL_PARAMS,
    SEED,
)
from mibici.modeling.artifacts import log_config, log_data, log_figure, log_json, log_model
from mibici.modeling.common import (
    ModelResult,
    ModelSpec,
    get_spec,
    linear_preprocessor,
    run_name_with_changes,
    split_frames,
)
from mibici.modeling.params import resolve_params
from mibici.modeling.tracking import common_params, start_run

P1_FEATURES = [
    "departures_lag1",
    "departures_lag2",
    "departures_lag7",
    "departures_lag14",
    "departures_roll7_mean",
    "departures_roll28_mean",
    "arrivals_lag1",
    "arrivals_lag7",
    "arrivals_roll7_mean",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
    "capacity",
    "latitude",
    "longitude",
]


def regression_metrics(y_true, y_pred, prefix: str) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}_rmse": root_mean_squared_error(y_true, y_pred),
        f"{prefix}_wape": np.abs(y_true - y_pred).sum() / y_true.sum(),
        f"{prefix}_r2": r2_score(y_true, y_pred),
    }


P1_CATEGORICAL = ["dow", "month"]


def p1_seasonal_naive(parts: dict[str, pd.DataFrame], hp: dict) -> ModelResult:
    return ModelResult(
        "p1-baseline-naive-estacional-7d",
        "P1 baseline: predice los viajes del mismo día de la semana anterior.",
        {"model_class": "SeasonalNaive", "season_lag_days": 7},
        {s: parts[s]["departures_lag7"] for s in ("valid", "test")},
    )


def p1_rolling_mean_7d(parts: dict[str, pd.DataFrame], hp: dict) -> ModelResult:
    return ModelResult(
        "p1-baseline-media-movil-7d",
        "P1 baseline: predice el promedio de viajes de los 7 días anteriores.",
        {"model_class": "RollingMean", "window_days": 7},
        {s: parts[s]["departures_roll7_mean"] for s in ("valid", "test")},
    )


def p1_station_weekday_mean(parts: dict[str, pd.DataFrame], hp: dict) -> ModelResult:
    """Promedio de train por estación y día de la semana.

    Las estaciones sin historia en train (abiertas en 2026) usan su propio promedio de 28 días.
    """
    hist = parts["train"].groupby(["station_id", "dow"])["departures"].mean().rename("hist")
    preds, n_fallback = {}, {}
    for s in ("valid", "test"):
        x = parts[s].join(hist, on=["station_id", "dow"])
        n_fallback[s] = int(x["hist"].isna().sum())
        preds[s] = x["hist"].fillna(x["departures_roll28_mean"])
    return ModelResult(
        "p1-baseline-historico-estacion-dia-semana",
        "P1 baseline: promedio de train de la estación en ese día de la semana; estaciones "
        "sin historia en train usan su promedio de 28 días.",
        {
            "model_class": "StationWeekdayMean",
            "fit_period": "train",
            "fallback": "departures_roll28_mean",
            "n_valid_fallback": n_fallback["valid"],
            "n_test_fallback": n_fallback["test"],
        },
        preds,
    )


def p1_poisson_glm(parts: dict[str, pd.DataFrame], hp: dict) -> ModelResult:
    numeric = [c for c in P1_FEATURES if c not in P1_CATEGORICAL]
    counts = [c for c in numeric if c.startswith(("departures_", "arrivals_"))]
    model = make_pipeline(
        linear_preprocessor(P1_CATEGORICAL, numeric, log_counts=counts), PoissonRegressor(**hp)
    )
    model.fit(parts["train"][P1_FEATURES], parts["train"]["departures"])
    return ModelResult(
        "p1-poisson-lineal-lags-calendario",
        "P1 modelo lineal: regresión de Poisson con las variables del candidato (calendario "
        "en one-hot, conteos en log1p); escalón entre los baselines y el gradient boosting.",
        {
            "model_class": "PoissonRegressor",
            "features": ",".join(P1_FEATURES),
            "encoding": f"onehot({','.join(P1_CATEGORICAL)})+log1p(conteos)+standard",
            **{f"hp_{k}": v for k, v in hp.items()},
            "hp_n_iter_fitted": int(model[-1].n_iter_),
        },
        {s: model.predict(parts[s][P1_FEATURES]) for s in ("valid", "test")},
        estimator=model,
        features=P1_FEATURES,
    )


def p1_hgb(parts: dict[str, pd.DataFrame], hp: dict) -> ModelResult:
    model = HistGradientBoostingRegressor(random_state=SEED, **hp)
    model.fit(parts["train"][P1_FEATURES], parts["train"]["departures"])
    return ModelResult(
        "p1-hgb-regressor-lags-calendario",
        "P1 candidato: HistGradientBoostingRegressor (Poisson) con lags y calendario.",
        {
            "model_class": "HistGradientBoostingRegressor",
            "features": ",".join(P1_FEATURES),
            **{f"hp_{k}": v for k, v in hp.items()},
            "hp_n_iter_fitted": model.n_iter_,
        },
        {s: model.predict(parts[s][P1_FEATURES]) for s in ("valid", "test")},
        estimator=model,
        features=P1_FEATURES,
    )


P1_MODELS: dict[str, ModelSpec] = {
    "baseline": ModelSpec(p1_seasonal_naive, "naive", "baseline"),
    "roll7": ModelSpec(p1_rolling_mean_7d, "naive", "baseline"),
    "historico": ModelSpec(p1_station_weekday_mean, "historical", "baseline"),
    "poisson": ModelSpec(p1_poisson_glm, "linear", "candidate", P1_MODEL_PARAMS["poisson"]),
    "hgb": ModelSpec(p1_hgb, "boosting", "candidate", P1_MODEL_PARAMS["hgb"]),
}


def run_p1(role: str, params: str = "", stage: str = "") -> None:
    spec = get_spec(P1_MODELS, "p1", role)
    hp = resolve_params(spec.defaults, params)
    df = pd.read_parquet(P1_DATASET_PATH)
    parts = split_frames(df)
    run_params = common_params("p1", role, "estacion-dia", "1 dia")
    run_params |= {
        "target": "departures",
        "n_train": len(parts["train"]),
        "n_valid": len(parts["valid"]),
        "n_test": len(parts["test"]),
    }
    result = spec.fit(parts, hp)
    run_params |= result.params
    preds = result.scores
    run_name = run_name_with_changes(result.run_name, hp, spec.defaults)

    with start_run("P1", role, spec.family, stage or spec.stage, run_name, result.description):
        mlflow.log_params(run_params)
        metrics = {}
        for s in ("valid", "test"):
            metrics |= regression_metrics(parts[s]["departures"], preds[s], s)
        mlflow.log_metrics(metrics)
        log_data(parts["train"], "p1", P1_DATASET_PATH, target="departures")
        log_config({"question": "p1", "model": role, "hyperparameters": hp, "params": run_params})
        if result.estimator is not None:
            log_model(result.estimator, parts["train"][result.features].head(5))
        log_figure(
            plots.plot_daily_forecast,
            "p1_test_viajes_diarios.png",
            parts["test"]["date"],
            parts["test"]["departures"].values,
            preds["test"],
            title=f"P1 - {run_name} (test, suma de estaciones)",
        )
        log_figure(
            plots.plot_regression_diagnostics,
            "p1_test_real_vs_predicho_residuales.png",
            parts["test"]["departures"].values,
            preds["test"],
            title=f"P1 - {run_name} (test, estación-día)",
        )
        log_json(metrics, "metrics.json")
    logger.success(f"{run_name}: test_mae={metrics['test_mae']:.3f}")
