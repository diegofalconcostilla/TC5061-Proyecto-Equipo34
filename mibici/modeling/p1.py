"""P1: regresión de viajes iniciados por estación y día (horizonte de 1 día)."""

from __future__ import annotations

from collections.abc import Callable

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
    SEED,
)
from mibici.modeling.common import ModelResult, linear_preprocessor, split_frames
from mibici.modeling.tracking import common_params, log_figure, log_json, start_run

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


def p1_seasonal_naive(parts: dict[str, pd.DataFrame]) -> ModelResult:
    return ModelResult(
        "p1-baseline-naive-estacional-7d",
        "P1 baseline: predice los viajes del mismo día de la semana anterior.",
        {"model": "SeasonalNaive", "season_lag_days": 7},
        {s: parts[s]["departures_lag7"] for s in ("valid", "test")},
    )


def p1_rolling_mean_7d(parts: dict[str, pd.DataFrame]) -> ModelResult:
    return ModelResult(
        "p1-baseline-media-movil-7d",
        "P1 baseline: predice el promedio de viajes de los 7 días anteriores.",
        {"model": "RollingMean", "window_days": 7},
        {s: parts[s]["departures_roll7_mean"] for s in ("valid", "test")},
    )


def p1_station_weekday_mean(parts: dict[str, pd.DataFrame]) -> ModelResult:
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
            "model": "StationWeekdayMean",
            "fit_period": "train",
            "fallback": "departures_roll28_mean",
            "n_valid_fallback": n_fallback["valid"],
            "n_test_fallback": n_fallback["test"],
        },
        preds,
    )


def p1_poisson_glm(parts: dict[str, pd.DataFrame]) -> ModelResult:
    hp = {"alpha": 1e-4, "max_iter": 1000}
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
            "model": "PoissonRegressor",
            "features": ",".join(P1_FEATURES),
            "encoding": f"onehot({','.join(P1_CATEGORICAL)})+log1p(conteos)+standard",
            **{f"hp_{k}": v for k, v in hp.items()},
            "hp_n_iter_fitted": int(model[-1].n_iter_),
        },
        {s: model.predict(parts[s][P1_FEATURES]) for s in ("valid", "test")},
    )


def p1_hgb_candidate(parts: dict[str, pd.DataFrame]) -> ModelResult:
    hp = {
        "loss": "poisson",
        "learning_rate": 0.05,
        "max_iter": 500,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 50,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 30,
    }
    model = HistGradientBoostingRegressor(random_state=SEED, **hp)
    model.fit(parts["train"][P1_FEATURES], parts["train"]["departures"])
    return ModelResult(
        "p1-hgb-regressor-lags-calendario",
        "P1 candidato: HistGradientBoostingRegressor (Poisson) con lags y calendario.",
        {
            "model": "HistGradientBoostingRegressor",
            "features": ",".join(P1_FEATURES),
            **{f"hp_{k}": v for k, v in hp.items()},
            "hp_n_iter_fitted": model.n_iter_,
        },
        {s: model.predict(parts[s][P1_FEATURES]) for s in ("valid", "test")},
    )


P1_MODELS: dict[str, Callable[[dict[str, pd.DataFrame]], ModelResult]] = {
    "baseline": p1_seasonal_naive,
    "roll7": p1_rolling_mean_7d,
    "historico": p1_station_weekday_mean,
    "poisson": p1_poisson_glm,
    "candidate": p1_hgb_candidate,
}


def run_p1(role: str) -> None:
    df = pd.read_parquet(P1_DATASET_PATH)
    parts = split_frames(df)
    params = common_params("estacion-dia", "1 dia")
    params |= {
        "target": "departures",
        "n_train": len(parts["train"]),
        "n_valid": len(parts["valid"]),
        "n_test": len(parts["test"]),
    }
    result = P1_MODELS[role](parts)
    params |= result.params
    preds = result.scores

    with start_run("P1", role, result.run_name, result.description):
        mlflow.log_params(params)
        metrics = {}
        for s in ("valid", "test"):
            metrics |= regression_metrics(parts[s]["departures"], preds[s], s)
        mlflow.log_metrics(metrics)
        log_figure(
            plots.plot_daily_forecast,
            "p1_test_viajes_diarios.png",
            parts["test"]["date"],
            parts["test"]["departures"].values,
            preds["test"],
            title=f"P1 - {result.run_name} (test, suma de estaciones)",
        )
        log_json(metrics, "metrics.json")
    logger.success(f"{result.run_name}: test_mae={metrics['test_mae']:.3f}")
