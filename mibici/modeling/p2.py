"""P2: clasificación del riesgo de vaciado o llenado por estación y bloque de 3 horas."""

from __future__ import annotations

from collections.abc import Callable

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline

from mibici import plots
from mibici.config import (
    P2_BLOCK_HOURS,
    P2_DATASET_PATH,
    P2_RISK_CAPACITY_FRACTION,
    SEED,
)
from mibici.modeling.common import ModelResult, linear_preprocessor, split_frames
from mibici.modeling.tracking import common_params, log_figure, log_json, start_run

P2_FEATURES = [
    "block",
    "dow",
    "month",
    "is_weekend",
    "is_holiday",
    "capacity",
    "latitude",
    "longitude",
    "net_flow_lag1",
    "net_flow_lag6",
    "net_flow_lag42",
    "departures_lag1",
    "departures_lag6",
    "departures_lag42",
    "risk_lag6",
    "risk_lag42",
    "abs_net_flow_same_block_4w_mean",
]


def classification_metrics(y_true, y_score, y_pred, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall": recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1": f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}_pr_auc": average_precision_score(y_true, y_score),
        f"{prefix}_roc_auc": roc_auc_score(y_true, y_score),
        f"{prefix}_positive_rate": float(np.mean(y_true)),
    }


def best_f1_threshold(y_true, y_score) -> float:
    """Umbral que maximiza F1 en validación (nunca se elige con test)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


P2_CATEGORICAL = ["block", "dow", "month"]


def threshold_on_valid(
    parts: dict[str, pd.DataFrame], scores: dict[str, np.ndarray]
) -> tuple[float, dict[str, np.ndarray]]:
    """Elige en validación el umbral que maximiza F1 y lo aplica a valid y test."""
    decision = best_f1_threshold(parts["valid"]["risk"], scores["valid"])
    return decision, {s: (scores[s] >= decision).astype(int) for s in ("valid", "test")}


def p2_weekly_persistence(parts: dict[str, pd.DataFrame]) -> ModelResult:
    scores = {
        s: parts[s]["abs_net_flow_same_block_4w_mean"]
        / np.ceil(P2_RISK_CAPACITY_FRACTION * parts[s]["capacity"].astype(float))
        for s in ("valid", "test")
    }
    return ModelResult(
        "p2-baseline-persistencia-semanal",
        "P2 baseline: riesgo si el mismo bloque de la semana anterior estuvo en "
        "riesgo; score = |flujo neto| medio del bloque en 4 semanas / umbral.",
        {"model": "WeeklyPersistence", "label_rule": "risk_lag42"},
        scores,
        {s: parts[s]["risk_lag42"].astype(int) for s in ("valid", "test")},
    )


def p2_historical_rate(parts: dict[str, pd.DataFrame]) -> ModelResult:
    """Frecuencia de riesgo en train por estación, bloque y tipo de día.

    Las estaciones sin historia en train usan la frecuencia de todo el sistema en ese bloque y
    tipo de día.
    """
    keys, system_keys = ["station_id", "block", "is_weekend"], ["block", "is_weekend"]
    rate = parts["train"].groupby(keys)["risk"].mean().rename("rate")
    system = parts["train"].groupby(system_keys)["risk"].mean().rename("system_rate")
    scores, n_fallback = {}, {}
    for s in ("valid", "test"):
        x = parts[s].join(rate, on=keys).join(system, on=system_keys)
        n_fallback[s] = int(x["rate"].isna().sum())
        scores[s] = x["rate"].fillna(x["system_rate"]).to_numpy()
    decision, labels = threshold_on_valid(parts, scores)
    return ModelResult(
        "p2-baseline-tasa-historica-estacion-bloque",
        "P2 baseline: frecuencia de riesgo en train de la estación en ese bloque y tipo de "
        "día; umbral elegido en validación maximizando F1.",
        {
            "model": "HistoricalRiskRate",
            "rate_keys": ",".join(keys),
            "fit_period": "train",
            "fallback": "tasa del sistema por bloque y tipo de dia",
            "n_valid_fallback": n_fallback["valid"],
            "n_test_fallback": n_fallback["test"],
            "decision_threshold": round(decision, 4),
        },
        scores,
        labels,
    )


def p2_logistic(parts: dict[str, pd.DataFrame]) -> ModelResult:
    hp = {"C": 1.0, "class_weight": "balanced", "max_iter": 1000}
    numeric = [c for c in P2_FEATURES if c not in P2_CATEGORICAL]
    model = make_pipeline(
        linear_preprocessor(P2_CATEGORICAL, numeric), LogisticRegression(random_state=SEED, **hp)
    )
    model.fit(parts["train"][P2_FEATURES], parts["train"]["risk"])
    scores = {s: model.predict_proba(parts[s][P2_FEATURES])[:, 1] for s in ("valid", "test")}
    decision, labels = threshold_on_valid(parts, scores)
    return ModelResult(
        "p2-logistica-flujos-lag",
        "P2 modelo lineal: regresión logística balanceada con las variables del candidato "
        "(bloque y calendario en one-hot); umbral elegido en validación maximizando F1.",
        {
            "model": "LogisticRegression",
            "features": ",".join(P2_FEATURES),
            "encoding": f"onehot({','.join(P2_CATEGORICAL)})+standard",
            **{f"hp_{k}": v for k, v in hp.items()},
            "hp_n_iter_fitted": int(model[-1].n_iter_[0]),
            "decision_threshold": round(decision, 4),
        },
        scores,
        labels,
    )


def p2_hgb_candidate(parts: dict[str, pd.DataFrame]) -> ModelResult:
    hp = {
        "learning_rate": 0.05,
        "max_iter": 400,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 100,
        "l2_regularization": 1.0,
        "class_weight": "balanced",
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 30,
    }
    model = HistGradientBoostingClassifier(random_state=SEED, **hp)
    model.fit(parts["train"][P2_FEATURES], parts["train"]["risk"])
    scores = {s: model.predict_proba(parts[s][P2_FEATURES])[:, 1] for s in ("valid", "test")}
    decision, labels = threshold_on_valid(parts, scores)
    return ModelResult(
        "p2-hgb-classifier-flujos-lag",
        "P2 candidato: HistGradientBoostingClassifier balanceado; umbral elegido "
        "en validación maximizando F1.",
        {
            "model": "HistGradientBoostingClassifier",
            "features": ",".join(P2_FEATURES),
            **{f"hp_{k}": v for k, v in hp.items()},
            "hp_n_iter_fitted": model.n_iter_,
            "decision_threshold": round(decision, 4),
        },
        scores,
        labels,
    )


P2_MODELS: dict[str, Callable[[dict[str, pd.DataFrame]], ModelResult]] = {
    "baseline": p2_weekly_persistence,
    "tasa": p2_historical_rate,
    "logistica": p2_logistic,
    "candidate": p2_hgb_candidate,
}


def run_p2(role: str) -> None:
    df = pd.read_parquet(P2_DATASET_PATH)
    parts = split_frames(df)
    params = common_params(f"estacion-bloque {P2_BLOCK_HOURS}h", "1 bloque (3 h)")
    params |= {
        "target": "risk",
        "risk_capacity_fraction": P2_RISK_CAPACITY_FRACTION,
        "n_train": len(parts["train"]),
        "n_valid": len(parts["valid"]),
        "n_test": len(parts["test"]),
    }
    result = P2_MODELS[role](parts)
    params |= result.params
    run_name, scores, labels = result.run_name, result.scores, result.labels
    description = result.description

    with start_run("P2", role, run_name, description):
        mlflow.log_params(params)
        metrics = {}
        for s in ("valid", "test"):
            metrics |= classification_metrics(parts[s]["risk"], scores[s], labels[s], s)
        mlflow.log_metrics(metrics)
        log_figure(
            plots.plot_classification,
            "p2_test_pr_confusion.png",
            parts["test"]["risk"],
            scores["test"],
            labels["test"],
            title=f"P2 - {run_name} (test)",
        )
        log_json(metrics, "metrics.json")
    logger.success(f"{run_name}: test_pr_auc={metrics['test_pr_auc']:.3f}")
