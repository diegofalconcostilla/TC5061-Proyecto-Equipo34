"""Experimento base del proyecto MiBici registrado en MLflow Tracking.

Uso desde la terminal:

    python -m mibici.modeling.train                         # todos los modelos de P1-P3
    python -m mibici.modeling.train --question p2 --model candidate
    python -m mibici.modeling.train --question p1 --model poisson

Cada pregunta tiene una tabla de modelos (``P1_MODELS``, ``P2_MODELS``): de baselines sencillos
a un modelo lineal y al candidato de gradient boosting, para medir cuánto aporta cada nivel de
complejidad.

Cada run registra los parámetros necesarios para reproducirlo (rango de fechas, versión de los
datos, cortes de partición, semilla, modelo e hiperparámetros), métricas con prefijo del
conjunto en que se calculan (``valid_``, ``test_``, ``train_``), al menos un artefacto y
etiquetas de equipo, pregunta y commit de Git. No se registran datos personales.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
import subprocess
import tempfile
from typing import NamedTuple

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
import typer

from mibici import plots
from mibici.config import (
    DATE_END,
    DATE_START,
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    P1_DATASET_PATH,
    P2_BLOCK_HOURS,
    P2_DATASET_PATH,
    P2_RISK_CAPACITY_FRACTION,
    P3_BASELINE_K,
    P3_DATASET_PATH,
    P3_K_RANGE,
    P3_TEST_DATASET_PATH,
    PROJ_ROOT,
    SEED,
    TEAM,
    TEST_START,
    VALID_START,
)
from mibici.dataset import get_data_version
from mibici.features import p3_feature_columns

app = typer.Typer(help="Experimento base de MiBici con MLflow Tracking.")

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


# ---------------------------------------------------------------------------
# Utilidades comunes
# ---------------------------------------------------------------------------
def git_info() -> dict[str, str]:
    """Commit, rama y estado del árbol de trabajo de Git para ligar el run al código."""

    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=PROJ_ROOT, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    status = run("status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": str(bool(status) and status != "unknown").lower(),
    }


def setup_mlflow() -> None:
    """Configura el tracking local (SQLite) y el experimento único del proyecto."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME,
            artifact_location=MLFLOW_ARTIFACTS_DIR.as_uri(),
            tags={"team": TEAM, "project": "MiBici TC5061"},
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def common_params(granularity: str, horizon: str) -> dict:
    """Parámetros de reproducibilidad compartidos por todos los runs."""
    return {
        "date_start": DATE_START,
        "date_end": DATE_END,
        "data_version": get_data_version(),
        "valid_start": VALID_START,
        "test_start": TEST_START,
        "seed": SEED,
        "granularity": granularity,
        "horizon": horizon,
    }


def start_run(question: str, role: str, run_name: str, description: str):
    """Abre un run con nombre descriptivo y las etiquetas obligatorias."""
    tags = {
        "team": TEAM,
        "question": question,
        "model_role": role,
        "mlflow.note.content": description,
        **git_info(),
    }
    return mlflow.start_run(run_name=run_name, tags=tags)


def log_json(data: dict, name: str) -> None:
    """Registra un diccionario como artefacto JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(path))


def log_figure(builder, name: str, *args, **kwargs) -> None:
    """Genera una figura con ``builder`` en un directorio temporal y la registra."""
    with tempfile.TemporaryDirectory() as tmp:
        path = builder(*args, path=Path(tmp) / name, **kwargs)
        mlflow.log_artifact(str(path), artifact_path="figures")


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {s: df[df["split"] == s] for s in ("train", "valid", "test")}


class ModelResult(NamedTuple):
    """Lo que devuelve cada modelo de la tabla de una pregunta."""

    run_name: str
    description: str
    params: dict
    scores: dict[str, np.ndarray]  # predicción (P1) o score continuo (P2), en valid y test
    labels: dict[str, np.ndarray] | None = None  # predicción binaria (solo P2)


def linear_preprocessor(
    categorical: list[str], numeric: list[str], log_counts: list[str] | None = None
) -> ColumnTransformer:
    """One-hot para las variables de calendario y estandarización del resto.

    Un modelo lineal no puede aprender un patrón semanal si ``dow`` entra como número (lunes = 0
    ... domingo = 6); con one-hot cada día tiene su propio coeficiente. Con enlace logarítmico
    (Poisson), los conteos deben entrar como ``log1p``: si entran tal cual, la predicción crece
    exponencialmente con la demanda reciente.
    """
    log_counts = log_counts or []
    return ColumnTransformer(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("log", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), log_counts),
            ("scale", StandardScaler(), [c for c in numeric if c not in log_counts]),
        ]
    )


# ---------------------------------------------------------------------------
# P1: regresión de viajes por estación y día
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# P2: clasificación del riesgo por estación y bloque de 3 horas
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# P3: clustering de estaciones por patrón de uso
# ---------------------------------------------------------------------------
def cluster_metrics(x, labels, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_silhouette": silhouette_score(x, labels, random_state=SEED),
        f"{prefix}_davies_bouldin": davies_bouldin_score(x, labels),
        f"{prefix}_calinski_harabasz": calinski_harabasz_score(x, labels),
    }


def run_p3(role: str) -> None:
    train = pd.read_parquet(P3_DATASET_PATH)
    test = pd.read_parquet(P3_TEST_DATASET_PATH)
    profile_cols = p3_feature_columns(train)
    # Espacio de evaluación común: perfiles de uso estandarizados con estadísticas de train.
    scaler = StandardScaler().fit(train[profile_cols])
    x_profile = scaler.transform(train[profile_cols])

    params = common_params("estacion", "perfil del periodo de entrenamiento")
    params |= {"n_stations_train": len(train), "n_stations_test": len(test)}
    extra_metrics: dict[str, float] = {}

    if role == "baseline":
        run_name = "p3-baseline-kmeans-geografico"
        description = "P3 baseline: KMeans solo con latitud y longitud (zonas geográficas)."
        cols, k = ["latitude", "longitude"], P3_BASELINE_K
        params |= {"model": "KMeans", "features": "latitude,longitude", "hp_n_clusters": k}
        model = make_pipeline(StandardScaler(), KMeans(k, n_init=20, random_state=SEED))
        labels = model.fit_predict(train[cols])
        k_curve = None
    else:
        run_name = "p3-kmeans-perfiles-uso"
        description = (
            "P3 candidato: KMeans sobre perfiles horarios de uso, volumen y balance "
            "de flujos; k elegido por silueta."
        )
        cols = profile_cols
        k_curve = {}
        for k in P3_K_RANGE:
            km = KMeans(k, n_init=20, random_state=SEED).fit(x_profile)
            k_curve[k] = silhouette_score(x_profile, km.labels_, random_state=SEED)
        k = max(k_curve, key=k_curve.get)
        params |= {
            "model": "KMeans",
            "features": "perfil_horario_48+volumen+balance+duracion",
            "n_features": len(cols),
            "hp_n_clusters": k,
            "hp_k_range": f"{min(P3_K_RANGE)}-{max(P3_K_RANGE)}",
        }
        model = make_pipeline(StandardScaler(), KMeans(k, n_init=20, random_state=SEED))
        labels = model.fit_predict(train[cols])
        extra_metrics = {f"train_silhouette_k{kk}": v for kk, v in k_curve.items()}

    # Estabilidad temporal: se asignan las estaciones con su perfil del periodo de test a los
    # grupos aprendidos en train y se compara con su grupo original (ARI).
    common = train[["station_id"]].assign(label_train=labels).merge(test, on="station_id")
    stability = adjusted_rand_score(common["label_train"], model.predict(common[cols]))

    with start_run("P3", role, run_name, description):
        mlflow.log_params(params | {"hp_n_init": 20})
        metrics = cluster_metrics(x_profile, labels, "train_profile")
        metrics |= cluster_metrics(model[:-1].transform(train[cols]), labels, "train_own")
        metrics["test_stability_ari"] = stability
        metrics["train_min_cluster_size"] = int(np.bincount(labels).min())
        mlflow.log_metrics(metrics | extra_metrics)
        log_figure(
            plots.plot_cluster_map,
            "p3_mapa_grupos.png",
            train,
            labels,
            title=f"P3 - {run_name}: estaciones por grupo",
        )
        log_figure(
            plots.plot_cluster_profiles,
            "p3_perfiles_grupos.png",
            train,
            labels,
            title=f"P3 - {run_name}: perfil horario medio por grupo",
        )
        if k_curve:
            log_figure(
                plots.plot_k_selection,
                "p3_seleccion_k.png",
                list(k_curve),
                list(k_curve.values()),
                k,
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p3_asignacion_estaciones.csv"
            train[["station_id", "name", "municipio", "latitude", "longitude"]].assign(
                cluster=labels
            ).to_csv(path, index=False)
            mlflow.log_artifact(str(path))
        log_json(metrics, "metrics.json")
    logger.success(
        f"{run_name}: k={k}, train_profile_silhouette={metrics['train_profile_silhouette']:.3f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
RUNNERS = {"p1": run_p1, "p2": run_p2, "p3": run_p3}
ROLES = {"p1": list(P1_MODELS), "p2": list(P2_MODELS), "p3": ["baseline", "candidate"]}


@app.command()
def main(
    question: str = typer.Option("all", help="p1, p2, p3 o all."),
    model: str = typer.Option(
        "all",
        help="Modelo a correr o all. P1: "
        + ", ".join(ROLES["p1"])
        + ". P2: "
        + ", ".join(ROLES["p2"])
        + ". P3: "
        + ", ".join(ROLES["p3"])
        + ".",
    ),
):
    """Entrena y registra en MLflow los modelos del experimento base."""
    questions = list(RUNNERS) if question == "all" else [question.lower()]
    setup_mlflow()
    for q in questions:
        if model != "all" and model not in ROLES[q]:
            if question == "all":
                continue
            raise typer.BadParameter(
                f"{q} no tiene el modelo {model!r}; opciones: {', '.join(ROLES[q])}"
            )
        roles = ROLES[q] if model == "all" else [model]
        for role in roles:
            logger.info(f"Ejecutando {q.upper()} - {role}")
            RUNNERS[q](role)


if __name__ == "__main__":
    app()
