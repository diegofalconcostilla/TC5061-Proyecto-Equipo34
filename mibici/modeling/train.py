"""Experimento base del proyecto MiBici registrado en MLflow Tracking.

Uso desde la terminal:

    python -m mibici.modeling.train                         # 6 runs: baseline y candidato de P1-P3
    python -m mibici.modeling.train --question p2 --model candidate

Cada run registra los parámetros necesarios para reproducirlo (rango de fechas, versión de los
datos, cortes de partición, semilla, modelo e hiperparámetros), métricas con prefijo del
conjunto en que se calculan (``valid_``, ``test_``, ``train_``), al menos un artefacto y
etiquetas de equipo, pregunta y commit de Git. No se registran datos personales.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
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
from sklearn.preprocessing import StandardScaler
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

    if role == "baseline":
        run_name = "p1-baseline-naive-estacional-7d"
        description = "P1 baseline: predice los viajes del mismo día de la semana anterior."
        params |= {"model": "SeasonalNaive", "season_lag_days": 7}
        preds = {s: parts[s]["departures_lag7"] for s in ("valid", "test")}
    else:
        run_name = "p1-hgb-regressor-lags-calendario"
        description = (
            "P1 candidato: HistGradientBoostingRegressor (Poisson) con lags y calendario."
        )
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
        params |= {
            "model": "HistGradientBoostingRegressor",
            "features": ",".join(P1_FEATURES),
            **{f"hp_{k}": v for k, v in hp.items()},
        }
        model = HistGradientBoostingRegressor(random_state=SEED, **hp)
        model.fit(parts["train"][P1_FEATURES], parts["train"]["departures"])
        params["hp_n_iter_fitted"] = model.n_iter_
        preds = {s: model.predict(parts[s][P1_FEATURES]) for s in ("valid", "test")}

    with start_run("P1", role, run_name, description):
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
            title=f"P1 - {run_name} (test, suma de estaciones)",
        )
        log_json(metrics, "metrics.json")
    logger.success(f"{run_name}: test_mae={metrics['test_mae']:.3f}")


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

    if role == "baseline":
        run_name = "p2-baseline-persistencia-semanal"
        description = (
            "P2 baseline: riesgo si el mismo bloque de la semana anterior estuvo en "
            "riesgo; score = |flujo neto| medio del bloque en 4 semanas / umbral."
        )
        params |= {"model": "WeeklyPersistence", "label_rule": "risk_lag42"}
        threshold = np.ceil(P2_RISK_CAPACITY_FRACTION * df["capacity"].astype(float))
        scores = {
            s: (parts[s]["abs_net_flow_same_block_4w_mean"] / threshold[parts[s].index])
            for s in ("valid", "test")
        }
        labels = {s: parts[s]["risk_lag42"].astype(int) for s in ("valid", "test")}
    else:
        run_name = "p2-hgb-classifier-flujos-lag"
        description = (
            "P2 candidato: HistGradientBoostingClassifier balanceado; umbral elegido "
            "en validación maximizando F1."
        )
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
        params |= {
            "model": "HistGradientBoostingClassifier",
            "features": ",".join(P2_FEATURES),
            **{f"hp_{k}": v for k, v in hp.items()},
        }
        model = HistGradientBoostingClassifier(random_state=SEED, **hp)
        model.fit(parts["train"][P2_FEATURES], parts["train"]["risk"])
        params["hp_n_iter_fitted"] = model.n_iter_
        scores = {s: model.predict_proba(parts[s][P2_FEATURES])[:, 1] for s in ("valid", "test")}
        decision = best_f1_threshold(parts["valid"]["risk"], scores["valid"])
        params["decision_threshold"] = round(decision, 4)
        labels = {s: (scores[s] >= decision).astype(int) for s in ("valid", "test")}

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


@app.command()
def main(
    question: str = typer.Option("all", help="p1, p2, p3 o all."),
    model: str = typer.Option("all", help="baseline, candidate o all."),
):
    """Entrena y registra en MLflow los modelos del experimento base."""
    questions = list(RUNNERS) if question == "all" else [question.lower()]
    roles = ["baseline", "candidate"] if model == "all" else [model]
    setup_mlflow()
    for q in questions:
        for role in roles:
            logger.info(f"Ejecutando {q.upper()} - {role}")
            RUNNERS[q](role)


if __name__ == "__main__":
    app()
