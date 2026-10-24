"""Entrena un modelo de P1, P2 o P3 y lo registra en el experimento de MLflow.

Se lanza desde el MLflow Project (un run por invocación):

    mlflow run . -e train -P question=p1 -P model=hgb
    mlflow run . -e train -P question=p1 -P model=hgb -P params="learning_rate=0.1,max_iter=300"

o, durante el desarrollo, directamente:

    python -m mibici.modeling.train --question p2 --model logistica --params "C=0.5"

Cada pregunta tiene su tabla de modelos (``P1_MODELS`` en ``p1.py``, etc.) y los
hiperparámetros por defecto viven en ``config.py`` (``P1_MODEL_PARAMS``, etc.).
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import MiniBatchKMeans, Birch
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
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
        
    elif role == "candidate" or role == "candidate-hgb":
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

    elif role == "candidate-rf":
        run_name = "p1-random-forest-regressor"
        description = "P1 variante 1: RandomForestRegressor con hiperparámetros base."
        hp = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_leaf": 10,
        }
        params |= {
            "model": "RandomForestRegressor",
            "features": ",".join(P1_FEATURES),
            **{f"hp_{k}": v for k, v in hp.items()},
        }
        model = RandomForestRegressor(random_state=SEED, **hp, n_jobs=-1)
        model.fit(parts["train"][P1_FEATURES], parts["train"]["departures"])
        preds = {s: model.predict(parts[s][P1_FEATURES]) for s in ("valid", "test")}

    elif role == "candidate-ridge":
        run_name = "p1-ridge-regression"
        description = "P1 variante 2: Ridge Regression con estandarización de variables."
        hp = {
            "alpha": 1.0,
        }
        params |= {
            "model": "Ridge",
            "features": ",".join(P1_FEATURES),
            **{f"hp_{k}": v for k, v in hp.items()},
        }
        # Creamos un pipeline que escala los datos y aplica Ridge
        model = make_pipeline(StandardScaler(), Ridge(**hp))
        model.fit(parts["train"][P1_FEATURES], parts["train"]["departures"])
        preds = {s: model.predict(parts[s][P1_FEATURES]) for s in ("valid", "test")}

    elif role == "candidate-hgb-v2":
        run_name = "p1-hgb-regressor-squared-error"
        description = "P1 variante 3: HistGradientBoostingRegressor con función de pérdida squared_error."
        hp = {
            "loss": "squared_error",
            "learning_rate": 0.1,
            "max_iter": 300,
            "max_leaf_nodes": 31,
        }
        params |= {
            "model": "HistGradientBoostingRegressorV2",
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
            metrics |= regression_metrics(parts["test" if s=="test" else "valid"]["departures"] if False else parts[s]["departures"], preds[s], s)
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

    elif role == "candidate" or role == "candidate-hgb":
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

    elif role == "candidate-rf":
        run_name = "p2-random-forest-classifier"
        description = "P2 variante 1: RandomForestClassifier con balanceo de clases."
        hp = {
            "n_estimators": 150,
            "max_depth": 12,
            "min_samples_leaf": 50,
            "class_weight": "balanced",
        }
        params |= {
            "model": "RandomForestClassifier",
            "features": ",".join(P2_FEATURES),
            **{f"hp_{k}": v for k, v in hp.items()},
        }
        model = RandomForestClassifier(random_state=SEED, **hp, n_jobs=-1)
        model.fit(parts["train"][P2_FEATURES], parts["train"]["risk"])
        scores = {s: model.predict_proba(parts[s][P2_FEATURES])[:, 1] for s in ("valid", "test")}
        decision = best_f1_threshold(parts["valid"]["risk"], scores["valid"])
        params["decision_threshold"] = round(decision, 4)
        labels = {s: (scores[s] >= decision).astype(int) for s in ("valid", "test")}

    elif role == "candidate-lr":
        run_name = "p2-logistic-regression"
        description = "P2 variante 2: LogisticRegression con escalado de características."
        hp = {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 500,
        }
        params |= {
            "model": "LogisticRegression",
            "features": ",".join(P2_FEATURES),
            **{f"hp_{k}": v for k, v in hp.items()},
        }
        model = make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED, **hp))
        model.fit(parts["train"][P2_FEATURES], parts["train"]["risk"])
        scores = {s: model.predict_proba(parts[s][P2_FEATURES])[:, 1] for s in ("valid", "test")}
        decision = best_f1_threshold(parts["valid"]["risk"], scores["valid"])
        params["decision_threshold"] = round(decision, 4)
        labels = {s: (scores[s] >= decision).astype(int) for s in ("valid", "test")}

    elif role == "candidate-hgb-v2":
        run_name = "p2-hgb-classifier-fast"
        description = "P2 variante 3: HistGradientBoostingClassifier con mayor tasa de aprendizaje."
        hp = {
            "learning_rate": 0.1,
            "max_iter": 200,
            "max_leaf_nodes": 31,
            "class_weight": "balanced",
            "early_stopping": True,
        }
        params |= {
            "model": "HistGradientBoostingClassifierV2",
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
    scaler = StandardScaler().fit(train[profile_cols])
    x_profile = scaler.transform(train[profile_cols])

    params = common_params("estacion", "perfil del periodo de entrenamiento")
    params |= {"n_stations_train": len(train), "n_stations_test": len(test)}
    extra_metrics: dict[str, float] = {}
    k_curve = None

    if role == "baseline":
        run_name = "p3-baseline-kmeans-geografico"
        description = "P3 baseline: KMeans solo con latitud y longitud (zonas geográficas)."
        cols, k = ["latitude", "longitude"], P3_BASELINE_K
        params |= {"model": "KMeans", "features": "latitude,longitude", "hp_n_clusters": k}
        model = make_pipeline(StandardScaler(), KMeans(k, n_init=20, random_state=SEED))
        labels = model.fit_predict(train[cols])

    elif role == "candidate" or role == "candidate-kmeans":
        run_name = "p3-kmeans-perfiles-uso"
        description = "P3 candidato: KMeans sobre perfiles horarios de uso; k elegido por silueta."
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
        }
        model = make_pipeline(StandardScaler(), KMeans(k, n_init=20, random_state=SEED))
        labels = model.fit_predict(train[cols])
        extra_metrics = {f"train_silhouette_k{kk}": v for kk, v in k_curve.items()}

    elif role == "candidate-minibatch":
        run_name = "p3-minibatch-kmeans"
        description = "P3 variante 1: MiniBatchKMeans sobre perfiles de uso."
        cols = profile_cols
        k = 3
        params |= {"model": "MiniBatchKMeans", "features": "perfil_horario", "hp_n_clusters": k}
        model = make_pipeline(StandardScaler(), MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=10))
        labels = model.fit_predict(train[cols])

    elif role == "candidate-birch":
        run_name = "p3-birch-clustering"
        description = "P3 variante 2: Birch clustering sobre perfiles de uso."
        cols = profile_cols
        k = 3
        params |= {"model": "Birch", "features": "perfil_horario", "hp_n_clusters": k}
        model = make_pipeline(StandardScaler(), Birch(n_clusters=k))
        labels = model.fit_predict(train[cols])

    elif role == "candidate-gmm":
        run_name = "p3-gaussian-mixture"
        description = "P3 variante 3: Gaussian Mixture Model (GMM) sobre perfiles de uso."
        cols = profile_cols
        k = 3
        params |= {"model": "GaussianMixture", "features": "perfil_horario", "hp_n_components": k}
        # GMM no es un estimador estándar con predict_predict dentro de pipeline igual, lo manejamos limpio:
        scaler_gmm = StandardScaler()
        x_scaled = scaler_gmm.fit_transform(train[cols])
        gmm = GaussianMixture(n_components=k, random_state=SEED)
        labels = gmm.fit_predict(x_scaled)
        
        # Envoltorio simple para compatibilidad con el resto del script
        class GMMWrapper:
            def __init__(self, sc, gm):
                self.sc = sc
                self.gm = gm
            def predict(self, X):
                return self.gm.predict(self.sc.transform(X))
            def __getitem__(self, idx): # Para simular model[:-1]
                return self.sc
        model = GMMWrapper(scaler_gmm, gmm)
        k = 3

    # Estabilidad temporal
    common = train[["station_id"]].assign(label_train=labels).merge(test, on="station_id")
    stability = adjusted_rand_score(common["label_train"], model.predict(common[cols]))

    with start_run("P3", role, run_name, description):
        mlflow.log_params(params)
        metrics = cluster_metrics(x_profile, labels, "train_profile")
        # Si usamos GMM o Birch, manejamos el transform de manera segura
        if role == "candidate-gmm":
            metrics |= cluster_metrics(scaler_gmm.transform(train[cols]), labels, "train_own")
        else:
            metrics |= cluster_metrics(model[:-1].transform(train[cols]), labels, "train_own")
            
        metrics["test_stability_ari"] = stability
        metrics["train_min_cluster_size"] = int(np.bincount(labels).min())
        mlflow.log_metrics(metrics | extra_metrics)
        
        log_figure(plots.plot_cluster_map, "p3_mapa_grupos.png", train, labels, title=f"P3 - {run_name}")
        log_figure(plots.plot_cluster_profiles, "p3_perfiles_grupos.png", train, labels, title=f"P3 - {run_name}")
        if k_curve:
            log_figure(plots.plot_k_selection, "p3_seleccion_k.png", list(k_curve), list(k_curve.values()), k)
            
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p3_asignacion_estaciones.csv"
            train[["station_id", "name", "municipio", "latitude", "longitude"]].assign(cluster=labels).to_csv(path, index=False)
            mlflow.log_artifact(str(path))
        log_json(metrics, "metrics.json")
    logger.success(f"{run_name}: k={k}, train_profile_silhouette={metrics['train_profile_silhouette']:.3f}")

app = typer.Typer(help="Entrena un modelo de MiBici y lo registra en MLflow.")

RUNNERS = {"p1": run_p1, "p2": run_p2, "p3": run_p3}
MODELS = {"p1": P1_MODELS, "p2": P2_MODELS, "p3": P3_MODELS}


@app.command()
def main(
    question: str = typer.Option(..., help="p1, p2 o p3."),
    model: str = typer.Option(
        "baseline",
        help="; ".join(f"{q.upper()}: {', '.join(models)}" for q, models in MODELS.items()),
    ),
    params: str = typer.Option("", help='Cambios a los defaults de config.py: "k=v,k2=v2".'),
    stage: str = typer.Option(
        "", help=f"stage_of_work ({', '.join(STAGES)}); por defecto, el del modelo."
    ),
):
    """Entrena un modelo y registra parámetros, métricas y artefactos en un run."""
    question = question.lower()
    if question not in RUNNERS:
        raise typer.BadParameter(f"question debe ser p1, p2 o p3, no {question!r}")
    setup_mlflow()
    RUNNERS[question](model, params, stage)


if __name__ == "__main__":
    app()
