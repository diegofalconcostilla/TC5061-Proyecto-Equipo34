"""P3: clustering de estaciones por patrón de uso."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import tempfile
from typing import NamedTuple

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici import plots
from mibici.config import (
    P3_BASELINE_K,
    P3_DATASET_PATH,
    P3_K_RANGE,
    P3_RULE_EVENING_HOURS,
    P3_RULE_MORNING_HOURS,
    P3_RULE_THRESHOLDS,
    P3_TEST_DATASET_PATH,
    SEED,
)
from mibici.features import p3_feature_columns
from mibici.modeling.tracking import common_params, log_figure, log_json, start_run


def cluster_metrics(x, labels, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_silhouette": silhouette_score(x, labels, random_state=SEED),
        f"{prefix}_davies_bouldin": davies_bouldin_score(x, labels),
        f"{prefix}_calinski_harabasz": calinski_harabasz_score(x, labels),
    }


def p3_morning_share(df: pd.DataFrame) -> pd.Series:
    """Fracción de las salidas pico de días hábiles (mañana + tarde) que ocurren en la mañana."""
    morning = df[[f"dep_wd_h{h:02d}" for h in P3_RULE_MORNING_HOURS]].sum(axis=1)
    evening = df[[f"dep_wd_h{h:02d}" for h in P3_RULE_EVENING_HOURS]].sum(axis=1)
    return (morning / (morning + evening)).fillna(0.5)


def p3_rule_labels(df: pd.DataFrame) -> np.ndarray:
    """0 = origen matutino (residencial), 1 = destino matutino (laboral), 2 = mixta.

    Los cortes son fijos: la regla no se ajusta con datos, así que se aplica igual a train y a
    test y su estabilidad (ARI) mide cuánto cambia el perfil de cada estación entre periodos.
    """
    low, high = P3_RULE_THRESHOLDS
    share = p3_morning_share(df)
    return np.where(share > high, 0, np.where(share < low, 1, 2))


class P3Result(NamedTuple):
    """Lo que devuelve cada modelo de la tabla de P3."""

    run_name: str
    description: str
    params: dict
    labels: np.ndarray  # grupo de cada estación de train
    assign: Callable[[pd.DataFrame], np.ndarray]  # asigna grupo a perfiles de otro periodo
    x_own: np.ndarray  # espacio en que el modelo agrupa, para la silueta propia
    k_curve: dict[int, float] | None = None  # silueta por k, si el modelo elige k


def p3_geographic_kmeans(train: pd.DataFrame, x_profile: np.ndarray) -> P3Result:
    cols, k = ["latitude", "longitude"], P3_BASELINE_K
    model = make_pipeline(StandardScaler(), KMeans(k, n_init=20, random_state=SEED))
    labels = model.fit_predict(train[cols])
    return P3Result(
        "p3-baseline-kmeans-geografico",
        "P3 baseline: KMeans solo con latitud y longitud (zonas geográficas).",
        {"model": "KMeans", "features": "latitude,longitude", "hp_n_clusters": k, "hp_n_init": 20},
        labels,
        lambda df: model.predict(df[cols]),
        model[:-1].transform(train[cols]),
    )


def p3_morning_evening_rule(train: pd.DataFrame, x_profile: np.ndarray) -> P3Result:
    low, high = P3_RULE_THRESHOLDS
    morning, evening = P3_RULE_MORNING_HOURS, P3_RULE_EVENING_HOURS
    return P3Result(
        "p3-baseline-regla-manana-tarde",
        "P3 baseline: regla fija por la fracción de salidas pico de días hábiles en la "
        "mañana; > 0.6 origen matutino, < 0.4 destino matutino, resto mixta.",
        {
            "model": "MorningEveningRule",
            "features": "morning_share",
            "morning_hours": f"{morning.start:02d}:00-{morning.stop:02d}:00",
            "evening_hours": f"{evening.start:02d}:00-{evening.stop:02d}:00",
            "rule_low_threshold": low,
            "rule_high_threshold": high,
        },
        p3_rule_labels(train),
        p3_rule_labels,
        p3_morning_share(train).to_frame().to_numpy(),
    )


def p3_profile_kmeans(train: pd.DataFrame, x_profile: np.ndarray) -> P3Result:
    cols = p3_feature_columns(train)
    k_curve = {}
    for k in P3_K_RANGE:
        km = KMeans(k, n_init=20, random_state=SEED).fit(x_profile)
        k_curve[k] = silhouette_score(x_profile, km.labels_, random_state=SEED)
    k = max(k_curve, key=k_curve.get)
    model = make_pipeline(StandardScaler(), KMeans(k, n_init=20, random_state=SEED))
    labels = model.fit_predict(train[cols])
    return P3Result(
        "p3-kmeans-perfiles-uso",
        "P3 candidato: KMeans sobre perfiles horarios de uso, volumen y balance "
        "de flujos; k elegido por silueta.",
        {
            "model": "KMeans",
            "features": "perfil_horario_48+volumen+balance+duracion",
            "n_features": len(cols),
            "hp_n_clusters": k,
            "hp_k_range": f"{min(P3_K_RANGE)}-{max(P3_K_RANGE)}",
            "hp_n_init": 20,
        },
        labels,
        lambda df: model.predict(df[cols]),
        model[:-1].transform(train[cols]),
        k_curve,
    )


P3_MODELS: dict[str, Callable[[pd.DataFrame, np.ndarray], P3Result]] = {
    "baseline": p3_geographic_kmeans,
    "regla": p3_morning_evening_rule,
    "candidate": p3_profile_kmeans,
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
    result = P3_MODELS[role](train, x_profile)
    params |= result.params
    run_name, labels, k_curve = result.run_name, result.labels, result.k_curve
    k = len(np.unique(labels))
    extra_metrics = {f"train_silhouette_k{kk}": v for kk, v in (k_curve or {}).items()}

    # Estabilidad temporal: se asignan las estaciones con su perfil del periodo de test a los
    # grupos aprendidos en train y se compara con su grupo original (ARI).
    common = train[["station_id"]].assign(label_train=labels).merge(test, on="station_id")
    stability = adjusted_rand_score(common["label_train"], result.assign(common))

    with start_run("P3", role, run_name, result.description):
        mlflow.log_params(params)
        metrics = cluster_metrics(x_profile, labels, "train_profile")
        metrics |= cluster_metrics(result.x_own, labels, "train_own")
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
