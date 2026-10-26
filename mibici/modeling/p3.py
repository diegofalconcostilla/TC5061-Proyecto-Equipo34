"""P3: clustering de estaciones por patrón de uso."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import Birch, KMeans, MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mibici import plots
from mibici.config import (
    P3_DATASET_PATH,
    P3_K_RANGE,
    P3_MODEL_PARAMS,
    P3_RULE_EVENING_HOURS,
    P3_RULE_MORNING_HOURS,
    P3_RULE_THRESHOLDS,
    P3_TEST_DATASET_PATH,
    SEED,
)
from mibici.features import p3_feature_columns
from mibici.modeling.artifacts import (
    log_config,
    log_data,
    log_figure,
    log_json,
    log_model,
    log_table,
)
from mibici.modeling.common import ModelSpec, RunSummary, get_spec, run_name_with_changes
from mibici.modeling.params import resolve_params
from mibici.modeling.tracking import common_params, start_run


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
    estimator: object | None = None  # modelo entrenado que se registra; None en reglas
    features: list[str] | None = None  # columnas de entrada de ``estimator``


def p3_geographic_kmeans(train: pd.DataFrame, x_profile: np.ndarray, hp: dict) -> P3Result:
    cols, k = ["latitude", "longitude"], hp["n_clusters"]
    model = make_pipeline(StandardScaler(), KMeans(k, n_init=hp["n_init"], random_state=SEED))
    labels = model.fit_predict(train[cols])
    return P3Result(
        "p3-baseline-kmeans-geografico",
        "P3 baseline: KMeans solo con latitud y longitud (zonas geográficas).",
        {
            "model_class": "KMeans",
            "features": "latitude,longitude",
            **{f"hp_{key}": v for key, v in hp.items()},
        },
        labels,
        lambda df: model.predict(df[cols]),
        model[:-1].transform(train[cols]),
        estimator=model,
        features=cols,
    )


def p3_morning_evening_rule(train: pd.DataFrame, x_profile: np.ndarray, hp: dict) -> P3Result:
    low, high = P3_RULE_THRESHOLDS
    morning, evening = P3_RULE_MORNING_HOURS, P3_RULE_EVENING_HOURS
    return P3Result(
        "p3-baseline-regla-manana-tarde",
        "P3 baseline: regla fija por la fracción de salidas pico de días hábiles en la "
        "mañana; > 0.6 origen matutino, < 0.4 destino matutino, resto mixta.",
        {
            "model_class": "MorningEveningRule",
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


def p3_profile_kmeans(train: pd.DataFrame, x_profile: np.ndarray, hp: dict) -> P3Result:
    """KMeans sobre perfiles de uso; con ``n_clusters = 0`` elige k en P3_K_RANGE por silueta."""
    cols, n_init = p3_feature_columns(train), hp["n_init"]
    k_curve = None
    if hp["n_clusters"] == 0:
        k_curve = {}
        for k in P3_K_RANGE:
            km = KMeans(k, n_init=n_init, random_state=SEED).fit(x_profile)
            k_curve[k] = silhouette_score(x_profile, km.labels_, random_state=SEED)
        k = max(k_curve, key=k_curve.get)
        k_params = {"hp_k_range": f"{min(P3_K_RANGE)}-{max(P3_K_RANGE)}"}
        description = "k elegido por silueta"
    else:
        k, k_params, description = hp["n_clusters"], {}, f"k fijo = {hp['n_clusters']}"
    model = make_pipeline(StandardScaler(), KMeans(k, n_init=n_init, random_state=SEED))
    labels = model.fit_predict(train[cols])
    return P3Result(
        "p3-kmeans-perfiles-uso",
        "P3 candidato: KMeans sobre perfiles horarios de uso, volumen y balance "
        f"de flujos; {description}.",
        {
            "model_class": "KMeans",
            "features": "perfil_horario_48+volumen+balance+duracion",
            "n_features": len(cols),
            "hp_n_clusters": k,
            **k_params,
            "hp_n_init": n_init,
        },
        labels,
        lambda df: model.predict(df[cols]),
        model[:-1].transform(train[cols]),
        k_curve,
        estimator=model,
        features=cols,
    )


def p3_profile_clustering(
    train: pd.DataFrame, estimator, run_name: str, description: str, hp: dict
) -> P3Result:
    """Agrupa los perfiles de uso (mismas variables que el KMeans candidato) con ``estimator``."""
    cols = p3_feature_columns(train)
    model = make_pipeline(StandardScaler(), estimator)
    labels = model.fit_predict(train[cols])
    return P3Result(
        run_name,
        description,
        {
            "model_class": type(estimator).__name__,
            "features": "perfil_horario_48+volumen+balance+duracion",
            "n_features": len(cols),
            **{f"hp_{key}": v for key, v in hp.items()},
        },
        labels,
        lambda df: model.predict(df[cols]),
        model[:-1].transform(train[cols]),
        estimator=model,
        features=cols,
    )


def p3_minibatch_kmeans(train: pd.DataFrame, x_profile: np.ndarray, hp: dict) -> P3Result:
    return p3_profile_clustering(
        train,
        MiniBatchKMeans(
            hp["n_clusters"], n_init=hp["n_init"], batch_size=hp["batch_size"], random_state=SEED
        ),
        "p3-minibatch-kmeans-perfiles-uso",
        "P3 candidato: MiniBatchKMeans sobre perfiles de uso (KMeans aproximado por lotes, "
        "más barato de entrenar).",
        hp,
    )


def p3_birch(train: pd.DataFrame, x_profile: np.ndarray, hp: dict) -> P3Result:
    return p3_profile_clustering(
        train,
        Birch(n_clusters=hp["n_clusters"], threshold=hp["threshold"]),
        "p3-birch-perfiles-uso",
        "P3 candidato: Birch (árbol de características y agrupamiento jerárquico) sobre "
        "perfiles de uso.",
        hp,
    )


def p3_gaussian_mixture(train: pd.DataFrame, x_profile: np.ndarray, hp: dict) -> P3Result:
    return p3_profile_clustering(
        train,
        GaussianMixture(
            hp["n_clusters"],
            covariance_type=hp["covariance_type"],
            n_init=hp["n_init"],
            random_state=SEED,
        ),
        "p3-gaussian-mixture-perfiles-uso",
        "P3 candidato: mezcla de gaussianas sobre perfiles de uso (grupos elípticos y "
        "asignación probabilística).",
        hp,
    )


P3_MODELS: dict[str, ModelSpec] = {
    "baseline": ModelSpec(p3_geographic_kmeans, "kmeans", "baseline", P3_MODEL_PARAMS["baseline"]),
    "regla": ModelSpec(p3_morning_evening_rule, "rule", "baseline"),
    "kmeans": ModelSpec(p3_profile_kmeans, "kmeans", "candidate", P3_MODEL_PARAMS["kmeans"]),
    "minibatch": ModelSpec(
        p3_minibatch_kmeans, "kmeans", "candidate", P3_MODEL_PARAMS["minibatch"]
    ),
    "birch": ModelSpec(p3_birch, "hierarchical", "candidate", P3_MODEL_PARAMS["birch"]),
    "gmm": ModelSpec(p3_gaussian_mixture, "mixture", "candidate", P3_MODEL_PARAMS["gmm"]),
}


def run_p3(role: str, params: str = "", stage: str = "", nested: bool = False) -> RunSummary:
    spec = get_spec(P3_MODELS, "p3", role)
    hp = resolve_params(spec.defaults, params)
    train = pd.read_parquet(P3_DATASET_PATH)
    test = pd.read_parquet(P3_TEST_DATASET_PATH)
    profile_cols = p3_feature_columns(train)
    # Espacio de evaluación común: perfiles de uso estandarizados con estadísticas de train.
    scaler = StandardScaler().fit(train[profile_cols])
    x_profile = scaler.transform(train[profile_cols])

    run_params = common_params("p3", role, "estacion", "perfil del periodo de entrenamiento")
    run_params |= {"n_stations_train": len(train), "n_stations_test": len(test)}
    result = spec.fit(train, x_profile, hp)
    run_params |= result.params
    labels, k_curve = result.labels, result.k_curve
    run_name = run_name_with_changes(result.run_name, hp, spec.defaults)
    k = len(np.unique(labels))
    extra_metrics = {f"train_silhouette_k{kk}": v for kk, v in (k_curve or {}).items()}

    # Estabilidad temporal: se asignan las estaciones con su perfil del periodo de test a los
    # grupos aprendidos en train y se compara con su grupo original (ARI).
    common = train[["station_id"]].assign(label_train=labels).merge(test, on="station_id")
    stability = adjusted_rand_score(common["label_train"], result.assign(common))

    with start_run(
        "P3", role, spec.family, stage or spec.stage, run_name, result.description, nested
    ) as run:
        mlflow.log_params(run_params)
        metrics = cluster_metrics(x_profile, labels, "train_profile")
        metrics |= cluster_metrics(result.x_own, labels, "train_own")
        metrics["test_stability_ari"] = stability
        metrics["train_min_cluster_size"] = int(np.bincount(labels).min())
        mlflow.log_metrics(metrics | extra_metrics)
        log_data(train, "p3", P3_DATASET_PATH, target=None)
        log_config({"question": "p3", "model": role, "hyperparameters": hp, "params": run_params})
        if result.estimator is not None:
            log_model(result.estimator, train[result.features].head(5))
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
        log_table(
            train[["station_id", "name", "municipio", "latitude", "longitude"]].assign(
                cluster=labels
            ),
            "p3_asignacion_estaciones.csv",
        )
        log_json(metrics, "metrics.json")
    logger.success(
        f"{run_name}: k={k}, train_profile_silhouette={metrics['train_profile_silhouette']:.3f}"
    )
    return RunSummary(run.info.run_id, run_name, hp, metrics)
