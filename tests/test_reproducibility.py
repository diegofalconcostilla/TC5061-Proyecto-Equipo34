"""Entrenar dos veces con la misma semilla (SEED) da exactamente las mismas predicciones.

Llaman a las funciones de entrenamiento del proyecto (no solo a scikit-learn) con datos
sintéticos que tienen las mismas columnas que los datasets reales.
"""

import numpy as np
import pandas as pd
import pytest

from mibici.config import P1_MODEL_PARAMS, P2_MODEL_PARAMS, P3_MODEL_PARAMS
from mibici.features import P3_SCALAR_FEATURES
from mibici.modeling.p1 import P1_FEATURES, p1_hgb, p1_random_forest
from mibici.modeling.p2 import P2_FEATURES, p2_hgb, p2_random_forest
from mibici.modeling.p3 import p3_gaussian_mixture, p3_minibatch_kmeans


def make_parts(features: list[str], target: str, n: int = 600, seed: int = 0):
    """Train, valid y test sintéticos; el objetivo depende de las features para que haya señal."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(features))), columns=features)
    signal = df.iloc[:, 0] + 0.5 * df.iloc[:, 1]
    if target == "risk":
        df[target] = (signal + rng.normal(scale=0.5, size=n) > 0.8).astype(int)
    else:
        df[target] = rng.poisson(np.exp(1 + 0.3 * signal))
    split = np.repeat(["train", "valid", "test"], [n // 2, n // 4, n - n // 2 - n // 4])
    return {s: df[split == s] for s in ("train", "valid", "test")}


def make_profiles(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hourly = [f"dep_{d}_h{h:02d}" for d in ("wd", "we") for h in range(24)]
    cols = hourly + P3_SCALAR_FEATURES
    centers = rng.normal(scale=3, size=(3, len(cols)))
    x = centers[rng.integers(0, 3, size=n)] + rng.normal(size=(n, len(cols)))
    return pd.DataFrame(x, columns=cols)


def assert_same_scores(first, second):
    # Los random forest (n_jobs=-1) entrenan árboles idénticos, pero al predecir suman los
    # árboles en paralelo y el orden de la suma cambia los últimos bits (~1e-15). Por eso se
    # compara con una tolerancia de punto flotante y no bit a bit.
    for s in ("valid", "test"):
        np.testing.assert_allclose(first.scores[s], second.scores[s], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    "fit, params",
    [(p1_hgb, P1_MODEL_PARAMS["hgb"]), (p1_random_forest, P1_MODEL_PARAMS["rf"])],
    ids=["hgb", "rf"],
)
def test_p1_training_is_reproducible(fit, params):
    parts = make_parts(P1_FEATURES, "departures")
    assert_same_scores(fit(parts, params), fit(parts, params))


@pytest.mark.parametrize(
    "fit, params",
    [(p2_hgb, P2_MODEL_PARAMS["hgb"]), (p2_random_forest, P2_MODEL_PARAMS["rf"])],
    ids=["hgb", "rf"],
)
def test_p2_training_is_reproducible(fit, params):
    parts = make_parts(P2_FEATURES, "risk")
    first, second = fit(parts, params), fit(parts, params)
    assert_same_scores(first, second)
    assert first.estimator.threshold == pytest.approx(second.estimator.threshold, abs=1e-12)


@pytest.mark.parametrize(
    "fit, params",
    [
        (p3_minibatch_kmeans, P3_MODEL_PARAMS["minibatch"]),
        (p3_gaussian_mixture, P3_MODEL_PARAMS["gmm"]),
    ],
    ids=["minibatch", "gmm"],
)
def test_p3_clustering_is_reproducible(fit, params):
    train = make_profiles()
    first, second = fit(train, None, params), fit(train, None, params)
    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(first.assign(train), second.assign(train))


def test_different_data_gives_different_model():
    # Control: la prueba de reproducibilidad no pasa por accidente (el modelo sí usa los datos).
    first = p1_hgb(make_parts(P1_FEATURES, "departures", seed=0), P1_MODEL_PARAMS["hgb"])
    second = p1_hgb(make_parts(P1_FEATURES, "departures", seed=1), P1_MODEL_PARAMS["hgb"])
    assert not np.array_equal(first.scores["test"], second.scores["test"])
