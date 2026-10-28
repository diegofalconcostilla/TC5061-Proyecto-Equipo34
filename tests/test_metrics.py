"""Métricas, umbral de decisión, bloques de P2 y versión de datos: casos normales y de borde."""

import numpy as np
import pandas as pd
import pytest

from mibici.dataset import compute_data_version
from mibici.features import block_start
from mibici.modeling.p1 import regression_metrics
from mibici.modeling.p2 import ThresholdClassifier, best_f1_threshold, classification_metrics
from mibici.modeling.p3 import cluster_metrics


def test_perfect_regression_has_zero_error():
    y = [3, 0, 7, 10]
    m = regression_metrics(y, y, "test")
    assert m["test_mae"] == 0 and m["test_rmse"] == 0 and m["test_wape"] == 0
    assert m["test_r2"] == 1


def test_regression_metrics_known_values():
    m = regression_metrics([2, 4, 6, 8], [3, 3, 7, 7], "valid")
    assert m["valid_mae"] == pytest.approx(1.0)
    assert m["valid_rmse"] == pytest.approx(1.0)
    assert m["valid_wape"] == pytest.approx(4 / 20)  # suma de errores / suma de viajes


def test_wape_with_zero_trips_is_not_finite():
    # Borde: si ninguna estación tuvo viajes, el WAPE no está definido (0 / 0).
    with np.errstate(invalid="ignore", divide="ignore"):
        m = regression_metrics([0, 0], [0, 1], "test")
    assert not np.isfinite(m["test_wape"])


def test_perfect_classification():
    y = np.array([0, 0, 1, 1])
    m = classification_metrics(y, [0.1, 0.2, 0.8, 0.9], y, "test")
    assert m["test_f1"] == 1 and m["test_pr_auc"] == 1 and m["test_roc_auc"] == 1
    assert m["test_positive_rate"] == 0.5


def test_classification_without_positive_predictions_does_not_fail():
    # Borde: el modelo nunca predice riesgo; precisión y F1 valen 0 en lugar de tronar.
    y = np.array([0, 1, 0, 1])
    m = classification_metrics(y, [0.4, 0.3, 0.2, 0.1], np.zeros(4, int), "valid")
    assert m["valid_precision"] == 0 and m["valid_recall"] == 0 and m["valid_f1"] == 0


def test_best_threshold_separates_perfect_scores():
    y = np.array([0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    threshold = best_f1_threshold(y, score)
    assert 0.3 < threshold <= 0.7
    assert ((score >= threshold).astype(int) == y).all()


def test_best_threshold_with_a_single_class_does_not_fail():
    # Borde: un periodo sin ningún bloque en riesgo.
    with np.errstate(invalid="ignore", divide="ignore"):
        threshold = best_f1_threshold(np.zeros(4, int), [0.1, 0.2, 0.3, 0.4])
    assert np.isfinite(threshold)


class FixedProba:
    def predict_proba(self, X):
        p = np.asarray(X, float).ravel()
        return np.column_stack([1 - p, p])


def test_threshold_classifier_uses_its_threshold_not_half():
    clf = ThresholdClassifier(FixedProba(), threshold=0.3)
    assert clf.predict([[0.1], [0.3], [0.45], [0.9]]).tolist() == [0, 1, 1, 1]
    assert clf.predict_proba([[0.3]])[0, 1] == pytest.approx(0.3)


def test_cluster_metrics_on_well_separated_groups():
    x = np.array([[0, 0], [0, 0.1], [0.1, 0], [10, 10], [10, 10.1], [10.1, 10]])
    m = cluster_metrics(x, np.array([0, 0, 0, 1, 1, 1]), "train")
    assert m["train_silhouette"] > 0.9
    assert m["train_davies_bouldin"] < 0.1


def test_cluster_metrics_reject_a_single_group():
    # Borde: con un solo grupo la silueta no está definida y scikit-learn lo rechaza.
    with pytest.raises(ValueError):
        cluster_metrics(np.random.default_rng(0).normal(size=(5, 2)), np.zeros(5, int), "train")


def test_block_start_edges():
    ts = pd.Series(pd.to_datetime(["2026-03-02 05:59", "2026-03-02 06:00", "2026-03-02 23:59"]))
    out = block_start(ts)
    assert pd.isna(out[0])  # antes de las 6:00 queda fuera de los bloques
    assert out[1] == pd.Timestamp("2026-03-02 06:00")
    assert out[2] == pd.Timestamp("2026-03-02 21:00")  # último bloque del día, no el siguiente


ENTRIES = [
    {"file": "2025_01.csv", "raw_sha256": "aaa", "rows_clean": 10},
    {"file": "2025_02.csv", "raw_sha256": "bbb", "rows_clean": 20},
]


def test_data_version_is_stable_and_ignores_order():
    assert compute_data_version(ENTRIES) == compute_data_version(ENTRIES[::-1])
    assert len(compute_data_version(ENTRIES)) == 12


def test_data_version_changes_when_a_file_changes():
    changed = [ENTRIES[0], {**ENTRIES[1], "raw_sha256": "ccc"}]
    assert compute_data_version(changed) != compute_data_version(ENTRIES)
