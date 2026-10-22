"""Envoltorios de modelos que se registran en MLflow."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class ThresholdClassifier(ClassifierMixin, BaseEstimator):
    """Clasificador ya entrenado con el umbral de decisión elegido en validación.

    ``predict`` aplica ``threshold`` al score (no el 0.5 por defecto de scikit-learn), así que el
    modelo que se carga del Model Registry decide igual que en el experimento.
    """

    def __init__(self, estimator, threshold: float):
        self.estimator = estimator
        self.threshold = threshold

    @property
    def classes_(self):
        return self.estimator.classes_

    def fit(self, X, y=None):
        return self  # se registra ya entrenado

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(np.int64)
