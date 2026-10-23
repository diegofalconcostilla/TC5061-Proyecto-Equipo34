"""Piezas compartidas por los modelos de P1, P2 y P3."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def split_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {s: df[df["split"] == s] for s in ("train", "valid", "test")}


class ModelSpec(NamedTuple):
    """Entrada de la tabla de modelos de una pregunta (``P1_MODELS``, etc.).

    ``fit`` recibe los datos de la pregunta y los hiperparámetros ya resueltos (defaults de
    ``config.py`` más los cambios de ``-P params``). ``family`` y ``stage`` se registran como
    etiquetas ``model_family`` y ``stage_of_work``.
    """

    fit: Callable
    family: str
    stage: str
    defaults: dict | None = None


def get_spec(models: dict[str, ModelSpec], question: str, role: str) -> ModelSpec:
    if role not in models:
        raise ValueError(f"{question} no tiene el modelo {role!r}; opciones: {', '.join(models)}")
    spec = models[role]
    return spec._replace(defaults=spec.defaults or {})


def run_name_with_changes(run_name: str, hp: dict, defaults: dict) -> str:
    """Agrega al nombre del run los hiperparámetros que difieren del default."""
    changes = [f"{k}={v}" for k, v in hp.items() if defaults.get(k) != v]
    return f"{run_name}[{','.join(changes)}]" if changes else run_name


class ModelResult(NamedTuple):
    """Lo que devuelve cada modelo de la tabla de una pregunta."""

    run_name: str
    description: str
    params: dict
    scores: dict[str, np.ndarray]  # predicción (P1) o score continuo (P2), en valid y test
    labels: dict[str, np.ndarray] | None = None  # predicción binaria (solo P2)
    estimator: object | None = None  # modelo entrenado que se registra; None en reglas
    features: list[str] | None = None  # columnas de entrada de ``estimator``


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


class RunSummary(NamedTuple):
    """Lo que un ``run_pX`` registró; lo usa la búsqueda para comparar combinaciones."""

    run_id: str
    run_name: str
    hyperparameters: dict
    metrics: dict[str, float]
