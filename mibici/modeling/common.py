"""Piezas compartidas por los modelos de P1, P2 y P3."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


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
