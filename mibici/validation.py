"""Validación de datos con DeepChecks: integridad de cada conjunto y drift entre train y test.

    python -m mibici.validation               # P1, P2 y P3
    python -m mibici.validation --question p2

Se validan las mismas columnas que usan los modelos. Si falla un chequeo de integridad el
comando termina con error; el drift solo se reporta, porque la partición es temporal
(2024-2025 contra 2026) y un cambio de distribución es esperado y se documenta.
"""

from __future__ import annotations

import warnings

from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.checks import (
    ConflictingLabels,
    DataDuplicates,
    DatasetsSizeComparison,
    FeatureDrift,
    IsSingleValue,
    LabelDrift,
    MixedDataTypes,
    MixedNulls,
    NewCategoryTrainTest,
    PercentOfNulls,
)
from loguru import logger
import pandas as pd
import typer

from mibici.config import (
    P1_DATASET_PATH,
    P2_DATASET_PATH,
    P3_DATASET_PATH,
    P3_TEST_DATASET_PATH,
    REPORTS_DIR,
    SEED,
    VALIDATION_SAMPLE_SIZE,
)
from mibici.features import p3_feature_columns
from mibici.modeling.p1 import P1_CATEGORICAL, P1_FEATURES
from mibici.modeling.p2 import P2_CATEGORICAL, P2_FEATURES

VALIDATION_REPORTS_DIR = REPORTS_DIR / "deepchecks"

# Columnas que ya sabemos que son constantes y no son un error de datos: en el periodo de
# entrenamiento de P3 no hay ninguna salida de 2:00 a 4:59 (ver reports/validacion_datos.md).
KNOWN_CONSTANT_COLUMNS = {
    "p3": [f"dep_{day}_h{hour:02d}" for day in ("wd", "we") for hour in (2, 3, 4)],
}

app = typer.Typer(help="Valida los datasets de P1, P2 y P3 con DeepChecks.")


def sample(df: pd.DataFrame) -> pd.DataFrame:
    """Muestra fija de VALIDATION_SAMPLE_SIZE filas (todas si hay menos)."""
    return df.sample(n=min(len(df), VALIDATION_SAMPLE_SIZE), random_state=SEED)


def load_train_test(question: str) -> tuple[Dataset, Dataset]:
    """Datasets de DeepChecks de train y test con las columnas que usan los modelos."""
    if question == "p1":
        df = pd.read_parquet(P1_DATASET_PATH)
        train, test = df[df["split"] == "train"], df[df["split"] == "test"]
        columns, label, categorical = P1_FEATURES + ["departures"], "departures", P1_CATEGORICAL
    elif question == "p2":
        df = pd.read_parquet(P2_DATASET_PATH)
        train, test = df[df["split"] == "train"], df[df["split"] == "test"]
        columns, label, categorical = P2_FEATURES + ["risk"], "risk", P2_CATEGORICAL
    elif question == "p3":
        train, test = pd.read_parquet(P3_DATASET_PATH), pd.read_parquet(P3_TEST_DATASET_PATH)
        columns, label, categorical = p3_feature_columns(train), None, []
    else:
        raise ValueError(f"question debe ser p1, p2 o p3, no {question!r}")
    return (
        Dataset(sample(train[columns]), label=label, cat_features=categorical),
        Dataset(sample(test[columns]), label=label, cat_features=categorical),
    )


def integrity_suite(constant_columns: list[str] | None = None) -> Suite:
    """Calidad de cada conjunto por separado (se corre sobre train y sobre test).

    ``constant_columns``: columnas que se sabe que son constantes; no cuentan como error.
    """
    return Suite(
        "Integridad",
        # Hasta 5 % de nulos por columna (hoy hay 0 %: los lags no dejan huecos).
        PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0.05),
        # Una sola forma de representar un faltante (no mezclar NaN, "", "null"...).
        MixedNulls().add_condition_different_nulls_less_equal_to(1),
        # Cada columna con un solo tipo de dato (no números y textos mezclados).
        MixedDataTypes().add_condition_rare_type_ratio_not_in_range(),
        # Hasta 5 % de filas idénticas (hoy < 1 %: estaciones sin actividad en el mismo bloque).
        DataDuplicates().add_condition_ratio_less_or_equal(0.05),
        # Ninguna columna con un solo valor: no aporta información al modelo.
        IsSingleValue(ignore_columns=constant_columns).add_condition_not_single_value(),
        # Hasta 5 % de filas con features iguales y etiqueta distinta (solo clasificación: P2).
        ConflictingLabels().add_condition_ratio_of_conflicting_labels_less_or_equal(0.05),
    )


def drift_suite() -> Suite:
    """Diferencias entre train y test."""
    return Suite(
        "Train contra test",
        # Drift de cada feature: KS (numéricas) o V de Cramer (categóricas) menor a 0.2.
        FeatureDrift().add_condition_drift_score_less_than(),
        # Drift de la etiqueta (P1 y P2; P3 no tiene etiqueta).
        LabelDrift().add_condition_drift_score_less_than(),
        # Ninguna categoría en test que no exista en train (p. ej., un mes o bloque nuevo).
        NewCategoryTrainTest().add_condition_new_category_ratio_less_or_equal(0),
        # Test con al menos 100 filas para que las métricas signifiquen algo.
        DatasetsSizeComparison().add_condition_test_size_greater_or_equal(100),
    )


def failed_checks(result) -> list[str]:
    return [check.get_header() for check in result.get_not_passed_checks()]


def validate(question: str) -> bool:
    """Corre las dos suites, guarda los HTML y devuelve si pasó la integridad."""
    train, test = load_train_test(question)
    VALIDATION_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    passed = True
    for suite in (integrity_suite(KNOWN_CONSTANT_COLUMNS.get(question)), drift_suite()):
        result = suite.run(train_dataset=train, test_dataset=test)
        path = VALIDATION_REPORTS_DIR / f"{question}_{suite.name.lower().replace(' ', '_')}.html"
        path.unlink(missing_ok=True)  # DeepChecks no sobrescribe: crearía "nombre (1).html"
        result.save_as_html(str(path), as_widget=False)  # HTML normal, no widget de Jupyter
        failed = failed_checks(result)
        if not failed:
            logger.success(f"{question.upper()} {suite.name}: todo pasó ({path.name})")
        elif suite.name == "Integridad":
            passed = False
            logger.error(f"{question.upper()} {suite.name}: falló {', '.join(failed)}")
        else:
            logger.warning(f"{question.upper()} {suite.name}: revisar {', '.join(failed)}")
    return passed


@app.command()
def main(question: str = typer.Option("all", help="p1, p2, p3 o all.")):
    """Valida las preguntas pedidas y termina con error si falla la integridad."""
    warnings.filterwarnings("ignore")  # DeepChecks avisa de cosas internas que no aplican
    questions = ["p1", "p2", "p3"] if question == "all" else [question.lower()]
    results = [validate(q) for q in questions]
    if not all(results):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
