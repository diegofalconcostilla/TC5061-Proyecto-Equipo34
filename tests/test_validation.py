"""Las suites de DeepChecks detectan los problemas que deben detectar.

Usan tablas sintéticas pequeñas, así que corren sin los datos reales de MiBici.
"""

from deepchecks.tabular import Dataset
import numpy as np
import pandas as pd

from mibici.validation import drift_suite, failed_checks, integrity_suite


def make_table(n: int = 300, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Tabla limpia: una feature numérica, una categórica y una etiqueta numérica."""
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=shift, size=n)
    return pd.DataFrame(
        {
            "x": x,
            "zone": rng.choice(["a", "b", "c"], size=n),
            "y": x + rng.normal(scale=0.1, size=n),
        }
    )


def integrity_failures(df: pd.DataFrame, constant_columns=None) -> list[str]:
    dataset = Dataset(df, label="y", cat_features=["zone"])
    return failed_checks(integrity_suite(constant_columns).run(train_dataset=dataset))


def drift_failures(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    def to_dataset(df):
        return Dataset(df, label="y", cat_features=["zone"])

    result = drift_suite().run(train_dataset=to_dataset(train), test_dataset=to_dataset(test))
    return failed_checks(result)


def test_clean_table_passes_integrity():
    assert integrity_failures(make_table()) == []


def test_nulls_are_detected():
    df = make_table()
    df.loc[:49, "x"] = np.nan  # 1/6 de nulos, arriba del 5 % permitido
    assert "PercentOfNulls" in integrity_failures(df)


def test_duplicates_are_detected():
    df = make_table()
    df = pd.concat([df, df.head(100)], ignore_index=True)  # 25 % de filas repetidas
    assert "Data Duplicates" in integrity_failures(df)


def test_constant_column_is_detected_unless_known():
    df = make_table().assign(constant=0)
    assert "Single Value in Column" in integrity_failures(df)
    assert "Single Value in Column" not in integrity_failures(df, constant_columns=["constant"])


def test_same_distribution_has_no_drift():
    assert drift_failures(make_table(seed=0), make_table(seed=1)) == []


def test_shifted_feature_is_detected_as_drift():
    failures = drift_failures(make_table(seed=0), make_table(shift=3.0, seed=1))
    assert "Feature Drift" in failures
