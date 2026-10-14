from datetime import date

import numpy as np
import pandas as pd
import pytest

from mibici.features import add_lag_features, assign_split, easter_sunday, risk_label


def test_easter_sunday_known_dates():
    assert easter_sunday(2024) == date(2024, 3, 31)
    assert easter_sunday(2025) == date(2025, 4, 20)
    assert easter_sunday(2026) == date(2026, 4, 5)


def test_lag_features_only_use_past_values():
    df = pd.DataFrame({"station_id": [1] * 5 + [2] * 5, "y": list(range(5)) + list(range(10, 15))})
    out = add_lag_features(df, "y", lags=[1], windows=[2], min_shift=1)
    # El lag y la media móvil de cada fila solo dependen de periodos anteriores.
    assert np.isnan(out.loc[0, "y_lag1"]) and np.isnan(out.loc[5, "y_lag1"])
    assert out.loc[3, "y_lag1"] == 2
    assert out.loc[3, "y_roll2_mean"] == 1.5  # media de y[1] y y[2]
    assert out.loc[6, "y_lag1"] == 10  # no se mezclan estaciones


def test_lag_smaller_than_horizon_is_rejected():
    df = pd.DataFrame({"station_id": [1, 1], "y": [1, 2]})
    with pytest.raises(ValueError):
        add_lag_features(df, "y", lags=[1], windows=[], min_shift=6)


def test_temporal_split_order():
    s = assign_split(pd.Series(pd.to_datetime(["2025-12-31", "2026-01-01", "2026-05-01"])))
    assert s.tolist() == ["train", "valid", "test"]


def test_risk_label_uses_half_capacity():
    labels = risk_label(pd.Series([-8, -7, 0, 8]), pd.Series([15, 15, 15, 15]))
    assert labels["risk_empty"].tolist() == [1, 0, 0, 0]
    assert labels["risk_full"].tolist() == [0, 0, 0, 1]
