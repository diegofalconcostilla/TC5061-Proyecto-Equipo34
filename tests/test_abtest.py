"""A/B de P1 con datos sintéticos: segmentos, bootstrap pareado, decisión y reporte."""

import numpy as np
import pandas as pd
import pytest

from mibici.modeling import abtest


def make_data(seed: int = 0) -> pd.DataFrame:
    """Estaciones 1-20 desde 2025 y 21-25 solo en 2026 (nuevas para el champion)."""
    rng = np.random.default_rng(seed)
    rows = []
    for station in range(1, 26):
        start = "2025-12-01" if station <= 20 else "2026-03-01"
        for date in pd.date_range(start, "2026-08-31", freq="7D"):
            rows.append(
                {
                    "station_id": station,
                    "date": date,
                    "departures": float(rng.poisson(20 + station)),
                    "is_weekend": int(date.dayofweek >= 5),
                    "is_holiday": 0,
                }
            )
    return pd.DataFrame(rows)


def test_eval_frame_keeps_only_the_ab_period():
    frame = abtest.eval_frame(make_data())
    dates = pd.to_datetime(frame["date"])

    assert dates.min() >= pd.Timestamp(abtest.AB_EVAL_START)
    assert dates.max() <= pd.Timestamp(abtest.AB_EVAL_END)
    assert set(frame["mes"]) <= {"julio", "agosto"}


def test_stations_without_training_rows_are_new():
    frame = abtest.eval_frame(make_data())
    kind = frame.groupby("station_id")["estacion"].first()

    assert (kind.loc[1:20] == "existente").all()
    assert (kind.loc[21:25] == "nueva").all()


def test_better_challenger_is_promoted():
    frame = abtest.eval_frame(make_data())
    y = frame["departures"].to_numpy()
    rng = np.random.default_rng(1)
    pred_a = y + rng.normal(0, 6, len(y))
    pred_b = y + rng.normal(0, 2, len(y))

    result = abtest.paired_bootstrap(y, pred_a, pred_b, frame["station_id"], n_boot=300)

    assert result["wape"]["ic95_superior"] < 0
    assert result["wape"]["prob_B_mejor"] > 0.99
    assert abtest.decide(result["wape"]) == "promover"


def test_worse_challenger_is_rejected():
    frame = abtest.eval_frame(make_data())
    y = frame["departures"].to_numpy()
    rng = np.random.default_rng(2)
    pred_a = y + rng.normal(0, 2, len(y))
    pred_b = y + rng.normal(0, 6, len(y))

    result = abtest.paired_bootstrap(y, pred_a, pred_b, frame["station_id"], n_boot=300)

    assert result["wape"]["ic95_inferior"] > 0
    assert abtest.decide(result["wape"]) == "rechazar"


def test_identical_versions_keep_the_champion():
    frame = abtest.eval_frame(make_data())
    y = frame["departures"].to_numpy()
    pred = y + np.random.default_rng(3).normal(0, 4, len(y))

    result = abtest.paired_bootstrap(y, pred, pred, frame["station_id"], n_boot=300)

    assert result["wape"]["diferencia"] == 0
    assert abtest.decide(result["wape"]) == "mantener"


def test_bootstrap_point_estimate_matches_the_metrics():
    frame = abtest.eval_frame(make_data())
    y = frame["departures"].to_numpy()
    pred_a, pred_b = y * 1.2, y * 0.9

    result = abtest.paired_bootstrap(y, pred_a, pred_b, frame["station_id"], n_boot=50)
    metrics = abtest.version_metrics(y, pred_a, pred_b)

    assert result["wape"]["diferencia"] == pytest.approx(metrics.loc["wape", "diferencia_B_menos_A"])
    assert metrics.loc["wape", "A"] == pytest.approx(0.2)
    assert metrics.loc["wape", "B"] == pytest.approx(0.1)


def test_segment_metrics_cover_every_group():
    frame = abtest.eval_frame(make_data())
    y = frame["departures"].to_numpy()

    segments = abtest.segment_metrics(frame, y * 1.1, y)

    assert set(segments["segmento"]) == set(abtest.SEGMENTS.values())
    for _, rows in segments.groupby("segmento"):
        assert rows["filas"].sum() == len(frame)
    np.testing.assert_allclose(segments["wape_A"], 0.1)
    np.testing.assert_allclose(segments["cambio_wape_pct"], -100)


def test_win_rate_adds_up_to_one():
    y = np.array([10.0, 10.0, 10.0, 10.0])
    wins = abtest.win_rate(y, np.array([12.0, 9.0, 10.0, 11.0]), np.array([11.0, 7.0, 10.0, 10.0]))

    assert wins == {"B_mejor": 0.5, "A_mejor": 0.25, "empate": 0.25}


def test_markdown_report_contains_the_decision():
    frame = abtest.eval_frame(make_data())
    y = frame["departures"].to_numpy()
    pred_a, pred_b = y * 1.2, y * 1.05
    metrics = abtest.version_metrics(y, pred_a, pred_b)
    bootstrap = abtest.paired_bootstrap(y, pred_a, pred_b, frame["station_id"], n_boot=50)
    segments = abtest.segment_metrics(frame, pred_a, pred_b)
    summary = {
        "A": {"alias": "champion", "version": 1, "run": "a"},
        "B": {"alias": "challenger", "version": 3, "run": "b"},
        "periodo": [abtest.AB_EVAL_START, abtest.AB_EVAL_END],
        "filas": len(frame),
        "estaciones": frame["station_id"].nunique(),
        "remuestreos": 50,
        "bootstrap": bootstrap,
        "estacion_dias_con_menor_error": abtest.win_rate(y, pred_a, pred_b),
        "decision": abtest.decide(bootstrap["wape"]),
    }

    text = abtest.to_markdown(summary, metrics, segments)

    assert "## Decisión" in text
    assert abtest.DECISIONS["promover"] in text
    assert "| Estación | nueva |" in text
