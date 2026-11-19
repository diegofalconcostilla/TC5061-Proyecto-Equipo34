"""Pruebas de carga: lectura de docker/podman stats y resumen de los CSV de Locust."""

import pandas as pd
import pytest

from load import docker_stats, report


@pytest.mark.parametrize(
    ("value", "mib"),
    [("201.6MiB", 201.6), ("2GiB", 2048), ("512KiB", 0.5), ("1.5GB", 1.5e9 / 2**20)],
)
def test_to_mib_docker_y_podman(value, mib):
    assert docker_stats.to_mib(value) == pytest.approx(mib)


def test_parse_line():
    row = docker_stats.parse_line("mibici-api;93.5%;201.6MiB / 2GiB", 100.0)
    assert row == {
        "timestamp": 100.0,
        "container": "mibici-api",
        "cpu_pct": 93.5,
        "mem_mib": 201.6,
        "mem_limit_mib": 2048.0,
    }


def make_history(levels: dict[int, tuple[float, float]], seconds: int = 12) -> pd.DataFrame:
    """Serie por segundo de Locust: {usuarios: (RPS, p95)} sostenidos `seconds` cada uno."""
    rows, t, total = [], 1_000, 0
    for users, (rps, p95) in levels.items():
        for _ in range(seconds):
            total += int(rps)
            rows.append(
                {
                    "Timestamp": t,
                    "User Count": users,
                    "Name": "Aggregated",
                    "Requests/s": rps,
                    "50%": p95 / 4,
                    "95%": p95,
                    "Total Request Count": total,
                    "Total Failure Count": 0,
                }
            )
            t += 1
    # Fila final de Locust al detenerse (0 usuarios) y un segundo de transición entre escalones.
    rows.append({**rows[-1], "Timestamp": t, "User Count": 0})
    rows.insert(seconds, {**rows[seconds], "User Count": 30})
    return pd.DataFrame(rows)


def make_resources(history: pd.DataFrame, cpu: float = 80.0) -> pd.DataFrame:
    times = history["Timestamp"].unique()
    return pd.DataFrame(
        {
            "timestamp": times.astype(float),
            "container": report.API,
            "cpu_pct": cpu,
            "mem_mib": 230.0,
            "mem_limit_mib": 2048.0,
        }
    )


def test_steps_descarta_transiciones_y_fila_final():
    h = report.prepare_history(make_history({20: (15, 60), 40: (30, 90)}))
    r = report.prepare_resources(make_resources(h), h["Timestamp"].min())
    steps = report.steps_table(h, r)
    assert steps["users"].tolist() == [20, 40]
    assert steps["rps"].tolist() == [15, 30]
    assert steps["p95"].tolist() == [60, 90]
    assert (steps["cpu_pct"] == 80).all()


def test_capacidad_es_el_mayor_rps_dentro_del_slo():
    steps = pd.DataFrame(
        {"users": [20, 40, 60, 80], "rps": [15, 30, 40, 41], "p95": [60, 90, 250, 900]}
    )
    cap = report.capacity(steps, slo_p95_ms=300)
    assert cap["users"] == 60
    assert report.capacity(steps, slo_p95_ms=10) is None


def test_workers_por_sufijo():
    assert report.workers_of("estres") == 1
    assert report.workers_of("estres_2workers") == 2


def test_discover_ordena_escenarios_y_variantes(tmp_path):
    for name in ["estres_2workers", "nominal_2workers", "nominal", "estres", "base", "vacia"]:
        (tmp_path / name).mkdir()
        if name != "vacia":
            (tmp_path / name / "locust_stats.csv").touch()
    assert report.discover(tmp_path) == [
        "base",
        "nominal",
        "estres",
        "nominal_2workers",
        "estres_2workers",
    ]


def test_summarize():
    stats = pd.DataFrame(
        [
            {"Name": "/predict/p2", "Request Count": 90, "Failure Count": 1},
            {
                "Name": "Aggregated",
                "Request Count": 100,
                "Failure Count": 2,
                "Requests/s": 50.0,
                "50%": 30,
                "95%": 120,
                "99%": 200,
            },
        ]
    )
    resources = pd.DataFrame(
        {
            "container": [report.API, report.API, "mibici-mlflow"],
            "cpu_pct": [50.0, 150.0, 900.0],
            "mem_mib": [200.0, 240.0, 1800.0],
            "mem_limit_mib": [2048.0, 2048.0, 16000.0],
        }
    )
    s = report.summarize(stats, resources)
    assert s["error_pct"] == pytest.approx(2.0)
    assert (s["rps"], s["p95"], s["p99"]) == (50.0, 120, 200)
    assert (s["cpu_avg"], s["cpu_max"], s["mem_max"]) == (100.0, 150.0, 240.0)
