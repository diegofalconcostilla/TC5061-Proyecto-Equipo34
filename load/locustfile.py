"""Pruebas de carga de la API de MiBici con Locust.

El tráfico sigue la estrategia de despliegue del README: P2 es la consulta en tiempo real del
equipo de rebalanceo; P1 y P3 se consultan menos, y P1 también desde un tablero que pide todas
las estaciones en una sola petición.
"""

import json
from pathlib import Path

from docker_stats import StatsRecorder
from locust import HttpUser, between, events, task
import requests

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "api"
DASHBOARD_STATIONS = 480
# Con varios workers de uvicorn cada uno carga sus propios modelos; varias rondas los calientan a todos.
WARMUP_ROUNDS = 5


def load_payload(name: str) -> dict:
    return json.loads((EXAMPLES / f"{name}.json").read_text())


PAYLOADS = {q: load_payload(q) for q in ("p1", "p2", "p3")}
INVALID = load_payload("p1_invalido")
_p1_records = PAYLOADS["p1"]["records"]
DASHBOARD = {
    "records": (_p1_records * (DASHBOARD_STATIONS // len(_p1_records) + 1))[:DASHBOARD_STATIONS]
}


@events.init_command_line_parser.add_listener
def add_options(parser):
    parser.add_argument(
        "--recursos-csv", default="", help="CSV de CPU y memoria (vacío: no medir)"
    )
    parser.add_argument("--contenedores", default="mibici-api,mibici-mlflow")
    parser.add_argument("--motor", default="docker", choices=["docker", "podman"])


@events.test_start.add_listener
def warm_up(environment, **_):
    """Carga los modelos antes de medir: la primera predicción los descarga de MLflow (~1 s)."""
    # environment.host solo trae valor si se pasa --host o se escribe en la UI.
    host = environment.host or UsuarioAPI.host
    for _ in range(WARMUP_ROUNDS):
        for q, payload in PAYLOADS.items():
            requests.post(f"{host}/predict/{q}", json=payload, timeout=60)


_recorder: StatsRecorder | None = None


@events.test_start.add_listener
def start_stats(environment, **_):
    global _recorder
    options = environment.parsed_options
    if options and options.recursos_csv:
        Path(options.recursos_csv).parent.mkdir(parents=True, exist_ok=True)
        _recorder = StatsRecorder(
            options.recursos_csv, options.contenedores.split(","), options.motor
        )
        _recorder.start()


@events.test_stop.add_listener
def stop_stats(environment, **_):
    global _recorder
    if _recorder:
        _recorder.stop()
        _recorder = None


class UsuarioAPI(HttpUser):
    host = "http://localhost:8000"
    wait_time = between(0.5, 2)

    def predict(self, question: str, payload: dict, name: str | None = None):
        with self.client.post(
            f"/predict/{question}", json=payload, name=name, catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            body = response.json()
            expected = len(payload["records"])
            if body.get("total_predictions") != expected:
                response.failure(f"{body.get('total_predictions')} predicciones de {expected}")
            elif body.get("source") != "registry":
                response.failure(f"modelo de origen {body.get('source')}")

    @task(6)
    def riesgo_p2(self):
        self.predict("p2", PAYLOADS["p2"])

    @task(2)
    def demanda_p1(self):
        self.predict("p1", PAYLOADS["p1"])

    @task(1)
    def grupo_p3(self):
        self.predict("p3", PAYLOADS["p3"])

    @task(1)
    def tablero_p1(self):
        self.predict("p1", DASHBOARD, name=f"/predict/p1 (tablero {DASHBOARD_STATIONS})")

    @task(1)
    def peticion_invalida(self):
        """Un registro fuera de rango: la API debe rechazarlo con 422 sin llegar al modelo."""
        with self.client.post(
            "/predict/p1", json=INVALID, name="/predict/p1 (inválida)", catch_response=True
        ) as response:
            if response.status_code == 422:
                response.success()
            else:
                response.failure(f"se esperaba 422 y llegó {response.status_code}")
