"""Registra CPU y memoria de los contenedores con `docker stats` o `podman stats`.

Uso independiente (Ctrl+C para terminar; CONTAINER=podman para usar podman):
    python load/docker_stats.py reports/carga/recursos.csv mibici-api mibici-mlflow

Desde Locust, el locustfile lo arranca y detiene con --recursos-csv.
"""

import csv
import os
import re
import subprocess  # nosec B404
import sys
import threading
import time

FORMAT = "{{.Name}};{{.CPUPerc}};{{.MemUsage}}"
COLUMNS = ["timestamp", "container", "cpu_pct", "mem_mib", "mem_limit_mib"]
# docker reporta KiB/MiB/GiB; podman, kB/MB/GB.
UNITS = {
    "b": 1 / 2**20,
    "kib": 1 / 2**10,
    "mib": 1,
    "gib": 2**10,
    "kb": 1e3 / 2**20,
    "mb": 1e6 / 2**20,
    "gb": 1e9 / 2**20,
}


def to_mib(value: str) -> float:
    match = re.fullmatch(r"([\d.]+)\s*([a-zA-Z]+)", value.strip())
    if not match:
        raise ValueError(f"memoria no reconocida: {value!r}")
    number, unit = match.groups()
    return float(number) * UNITS[unit.lower()]


def parse_line(line: str, timestamp: float) -> dict:
    """'mibici-api;93.5%;201.6MiB / 2GiB' -> fila con CPU en % y memoria en MiB."""
    name, cpu, mem = line.split(";")
    used, limit = mem.split("/")
    return {
        "timestamp": round(timestamp, 2),
        "container": name,
        "cpu_pct": float(cpu.strip().rstrip("%")),
        "mem_mib": round(to_mib(used), 1),
        "mem_limit_mib": round(to_mib(limit), 1),
    }


def sample(engine: str, containers: list[str]) -> list[dict]:
    """Una muestra por contenedor. `stats --no-stream` tarda ~1-2 s: mide la CPU en ese lapso."""
    # Lista de argumentos sin shell; el motor solo puede ser docker o podman (opción --motor).
    out = subprocess.run(  # nosec B603 B607
        [engine, "stats", "--no-stream", "--format", FORMAT, *containers],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    now = time.time()
    return [parse_line(line, now) for line in out.splitlines() if line.strip()]


class StatsRecorder:
    """Muestrea en un hilo aparte y escribe cada fila al CSV en cuanto llega."""

    def __init__(self, path, containers: list[str], engine: str = "docker"):
        self.path = path
        self.containers = containers
        self.engine = engine
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self):
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            while not self._stop.is_set():
                writer.writerows(sample(self.engine, self.containers))
                f.flush()


if __name__ == "__main__":
    path, *names = sys.argv[1:]
    recorder = StatsRecorder(path, names, os.environ.get("CONTAINER", "docker"))
    recorder.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        recorder.stop()
