"""Gráficas y resumen de las pruebas de carga (make load-report).

Lee reports/carga/<escenario>/ (CSV de Locust y recursos.csv) y escribe las figuras en
reports/figures/carga/ y el resumen para el análisis FinOps en reports/carga/resultados.md.
"""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import typer

from mibici.config import FIGURES_DIR, REPORTS_DIR
from mibici.plots import PALETTE, _save

LOAD_DIR = REPORTS_DIR / "carga"
LOAD_FIGURES_DIR = FIGURES_DIR / "carga"
SCENARIOS = ["base", "nominal", "estres"]
API = "mibici-api"
# Latencia máxima aceptable para definir la capacidad: el p95 que tolera el equipo de rebalanceo.
SLO_P95_MS = 300
# Un escalón cuenta si la carga se sostuvo al menos este tiempo (descarta los segundos de transición).
MIN_STEP_SECONDS = 10

app = typer.Typer(add_completion=False)


def prepare_history(h: pd.DataFrame) -> pd.DataFrame:
    """Serie por segundo de Locust (fila Aggregated), con el tiempo transcurrido en segundos."""
    h = h[(h["Name"] == "Aggregated") & (h["User Count"] > 0)].copy()
    h["t"] = (h["Timestamp"] - h["Timestamp"].min()).astype(float)
    total = h["Total Request Count"].clip(lower=1)
    h["error_pct"] = 100 * h["Total Failure Count"] / total
    return h.reset_index(drop=True)


def prepare_resources(r: pd.DataFrame, t0: float) -> pd.DataFrame:
    """Muestras de docker stats con el tiempo relativo al inicio de la corrida de Locust."""
    r = r.copy()
    r["t"] = r["timestamp"] - t0
    return r


def summarize(stats: pd.DataFrame, resources: pd.DataFrame) -> dict:
    """Totales de una corrida: peticiones, errores, RPS, percentiles y recursos de la API."""
    agg = stats[stats["Name"] == "Aggregated"].iloc[0]
    api = resources[resources["container"] == API]
    requests = int(agg["Request Count"])
    failures = int(agg["Failure Count"])
    return {
        "requests": requests,
        "failures": failures,
        "error_pct": 100 * failures / requests if requests else 0.0,
        "rps": float(agg["Requests/s"]),
        "p50": float(agg["50%"]),
        "p95": float(agg["95%"]),
        "p99": float(agg["99%"]),
        "cpu_avg": float(api["cpu_pct"].mean()),
        "cpu_max": float(api["cpu_pct"].max()),
        "mem_max": float(api["mem_mib"].max()),
        "mem_limit": float(api["mem_limit_mib"].max()),
    }


def steps_table(history: pd.DataFrame, resources: pd.DataFrame) -> pd.DataFrame:
    """Un renglón por nivel de usuarios sostenido: RPS, latencia, errores y CPU de la API."""
    api = resources[resources["container"] == API].sort_values("t")
    h = pd.merge_asof(history.sort_values("t"), api[["t", "cpu_pct", "mem_mib"]], on="t")
    steps = h.groupby("User Count").agg(
        seconds=("t", "count"),
        rps=("Requests/s", "mean"),
        p50=("50%", "median"),
        p95=("95%", "median"),
        error_pct=("error_pct", "last"),
        cpu_pct=("cpu_pct", "mean"),
        mem_mib=("mem_mib", "max"),
    )
    steps = steps[steps["seconds"] >= MIN_STEP_SECONDS].drop(columns="seconds")
    return steps.rename_axis("users").reset_index()


def capacity(steps: pd.DataFrame, slo_p95_ms: float = SLO_P95_MS) -> pd.Series | None:
    """El escalón con más RPS cuyo p95 cumple el SLO."""
    ok = steps[steps["p95"] <= slo_p95_ms]
    return None if ok.empty else ok.loc[ok["rps"].idxmax()]


def plot_series(
    histories: dict[str, pd.DataFrame], summaries: dict[str, dict], path: Path
) -> Path:
    """RPS, latencia y tasa de error contra el tiempo, una columna por escenario."""
    fig, axes = plt.subplots(3, len(histories), figsize=(5 * len(histories), 9), sharex="col")
    axes = axes.reshape(3, -1)
    for col, (name, h) in enumerate(histories.items()):
        ax_rps, ax_lat, ax_err = axes[:, col]
        ax_rps.plot(h["t"], h["Requests/s"], color=PALETTE[0], label="RPS")
        ax_users = ax_rps.twinx()
        ax_users.plot(h["t"], h["User Count"], color=PALETTE[2], ls=":", label="Usuarios")
        ax_users.set_ylabel("Usuarios")
        ax_rps.set_title(name)
        ax_lat.plot(h["t"], h["50%"], color=PALETTE[3], label="p50")
        ax_lat.plot(h["t"], h["95%"], color=PALETTE[1], label="p95")
        ax_lat.axhline(SLO_P95_MS, color="grey", ls="--", lw=1, label=f"SLO p95 {SLO_P95_MS} ms")
        ax_err.plot(h["t"], h["error_pct"], color=PALETTE[7])
        ax_err.set_ylim(bottom=0, top=max(1.0, h["error_pct"].max() * 1.2))
        if h["error_pct"].max() == 0:
            total = summaries[name]["requests"]
            ax_err.text(
                0.5,
                0.5,
                f"0 errores en {total:,} peticiones",
                ha="center",
                transform=ax_err.transAxes,
                color=PALETTE[7],
            )
        ax_err.set_xlabel("Segundos")
        for ax in (ax_rps, ax_lat, ax_err):
            ax.grid(alpha=0.3)
        if col == 0:
            ax_rps.set_ylabel("Peticiones por segundo")
            ax_lat.set_ylabel("Latencia (ms)")
            ax_err.set_ylabel("Errores acumulados (%)")
            lines = ax_rps.get_lines() + ax_users.get_lines()
            ax_rps.legend(lines, [line.get_label() for line in lines], loc="lower right")
            ax_lat.legend(loc="center right")
    return _save(fig, path)


def plot_saturation(steps: pd.DataFrame, cap: pd.Series | None, path: Path) -> Path:
    """Prueba de estrés: RPS y latencia por nivel de usuarios; marca la capacidad con el SLO."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(steps["users"], steps["p50"], "o-", color=PALETTE[3], label="p50")
    ax.plot(steps["users"], steps["p95"], "o-", color=PALETTE[1], label="p95")
    ax.axhline(SLO_P95_MS, color="grey", ls="--", lw=1, label=f"SLO p95 {SLO_P95_MS} ms")
    ax.set_xlabel("Usuarios concurrentes")
    ax.set_ylabel("Latencia (ms)")
    ax_rps = ax.twinx()
    ax_rps.plot(steps["users"], steps["rps"], "s-", color=PALETTE[0], label="RPS")
    ax_rps.set_ylabel("Peticiones por segundo")
    ax_rps.set_ylim(bottom=0)
    if cap is not None:
        ax.axvline(cap["users"], color=PALETTE[2], ls=":", lw=1.2)
        ax.annotate(
            f"capacidad: {cap['rps']:.0f} RPS",
            (cap["users"], SLO_P95_MS),
            xytext=(6, 8),
            textcoords="offset points",
        )
    lines = ax.get_lines()[:3] + ax_rps.get_lines()
    ax.legend(lines, [line.get_label() for line in lines], loc="upper left")
    ax.set_title("Prueba de estrés por escalones: saturación de la API")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_resources(resources: dict[str, pd.DataFrame], path: Path) -> Path:
    """CPU y memoria del contenedor de la API contra el tiempo, por escenario."""
    fig, (ax_cpu, ax_mem) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for color, (name, r) in zip(PALETTE, resources.items()):
        api = r[r["container"] == API]
        ax_cpu.plot(api["t"], api["cpu_pct"], color=color, label=name)
        ax_mem.plot(api["t"], api["mem_mib"], color=color, label=name)
    limit = max(r.loc[r["container"] == API, "mem_limit_mib"].max() for r in resources.values())
    ax_cpu.set_ylabel("CPU de la API (%; 100 = 1 vCPU)")
    ax_mem.set_ylabel("Memoria de la API (MiB)")
    ax_mem.set_title(f"Límite de memoria del contenedor: {limit:.0f} MiB", fontsize=9)
    ax_mem.set_ylim(bottom=0)
    ax_mem.set_xlabel("Segundos")
    for ax in (ax_cpu, ax_mem):
        ax.grid(alpha=0.3)
        ax.legend()
    ax_cpu.set_title("Recursos de la API durante la carga (docker stats)")
    return _save(fig, path)


def workers_of(name: str) -> int:
    """Workers de uvicorn de una corrida: 'estres_2workers' -> 2; sin sufijo, el default (1)."""
    match = re.search(r"(\d+)workers", name)
    return int(match.group(1)) if match else 1


def plot_workers(steps_by_run: dict[str, pd.DataFrame], path: Path) -> Path:
    """Estrés con 1 y más workers: RPS y p95 por nivel de usuarios en el mismo límite de CPU."""
    fig, (ax_rps, ax_p95) = plt.subplots(1, 2, figsize=(12, 4.5))
    for color, (name, steps) in zip(PALETTE, steps_by_run.items()):
        label = f"{workers_of(name)} worker(s)"
        ax_rps.plot(steps["users"], steps["rps"], "o-", color=color, label=label)
        ax_p95.plot(steps["users"], steps["p95"], "o-", color=color, label=label)
    ax_p95.axhline(SLO_P95_MS, color="grey", ls="--", lw=1, label=f"SLO p95 {SLO_P95_MS} ms")
    ax_rps.set_ylabel("Peticiones por segundo")
    ax_p95.set_ylabel("Latencia p95 (ms)")
    ax_rps.set_title("Throughput")
    ax_p95.set_title("Latencia")
    for ax in (ax_rps, ax_p95):
        ax.set_xlabel("Usuarios concurrentes")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Workers de uvicorn con el mismo límite de 2 vCPU")
    return _save(fig, path)


def to_markdown(
    summaries: dict[str, dict],
    steps_by_run: dict[str, pd.DataFrame],
    caps: dict[str, pd.Series | None],
) -> str:
    cpus = float(os.environ.get("API_CPUS", "2"))
    limit = next(iter(summaries.values()))["mem_limit"]
    lines = [
        "# Resultados de las pruebas de carga",
        "",
        (
            f"Generado el {datetime.now():%Y-%m-%d %H:%M} con `make load-report`. "
            f"Contenedor de la API con límite de {cpus:g} vCPU y {limit:.0f} MiB "
            "(docker-compose.yml); escenario de tráfico en `load/locustfile.py`."
        ),
        "",
        "## Resumen por escenario",
        "",
        (
            "| Escenario | Workers | Peticiones | Errores (%) | RPS | p50 (ms) | p95 (ms) "
            "| p99 (ms) | CPU API prom. (%) | CPU API máx. (%) | Memoria API máx. (MiB) |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in summaries.items():
        lines.append(
            f"| {name} | {workers_of(name)} | {s['requests']} | {s['error_pct']:.2f} "
            f"| {s['rps']:.1f} | {s['p50']:.0f} | {s['p95']:.0f} | {s['p99']:.0f} "
            f"| {s['cpu_avg']:.0f} | {s['cpu_max']:.0f} | {s['mem_max']:.0f} |"
        )
    lines += [
        "",
        (
            "CPU en porcentaje de un vCPU (100 % = un núcleo completo). En los escenarios de estrés "
            "el promedio mezcla todos los escalones; la tabla por escalón es la que importa."
        ),
    ]
    for name, steps in steps_by_run.items():
        lines += [
            "",
            f"## Estrés por escalones: {workers_of(name)} worker(s)",
            "",
            "| Usuarios | RPS | p50 (ms) | p95 (ms) | Errores (%) | CPU API (%) | Memoria API (MiB) |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, s in steps.iterrows():
            lines.append(
                f"| {s['users']:.0f} | {s['rps']:.1f} | {s['p50']:.0f} | {s['p95']:.0f} "
                f"| {s['error_pct']:.2f} | {s['cpu_pct']:.0f} | {s['mem_mib']:.0f} |"
            )
    lines += [
        "",
        "## Capacidad (insumo para FinOps)",
        "",
        f"Capacidad: el escalón con más RPS cuyo p95 <= {SLO_P95_MS} ms.",
        "",
        (
            "| Workers | Capacidad (peticiones/s) | Usuarios | p95 (ms) | Peticiones/hora "
            "| Peticiones/s por vCPU | CPU API (%) | Memoria API (MiB) |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, cap in caps.items():
        if cap is None:
            lines.append(f"| {workers_of(name)} | ningún escalón cumplió el SLO | | | | | | |")
            continue
        lines.append(
            f"| {workers_of(name)} | **{cap['rps']:.0f}** | {cap['users']:.0f} | {cap['p95']:.0f} "
            f"| {cap['rps'] * 3600:,.0f} | {cap['rps'] / cpus:.0f} | {cap['cpu_pct']:.0f} "
            f"| {cap['mem_mib']:.0f} |"
        )
    ok = {name: cap for name, cap in caps.items() if cap is not None}
    if len(ok) > 1:
        (base_name, base), *others = ok.items()
        for name, cap in others:
            lines += [
                "",
                (
                    f"Con {workers_of(name)} workers la capacidad cambia "
                    f"{100 * (cap['rps'] / base['rps'] - 1):+.0f} % respecto a "
                    f"{workers_of(base_name)} worker, con el mismo límite de {cpus:g} vCPU."
                ),
            ]
    lines += [
        "",
        "## Figuras",
        "",
        "- `reports/figures/carga/carga_series.png`: RPS, latencia y errores por escenario.",
        "- `reports/figures/carga/carga_saturacion.png`: curva de saturación del estrés.",
        "- `reports/figures/carga/carga_recursos.png`: CPU y memoria de la API.",
    ]
    if len(steps_by_run) > 1:
        lines.append("- `reports/figures/carga/carga_workers.png`: estrés con 1 y 2 workers.")
    lines.append("")
    return "\n".join(lines)


def discover(load_dir: Path) -> list[str]:
    """Corridas en reports/carga/: primero base, nominal y estres; luego las variantes."""
    found = {p.parent.name for p in load_dir.glob("*/locust_stats.csv")}

    def order(name: str) -> tuple[int, int, str]:
        prefix = name.split("_")[0]
        rank = SCENARIOS.index(prefix) if prefix in SCENARIOS else len(SCENARIOS)
        return (name not in SCENARIOS, rank, name)

    return sorted(found, key=order)


@app.command()
def main(load_dir: Path = LOAD_DIR, figures_dir: Path = LOAD_FIGURES_DIR):
    histories, resources, summaries = {}, {}, {}
    for name in discover(load_dir):
        folder = load_dir / name
        h = prepare_history(pd.read_csv(folder / "locust_stats_history.csv"))
        r = prepare_resources(pd.read_csv(folder / "recursos.csv"), h["Timestamp"].min())
        histories[name], resources[name] = h, r
        summaries[name] = summarize(pd.read_csv(folder / "locust_stats.csv"), r)
    if "estres" not in histories:
        raise typer.BadParameter(
            f"falta {load_dir / 'estres'}: corre make load-test ESCENARIO=estres"
        )

    stress = [name for name in histories if name.startswith("estres")]
    steps_by_run = {name: steps_table(histories[name], resources[name]) for name in stress}
    caps = {name: capacity(steps) for name, steps in steps_by_run.items()}
    main_runs = [name for name in SCENARIOS if name in histories]
    plot_series({n: histories[n] for n in main_runs}, summaries, figures_dir / "carga_series.png")
    plot_saturation(steps_by_run["estres"], caps["estres"], figures_dir / "carga_saturacion.png")
    plot_resources({n: resources[n] for n in main_runs}, figures_dir / "carga_recursos.png")
    if len(steps_by_run) > 1:
        plot_workers(steps_by_run, figures_dir / "carga_workers.png")
    (load_dir / "resultados.md").write_text(to_markdown(summaries, steps_by_run, caps))
    typer.echo(f"Resumen en {load_dir / 'resultados.md'}; figuras en {figures_dir}")


if __name__ == "__main__":
    app()
