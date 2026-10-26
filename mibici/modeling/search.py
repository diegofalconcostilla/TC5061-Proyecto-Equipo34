"""Búsqueda de hiperparámetros con nested runs: un run padre y un run hijo por combinación.

    mlflow run . -e search -P question=p1 -P model=hgb \
        -P grid="learning_rate=0.03|0.1;max_leaf_nodes=31|63"

Cada hijo es un entrenamiento normal (mismos artefactos y etiquetas que ``train``) con
``stage_of_work=search``. El padre elige la mejor combinación con ``SELECTION_METRIC`` (siempre
en validación) y registra la tabla ``search/results.csv`` y las métricas del ganador.
"""

from __future__ import annotations

from loguru import logger
import mlflow
import pandas as pd
import typer

from mibici.config import SELECTION_METRIC
from mibici.modeling.artifacts import log_table
from mibici.modeling.common import get_spec
from mibici.modeling.params import cli_text, expand_grid, resolve_params
from mibici.modeling.tracking import setup_mlflow, start_run
from mibici.modeling.train import MODELS, RUNNERS

app = typer.Typer(help="Búsqueda de hiperparámetros con nested runs en MLflow.")


def combine(fixed: str, combo: str) -> str:
    """Une los cambios fijos con los de la combinación; la combinación gana si se repiten."""
    return ",".join(filter(None, (fixed.strip(), combo)))


def run_search(question: str, role: str, grid: str, params: str = "") -> pd.DataFrame:
    spec = get_spec(MODELS[question], question, role)
    combos = [combine(params, c) for c in expand_grid(grid)]
    for combo in combos:  # validar toda la malla antes de entrenar
        resolve_params(spec.defaults, combo)
    metric, direction = SELECTION_METRIC[question]
    run_name = f"{question}-{role}-busqueda"
    description = (
        f"Búsqueda de {len(combos)} combinaciones de {role} ({question.upper()}); "
        f"se elige por {metric} ({direction}). Malla: {grid}"
    )

    with start_run(question.upper(), role, spec.family, "search", run_name, description):
        children = [RUNNERS[question](role, combo, "search", nested=True) for combo in combos]
        pick = min if direction == "min" else max
        best = pick(children, key=lambda c: c.metrics[metric])
        results = pd.DataFrame(
            {"run_name": c.run_name, "run_id": c.run_id, **c.hyperparameters, **c.metrics}
            for c in children
        ).sort_values(metric, ascending=direction == "min")

        mlflow.log_params(
            {
                "search_grid": grid,
                "search_fixed_params": params,
                "n_combinations": len(combos),
                "selection_metric": f"{metric} ({direction})",
            }
        )
        mlflow.log_params({f"best_{k}": v for k, v in best.hyperparameters.items()})
        mlflow.log_metrics(best.metrics)
        mlflow.set_tags({"best_run_id": best.run_id, "best_run_name": best.run_name})
        log_table(results, "results.csv", folder="search")
    logger.success(f"{run_name}: mejor {best.run_name} con {metric}={best.metrics[metric]:.4f}")
    return results


@app.command()
def main(
    question: str = typer.Option(..., help="p1, p2 o p3."),
    model: str = typer.Option(..., help="Modelo con hiperparámetros (p. ej. hgb, kmeans)."),
    grid: str = typer.Option(..., help='Malla: "k=v1|v2;k2=v3|v4".'),
    params: str = typer.Option("", help='Cambios fijos para todas las combinaciones: "k=v".'),
):
    """Entrena cada combinación de la malla como run hijo y resume en el run padre."""
    question, grid, params = question.lower(), cli_text(grid), cli_text(params)
    if question not in RUNNERS:
        raise typer.BadParameter(f"question debe ser p1, p2 o p3, no {question!r}")
    setup_mlflow()
    run_search(question, model, grid, params)


if __name__ == "__main__":
    app()
