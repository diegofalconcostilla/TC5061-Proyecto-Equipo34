"""Registra en el Model Registry el modelo de un run y le asigna un alias (``champion``).

    mlflow run . -e register -P question=p1                       # mejor candidato en validación
    mlflow run . -e register -P question=p1 -P run_name=p1-hgb-regressor-lags-calendario
    mlflow run . -e register -P question=p2 -P alias=challenger

El run se elige por nombre o por ``SELECTION_METRIC``, nunca por ``run_id``: los IDs cambian en
cada base de MLflow y los nombres no. Solo compiten runs ``candidate`` o ``search`` con modelo.
"""

from __future__ import annotations

from loguru import logger
import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
import typer

from mibici.config import (
    MLFLOW_EXPERIMENT_NAME,
    REGISTERED_MODEL_DESCRIPTIONS,
    REGISTERED_MODEL_NAMES,
    SELECTION_METRIC,
    TEAM,
)
from mibici.modeling.artifacts import MODEL_NAME
from mibici.modeling.params import cli_text
from mibici.modeling.tracking import git_info, setup_mlflow

app = typer.Typer(help="Registra un modelo en el Model Registry y le asigna un alias.")

ELIGIBLE_STAGES = ("candidate", "search")


def has_model(run: Run) -> bool:
    return bool(run.outputs and run.outputs.model_outputs)


def select_run(client: MlflowClient, question: str, run_name: str = "") -> Run:
    """Run más reciente con ``run_name`` o mejor candidato con modelo según SELECTION_METRIC."""
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"No existe el experimento {MLFLOW_EXPERIMENT_NAME!r}")
    metric, direction = SELECTION_METRIC[question]
    query = f"tags.question = '{question.upper()}' and attributes.status = 'FINISHED'"
    if run_name:
        query += f" and attributes.run_name = '{run_name}'"
        order = "attributes.start_time DESC"  # el más reciente con ese nombre
    else:
        order = f"metrics.{metric} {'ASC' if direction == 'min' else 'DESC'}"  # el mejor
    runs = client.search_runs(
        [experiment.experiment_id], query, order_by=[order], max_results=5000
    )
    for run in runs:
        eligible = run_name or run.data.tags.get("stage_of_work") in ELIGIBLE_STAGES
        if eligible and has_model(run) and metric in run.data.metrics:
            return run
    detail = f"llamado {run_name!r}" if run_name else f"candidate/search con {metric}"
    raise ValueError(f"No hay runs terminados de {question.upper()} con modelo {detail}")


def register(question: str, run_name: str = "", alias: str = "champion") -> int:
    client = MlflowClient()
    run = select_run(client, question, run_name)
    name = REGISTERED_MODEL_NAMES[question]
    metric, direction = SELECTION_METRIC[question]
    tags, params = run.data.tags, run.data.params
    value = run.data.metrics[metric]

    lineage = {
        "source_run_id": run.info.run_id,
        "source_run_name": run.info.run_name,
        "experiment": MLFLOW_EXPERIMENT_NAME,
        "model_family": tags.get("model_family", ""),
        "stage_of_work": tags.get("stage_of_work", ""),
        "git_commit": tags.get("git_commit", ""),
        "git_dirty": tags.get("git_dirty", ""),
        "data_version": tags.get("data_version", ""),
        "train_data_hash": params.get("train_data_hash", ""),
        "selection_metric": f"{metric} ({direction})",
        metric: f"{value:.6g}",
    }
    if tags.get("mlflow.parentRunId"):
        lineage["search_parent_run_id"] = tags["mlflow.parentRunId"]

    # register_model crea el modelo registrado si todavía no existe.
    version = mlflow.register_model(
        f"runs:/{run.info.run_id}/{MODEL_NAME}", name, tags=lineage
    ).version
    client.update_registered_model(name, description=REGISTERED_MODEL_DESCRIPTIONS[question])
    client.set_registered_model_tag(name, "team", TEAM)
    client.set_registered_model_tag(name, "question", question.upper())
    client.update_model_version(
        name,
        version,
        description=(
            f"{run.info.run_name} ({tags.get('model_family', '')}), {metric}={value:.4f}. "
            f"Datos {lineage['data_version']} (train {lineage['train_data_hash']}), "
            f"commit {lineage['git_commit'][:7]}."
        ),
    )
    client.set_registered_model_alias(name, alias, version)

    model = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")
    model.predict(model.input_example)
    logger.success(f"{name} v{version} @{alias} <- {run.info.run_name} ({metric}={value:.4f})")
    return int(version)


@app.command()
def main(
    question: str = typer.Option(..., help="p1, p2 o p3."),
    run_name: str = typer.Option("", help="Nombre del run; vacío = mejor candidato."),
    alias: str = typer.Option("champion", help="Alias de la versión (champion, challenger)."),
):
    """Registra el modelo del run elegido, documenta su linaje y le asigna el alias."""
    question, run_name, alias = question.lower(), cli_text(run_name), cli_text(alias)
    if question not in REGISTERED_MODEL_NAMES:
        raise typer.BadParameter(f"question debe ser p1, p2 o p3, no {question!r}")
    setup_mlflow()
    with mlflow.start_run():
        version = register(question, run_name, alias)
        mlflow.set_tags(
            {
                "mlflow.runName": f"{question}-registro-{alias}",
                "team": TEAM,
                "question": question.upper(),
                "stage_of_work": "register",
                **git_info(),
            }
        )
        mlflow.log_params(
            {"registered_model": REGISTERED_MODEL_NAMES[question], "registered_version": version}
        )


if __name__ == "__main__":
    app()
