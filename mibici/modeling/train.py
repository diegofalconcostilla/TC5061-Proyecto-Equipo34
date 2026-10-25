"""Entrena un modelo de P1, P2 o P3 y lo registra en el experimento de MLflow.

Se lanza desde el MLflow Project (un run por invocación):

    mlflow run . -e train -P question=p1 -P model=hgb
    mlflow run . -e train -P question=p1 -P model=hgb -P params="learning_rate=0.1,max_iter=300"

o, durante el desarrollo, directamente:

    python -m mibici.modeling.train --question p2 --model logistica --params "C=0.5"

Cada pregunta tiene su tabla de modelos (``P1_MODELS`` en ``p1.py``, etc.) y los
hiperparámetros por defecto viven en ``config.py`` (``P1_MODEL_PARAMS``, etc.).
"""

from __future__ import annotations

import typer

from mibici.modeling.p1 import P1_MODELS, run_p1
from mibici.modeling.p2 import P2_MODELS, run_p2
from mibici.modeling.p3 import P3_MODELS, run_p3
from mibici.modeling.params import cli_text
from mibici.modeling.tracking import STAGES, setup_mlflow

app = typer.Typer(help="Entrena un modelo de MiBici y lo registra en MLflow.")

RUNNERS = {"p1": run_p1, "p2": run_p2, "p3": run_p3}
MODELS = {"p1": P1_MODELS, "p2": P2_MODELS, "p3": P3_MODELS}


@app.command()
def main(
    question: str = typer.Option(..., help="p1, p2 o p3."),
    model: str = typer.Option(
        "baseline",
        help="; ".join(f"{q.upper()}: {', '.join(models)}" for q, models in MODELS.items()),
    ),
    params: str = typer.Option("", help='Cambios a los defaults de config.py: "k=v,k2=v2".'),
    stage: str = typer.Option(
        "", help=f"stage_of_work ({', '.join(STAGES)}); por defecto, el del modelo."
    ),
):
    """Entrena un modelo y registra parámetros, métricas y artefactos en un run."""
    question, params, stage = question.lower(), cli_text(params), cli_text(stage)
    if question not in RUNNERS:
        raise typer.BadParameter(f"question debe ser p1, p2 o p3, no {question!r}")
    setup_mlflow()
    RUNNERS[question](model, params, stage)


if __name__ == "__main__":
    app()
