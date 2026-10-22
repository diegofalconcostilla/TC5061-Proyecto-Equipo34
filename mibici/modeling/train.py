"""Experimento base del proyecto MiBici registrado en MLflow Tracking.

Uso desde la terminal:

    python -m mibici.modeling.train                         # todos los modelos de P1-P3
    python -m mibici.modeling.train --question p2 --model candidate
    python -m mibici.modeling.train --question p1 --model poisson

Cada pregunta tiene una tabla de modelos (``P1_MODELS``, ``P2_MODELS``, ``P3_MODELS``): de
baselines sencillos (reglas, promedios, KMeans geográfico) a modelos lineales y a los candidatos,
para medir cuánto aporta cada nivel de complejidad.

Cada run registra los parámetros necesarios para reproducirlo (rango de fechas, versión de los
datos, cortes de partición, semilla, modelo e hiperparámetros), métricas con prefijo del
conjunto en que se calculan (``valid_``, ``test_``, ``train_``), al menos un artefacto y
etiquetas de equipo, pregunta y commit de Git. No se registran datos personales.
"""

from __future__ import annotations

from loguru import logger
import typer

from mibici.modeling.p1 import P1_MODELS, run_p1
from mibici.modeling.p2 import P2_MODELS, run_p2
from mibici.modeling.p3 import P3_MODELS, run_p3
from mibici.modeling.tracking import setup_mlflow

app = typer.Typer(help="Experimento base de MiBici con MLflow Tracking.")


RUNNERS = {"p1": run_p1, "p2": run_p2, "p3": run_p3}
ROLES = {"p1": list(P1_MODELS), "p2": list(P2_MODELS), "p3": list(P3_MODELS)}


@app.command()
def main(
    question: str = typer.Option("all", help="p1, p2, p3 o all."),
    model: str = typer.Option(
        "all",
        help="Modelo a correr o all. P1: "
        + ", ".join(ROLES["p1"])
        + ". P2: "
        + ", ".join(ROLES["p2"])
        + ". P3: "
        + ", ".join(ROLES["p3"])
        + ".",
    ),
):
    """Entrena y registra en MLflow los modelos del experimento base."""
    questions = list(RUNNERS) if question == "all" else [question.lower()]
    setup_mlflow()
    for q in questions:
        if model != "all" and model not in ROLES[q]:
            if question == "all":
                continue
            raise typer.BadParameter(
                f"{q} no tiene el modelo {model!r}; opciones: {', '.join(ROLES[q])}"
            )
        roles = ROLES[q] if model == "all" else [model]
        for role in roles:
            logger.info(f"Ejecutando {q.upper()} - {role}")
            RUNNERS[q](role)


if __name__ == "__main__":
    app()
