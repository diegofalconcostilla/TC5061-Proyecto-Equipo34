"""A/B testing de P1: dos versiones registradas sobre el mismo periodo que ninguna vio.

    python -m mibici.modeling.abtest                    # champion contra challenger
    python -m mibici.modeling.abtest --promote          # además mueve @champion si gana B
    mlflow run . -e abtest --experiment-name mibici-equipo34

Las dos versiones predicen las mismas estación-días de ``AB_EVAL_START`` a ``AB_EVAL_END``:
ninguna los usó para entrenar ni para elegir hiperparámetros, así que la comparación es pareada
y justa. La métrica de decisión es el WAPE, la misma con la que se eligió el champion de P1.

Los días de una estación están correlacionados, así que el intervalo de la diferencia (B - A) se
calcula con bootstrap por estación: cada remuestreo toma estaciones completas con reemplazo.
B se promueve solo si todo el intervalo del 95 % queda por debajo de 0.

Escribe en ``reports/monitoreo/`` ``p1_ab_test.json``, ``p1_ab_test.md`` (tablas para el reporte)
y ``p1_ab_wape_diario.png``, y registra un run ``abtest`` en MLflow con lo mismo.
"""

from __future__ import annotations

import json

from loguru import logger
import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd
import typer

from mibici import plots
from mibici.config import (
    AB_BOOTSTRAP_SAMPLES,
    AB_EVAL_END,
    AB_EVAL_START,
    MONITORING_REPORTS_DIR,
    P1_DATASET_PATH,
    REGISTERED_MODEL_NAMES,
    SEED,
    VALID_START,
)
from mibici.modeling.p1 import P1_FEATURES, regression_metrics
from mibici.modeling.tracking import setup_mlflow, start_run

app = typer.Typer(help="A/B testing de dos versiones de P1 del Model Registry.")

TARGET = "departures"
SEGMENTS = {"estacion": "Estación", "tipo_dia": "Tipo de día", "mes": "Mes"}
MONTHS = {7: "julio", 8: "agosto"}
DECISIONS = {
    "promover": "Promover B: su WAPE es menor y el intervalo del 95 % no incluye 0.",
    "mantener": "Mantener A: el intervalo del 95 % incluye 0, no hay evidencia de que B sea mejor.",
    "rechazar": "Mantener A: B es peor y el intervalo del 95 % no incluye 0.",
}


def eval_frame(data: pd.DataFrame) -> pd.DataFrame:
    """Periodo del A/B con los segmentos de cada fila.

    Una estación es ``nueva`` si no tiene filas antes de ``VALID_START``, es decir, si no estaba
    en el entrenamiento del champion.
    """
    dates = pd.to_datetime(data["date"])
    known = set(data.loc[dates < pd.Timestamp(VALID_START), "station_id"])
    in_period = (dates >= pd.Timestamp(AB_EVAL_START)) & (dates <= pd.Timestamp(AB_EVAL_END))
    frame = data[in_period].copy()
    frame["estacion"] = np.where(frame["station_id"].isin(known), "existente", "nueva")
    weekend = (frame["is_weekend"] == 1) | (frame["is_holiday"] == 1)
    frame["tipo_dia"] = np.where(weekend, "fin de semana o festivo", "laborable")
    frame["mes"] = pd.to_datetime(frame["date"]).dt.month.map(MONTHS).fillna("otro")
    return frame.reset_index(drop=True)


def version_metrics(y, pred_a, pred_b) -> pd.DataFrame:
    """MAE, RMSE, WAPE y R² de cada versión sobre las mismas filas."""
    a = regression_metrics(y, pred_a, "m")
    b = regression_metrics(y, pred_b, "m")
    table = pd.DataFrame({"A": a, "B": b})
    table.index = [i.removeprefix("m_") for i in table.index]
    table["diferencia_B_menos_A"] = table["B"] - table["A"]
    return table


def paired_bootstrap(
    y, pred_a, pred_b, clusters, n_boot: int = AB_BOOTSTRAP_SAMPLES, seed: int = SEED
) -> dict[str, dict[str, float]]:
    """Diferencia B - A del WAPE y del MAE con intervalo del 95 % por bootstrap de estaciones."""
    y = np.asarray(y, float)
    err_a = np.abs(y - np.asarray(pred_a, float))
    err_b = np.abs(y - np.asarray(pred_b, float))
    sums = (
        pd.DataFrame({"y": y, "a": err_a, "b": err_b, "n": 1.0, "c": np.asarray(clusters)})
        .groupby("c")[["y", "a", "b", "n"]]
        .sum()
        .to_numpy()
    )
    k = len(sums)
    weights = np.random.default_rng(seed).multinomial(k, np.full(k, 1 / k), size=n_boot)
    boot = weights @ sums
    total = sums.sum(axis=0)
    result = {}
    for metric, denominator in (("wape", 0), ("mae", 3)):
        diffs = (boot[:, 2] - boot[:, 1]) / boot[:, denominator]
        low, high = np.percentile(diffs, [2.5, 97.5])
        result[metric] = {
            "diferencia": float((total[2] - total[1]) / total[denominator]),
            "ic95_inferior": float(low),
            "ic95_superior": float(high),
            "prob_B_mejor": float((diffs < 0).mean()),
        }
    return result


def decide(wape_interval: dict[str, float]) -> str:
    if wape_interval["ic95_superior"] < 0:
        return "promover"
    if wape_interval["ic95_inferior"] > 0:
        return "rechazar"
    return "mantener"


def segment_metrics(frame: pd.DataFrame, pred_a, pred_b) -> pd.DataFrame:
    """WAPE y MAE de cada versión por segmento; muestra dónde gana o pierde B."""
    df = frame.assign(err_a=np.abs(frame[TARGET] - pred_a), err_b=np.abs(frame[TARGET] - pred_b))
    rows = []
    for column, label in SEGMENTS.items():
        for group, part in df.groupby(column):
            wape_a = part["err_a"].sum() / part[TARGET].sum()
            wape_b = part["err_b"].sum() / part[TARGET].sum()
            rows.append(
                {
                    "segmento": label,
                    "grupo": group,
                    "filas": len(part),
                    "estaciones": part["station_id"].nunique(),
                    "viajes_promedio": part[TARGET].mean(),
                    "wape_A": wape_a,
                    "wape_B": wape_b,
                    "mae_A": part["err_a"].mean(),
                    "mae_B": part["err_b"].mean(),
                    "cambio_wape_pct": 100 * (wape_b - wape_a) / wape_a,
                }
            )
    return pd.DataFrame(rows)


def win_rate(y, pred_a, pred_b) -> dict[str, float]:
    """Proporción de estación-días en que cada versión tiene menor error absoluto."""
    y = np.asarray(y, float)
    err_a, err_b = np.abs(y - pred_a), np.abs(y - pred_b)
    return {
        "B_mejor": float((err_b < err_a).mean()),
        "A_mejor": float((err_a < err_b).mean()),
        "empate": float((err_a == err_b).mean()),
    }


def to_markdown(summary: dict, metrics: pd.DataFrame, segments: pd.DataFrame) -> str:
    a, b = summary["A"], summary["B"]
    wape, mae = summary["bootstrap"]["wape"], summary["bootstrap"]["mae"]
    wins = summary["estacion_dias_con_menor_error"]
    seg = segments.copy()
    for column in ("viajes_promedio", "mae_A", "mae_B"):
        seg[column] = seg[column].map("{:.2f}".format)
    for column in ("wape_A", "wape_B"):
        seg[column] = seg[column].map("{:.4f}".format)
    seg["cambio_wape_pct"] = seg["cambio_wape_pct"].map("{:+.1f} %".format)
    lines = [
        "# A/B testing de P1",
        "",
        f"- **A** = `{a['alias']}` v{a['version']} ({a['run']}).",
        f"- **B** = `{b['alias']}` v{b['version']} ({b['run']}).",
        (
            f"- Evaluación: {summary['periodo'][0]} a {summary['periodo'][1]}, "
            f"{summary['filas']:,} estación-días de {summary['estaciones']} estaciones. "
            "Ninguna versión entrenó con estas fechas."
        ),
        "",
        "## Métricas globales",
        "",
        "| Métrica | A | B | B - A |",
        "|---|---:|---:|---:|",
        *[
            f"| {m.upper()} | {r['A']:.4f} | {r['B']:.4f} | {r['diferencia_B_menos_A']:+.4f} |"
            for m, r in metrics.iterrows()
        ],
        "",
        f"## Diferencia B - A con bootstrap por estación ({summary['remuestreos']} remuestreos)",
        "",
        "| Métrica | B - A | IC 95 % | P(B mejor) |",
        "|---|---:|---|---:|",
        (
            f"| WAPE | {wape['diferencia']:+.4f} | [{wape['ic95_inferior']:+.4f}, "
            f"{wape['ic95_superior']:+.4f}] | {wape['prob_B_mejor']:.3f} |"
        ),
        (
            f"| MAE | {mae['diferencia']:+.3f} | [{mae['ic95_inferior']:+.3f}, "
            f"{mae['ic95_superior']:+.3f}] | {mae['prob_B_mejor']:.3f} |"
        ),
        "",
        (
            f"B tiene menor error en {wins['B_mejor']:.1%} de las estación-días y A en "
            f"{wins['A_mejor']:.1%} (empate en {wins['empate']:.1%})."
        ),
        "",
        "## Por segmento",
        "",
        "| " + " | ".join(seg.columns) + " |",
        "|" + "---|" * len(seg.columns),
        *["| " + " | ".join(map(str, row)) + " |" for row in seg.itertuples(index=False)],
        "",
        "## Decisión",
        "",
        f"**{DECISIONS[summary['decision']]}**",
        "",
    ]
    return "\n".join(lines)


def resolve(client: MlflowClient, name: str, alias: str) -> dict:
    version = client.get_model_version_by_alias(name, alias)
    return {
        "alias": alias,
        "version": int(version.version),
        "run": version.tags.get("source_run_name", ""),
        "run_id": version.run_id,
    }


@app.command()
def main(
    alias_a: str = typer.Option("champion", help="Alias de la versión vigente (A)."),
    alias_b: str = typer.Option("challenger", help="Alias de la versión alternativa (B)."),
    n_boot: int = typer.Option(AB_BOOTSTRAP_SAMPLES, help="Remuestreos del bootstrap."),
    promote: bool = typer.Option(False, help="Mover @champion a B si la decisión es promover."),
):
    """Compara A y B en el periodo del A/B, documenta el resultado y, si se pide, promueve B."""
    setup_mlflow()
    client = MlflowClient()
    name = REGISTERED_MODEL_NAMES["p1"]
    a, b = resolve(client, name, alias_a), resolve(client, name, alias_b)
    model_a = mlflow.pyfunc.load_model(f"models:/{name}/{a['version']}")
    model_b = mlflow.pyfunc.load_model(f"models:/{name}/{b['version']}")

    frame = eval_frame(pd.read_parquet(P1_DATASET_PATH))
    y = frame[TARGET].to_numpy(float)
    pred_a = np.asarray(model_a.predict(frame[P1_FEATURES]), float)
    pred_b = np.asarray(model_b.predict(frame[P1_FEATURES]), float)

    metrics = version_metrics(y, pred_a, pred_b)
    bootstrap = paired_bootstrap(y, pred_a, pred_b, frame["station_id"], n_boot)
    segments = segment_metrics(frame, pred_a, pred_b)
    decision = decide(bootstrap["wape"])
    summary = {
        "modelo": name,
        "A": a,
        "B": b,
        "periodo": [AB_EVAL_START, AB_EVAL_END],
        "filas": len(frame),
        "estaciones": int(frame["station_id"].nunique()),
        "remuestreos": n_boot,
        "semilla": SEED,
        "metricas": metrics.to_dict(orient="index"),
        "bootstrap": bootstrap,
        "estacion_dias_con_menor_error": win_rate(y, pred_a, pred_b),
        "segmentos": segments.to_dict(orient="records"),
        "decision": decision,
        "decision_texto": DECISIONS[decision],
    }

    MONITORING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = MONITORING_REPORTS_DIR / "p1_ab_test.json"
    md_path = MONITORING_REPORTS_DIR / "p1_ab_test.md"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(to_markdown(summary, metrics, segments), encoding="utf-8")
    plot_path = plots.plot_ab_daily_wape(
        frame["date"],
        y,
        pred_a,
        pred_b,
        (f"A: v{a['version']} ({alias_a})", f"B: v{b['version']} ({alias_b})"),
        "P1: WAPE diario del sistema en el periodo del A/B",
        path=MONITORING_REPORTS_DIR / "p1_ab_wape_diario.png",
    )

    run_name = f"p1-ab-v{a['version']}-vs-v{b['version']}"
    description = f"A/B de P1 {alias_a} v{a['version']} contra {alias_b} v{b['version']}."
    with start_run("P1", "abtest", "comparacion", "abtest", run_name, description) as run:
        mlflow.log_params(
            {
                "question": "p1",
                "version_a": a["version"],
                "version_b": b["version"],
                "eval_start": AB_EVAL_START,
                "eval_end": AB_EVAL_END,
                "n_eval": len(frame),
                "n_boot": n_boot,
                "seed": SEED,
            }
        )
        for side in ("A", "B"):
            mlflow.log_metrics({f"{side.lower()}_{m}": v for m, v in metrics[side].items()})
        mlflow.log_metrics(
            {f"diff_wape_{k}": v for k, v in bootstrap["wape"].items() if k != "prob_B_mejor"}
            | {"prob_b_better_wape": bootstrap["wape"]["prob_B_mejor"]}
        )
        mlflow.set_tag("ab_decision", decision)
        for path in (json_path, md_path, plot_path):
            mlflow.log_artifact(str(path), artifact_path="ab_test")
    for side in (a, b):
        client.set_model_version_tag(name, str(side["version"]), "ab_test_run_id", run.info.run_id)
    client.set_model_version_tag(name, str(b["version"]), "ab_test_decision", decision)

    if promote and decision == "promover":
        client.set_registered_model_alias(name, "champion", str(b["version"]))
        client.set_registered_model_alias(name, "challenger", str(a["version"]))
        logger.success(
            f"{name}: v{b['version']} es el nuevo champion; v{a['version']} queda de respaldo"
        )
    logger.success(f"{run_name}: {DECISIONS[decision]} Detalle en {md_path}")


if __name__ == "__main__":
    app()
