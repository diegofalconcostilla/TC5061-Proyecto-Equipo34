#!/usr/bin/env bash
# Experimento completo del equipo: baselines, candidatas, búsquedas y registro de champions.
#
# Es la fuente de verdad de la evidencia: la corrida oficial es `make all` desde un clon limpio
# de main (servidor de MLflow vacío). Un modelo que no esté aquí no aparece en la entrega.
#
#   make experiments                  # requiere el servidor (make mlflow-server) y make ingest
#   ENV_MANAGER=local make experiments
#   bash scripts/run_experiments.sh   # sin make (Windows con Git Bash), mismo resultado
#
# Para agregar una candidata: una línea `train` o `search` en la sección de su pregunta.
set -euo pipefail
cd "$(dirname "$0")/.."

# PYTHONUTF8: en Windows, la consola cp1252 no puede imprimir los mensajes de mlflow run.
export PYTHONUTF8=1
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:${MLFLOW_PORT:-5001}}"
mlflow_run() {
    mlflow run . --env-manager "${ENV_MANAGER:-uv}" --experiment-name mibici-equipo34 "$@"
}
train() { mlflow_run -e train -P question="$1" -P model="$2" -P params="${3:-}"; }
search() { mlflow_run -e search -P question="$1" -P model="$2" -P grid="$3" -P params="${4:-}"; }
register() {
    mlflow_run -e register -P question="$1" -P run_name="${2:-}" -P alias="${3:-champion}"
}

# --- P1: demanda diaria (selección: valid_wape, menor es mejor) ----------------------------
train p1 baseline
train p1 roll7
train p1 historico
train p1 poisson
train p1 ridge
train p1 rf
train p1 hgb
train p1 hgb "loss=squared_error"
search p1 poisson "alpha=0.0001|0.01|0.1"
search p1 hgb "learning_rate=0.03|0.1;max_leaf_nodes=31|63"

# --- P2: riesgo por bloque de 3 horas (selección: valid_pr_auc, mayor es mejor) ------------
train p2 baseline
train p2 tasa
train p2 logistica
train p2 rf
train p2 hgb
search p2 logistica "C=0.1|1.0|10.0"
search p2 hgb "learning_rate=0.03|0.1;max_leaf_nodes=31|63"

# --- P3: grupos de estaciones (selección: train_profile_silhouette, mayor es mejor) --------
train p3 baseline
train p3 regla
train p3 kmeans
train p3 minibatch
train p3 birch
train p3 gmm
search p3 kmeans "n_clusters=3|4|5|6|7|8;n_init=10|20"
search p3 gmm "n_clusters=3|4|5;covariance_type=full|diag"

# --- Registro: mejor candidata por pregunta como champion -----------------------------------
# Para fijar un run concreto: register p1 "<run_name>"; para un challenger: register p1 "<run_name>" challenger
register p1
register p2
register p3
