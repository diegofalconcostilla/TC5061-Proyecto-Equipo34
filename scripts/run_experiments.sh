#!/usr/bin/env bash
# Experimento completo del equipo: baselines, candidatas, búsquedas y registro de champions.
#
# Es la fuente de verdad de la evidencia: la corrida oficial es `make all` desde un clon limpio
# de main (servidor de MLflow vacío). Un modelo que no esté aquí no aparece en la entrega.
#
#   make experiments                  # requiere el servidor (make mlflow-server) y make ingest
#   ENV_MANAGER=local make experiments
#
# Para agregar una candidata: una línea `train` o `search` en la sección de su pregunta.
set -euo pipefail
cd "$(dirname "$0")/.."

train() { make --no-print-directory train Q="$1" M="$2" P="${3:-}"; }
search() { make --no-print-directory search Q="$1" M="$2" G="$3" P="${4:-}"; }
register() { make --no-print-directory register Q="$1" RUN="${2:-}" ALIAS="${3:-champion}"; }

# --- P1: demanda diaria por estación (selección: valid_wape, menor es mejor) ----------------
train p1 baseline
train p1 roll7
train p1 historico
train p1 poisson
train p1 hgb
# Candidatas y búsqueda de P1:

# --- P2: riesgo de desbalance por bloque de 3 h (selección: valid_pr_auc, mayor es mejor) ---
train p2 baseline
train p2 tasa
train p2 logistica
train p2 hgb
# Candidatas y búsqueda de P2:

# --- P3: grupos de estaciones por perfil de uso (selección: train_profile_silhouette) -------
train p3 baseline
train p3 regla
train p3 kmeans
search p3 kmeans "n_clusters=3|4|5;n_init=10|20"
# Candidatas y búsqueda de P3:

# --- Registro: mejor candidata por pregunta como champion -----------------------------------
# Para fijar un run concreto: register p1 "<run_name>"; para un challenger: register p1 "<run_name>" challenger
register p1
register p2
register p3
