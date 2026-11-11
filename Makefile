#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = TC5061-Proyecto-Equipo34
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

# MLflow: el puerto 5000 lo ocupa AirPlay en macOS.
MLFLOW_PORT ?= 5001
export MLFLOW_TRACKING_URI ?= http://127.0.0.1:$(MLFLOW_PORT)
# Entorno de mlflow run: uv (aislado, desde python_env.yaml) o local (el entorno activo).
export ENV_MANAGER ?= uv
ALIAS ?= champion
# Motor de contenedores de la API: podman o docker (mismo Containerfile, estándar OCI).
CONTAINER ?= podman
API_IMAGE ?= mibici-api:1.2
# Nombre de la máquina anfitriona dentro del contenedor, donde corre el servidor de MLflow.
CONTAINER_HOST = $(if $(filter docker,$(CONTAINER)),host.docker.internal,host.containers.internal)
MLFLOW_ALLOWED_HOSTS = localhost,localhost:$(MLFLOW_PORT),127.0.0.1,127.0.0.1:$(MLFLOW_PORT),host.containers.internal,host.containers.internal:$(MLFLOW_PORT),host.docker.internal,host.docker.internal:$(MLFLOW_PORT)
MLFLOW_RUN = mlflow run . --env-manager $(ENV_MANAGER) --experiment-name mibici-equipo34

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests

## Validar los datasets de P1, P2 y P3 con DeepChecks (reportes en reports/deepchecks/)
.PHONY: validate
validate:
	$(PYTHON_INTERPRETER) -m mibici.validation

## Chequeos de seguridad: código (bandit) y dependencias instaladas (pip-audit)
# setuptools 80.9.0 no se puede subir mientras DeepChecks necesite pkg_resources (ver README).
PIP_AUDIT_IGNORE = --ignore-vuln PYSEC-2026-3447 --ignore-vuln GHSA-xmj3-9gf7-h52w
.PHONY: security
security:
	bandit -r mibici
	pip-audit --skip-editable $(PIP_AUDIT_IGNORE)


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Levantar el servidor de tracking (SQLite + artefactos en ./mlartifacts servidos por HTTP)
# --artifacts-destination registra los modelos como mlflow-artifacts:/ (y no file:///ruta/local),
# así el contenedor de la API los descarga del servidor; --allowed-hosts acepta sus peticiones.
.PHONY: mlflow-server
mlflow-server:
	mlflow server --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts \
		--port $(MLFLOW_PORT) --allowed-hosts "$(MLFLOW_ALLOWED_HOSTS)"

## Ingesta y features desde el MLflow Project
.PHONY: ingest
ingest:
	$(MLFLOW_RUN) -e ingest

## (desarrollo) Descargar, limpiar y versionar los datos sin mlflow run
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m mibici.dataset ingest

## Verificar los datos locales contra data/manifest.json
.PHONY: verify
verify:
	$(PYTHON_INTERPRETER) -m mibici.dataset verify

## (desarrollo) Construir los datasets de P1, P2 y P3 sin mlflow run
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m mibici.features

## Entrenar un modelo con mlflow run: make train Q=p1 M=hgb P="learning_rate=0.1"
.PHONY: train
train:
	$(MLFLOW_RUN) -e train -P question=$(Q) -P model=$(M) -P params="$(P)"

## Búsqueda con nested runs: make search Q=p1 M=hgb G="learning_rate=0.03|0.1;max_iter=300|500"
.PHONY: search
search:
	$(MLFLOW_RUN) -e search -P question=$(Q) -P model=$(M) -P grid="$(G)" -P params="$(P)"

## Registrar y asignar alias: make register Q=p1 [RUN="nombre del run"] [ALIAS=champion]
.PHONY: register
register:
	$(MLFLOW_RUN) -e register -P question=$(Q) -P run_name="$(RUN)" -P alias=$(ALIAS)

## Predecir test con el modelo registrado: make predict Q=p1 [ALIAS=champion]
.PHONY: predict
predict:
	$(PYTHON_INTERPRETER) -m mibici.modeling.predict --question $(Q) --alias $(ALIAS)

## A/B de P1: champion contra challenger en julio-agosto de 2026 (reports/monitoreo/p1_ab_test.md)
.PHONY: abtest
abtest:
	$(MLFLOW_RUN) -e abtest

## Construir la imagen de la API: make api-build [CONTAINER=docker]
.PHONY: api-build
api-build:
	$(CONTAINER) build $(if $(filter podman,$(CONTAINER)),--format docker) -f Containerfile -t $(API_IMAGE) .

## Levantar la API en http://localhost:8000 con los modelos del registry (requiere make mlflow-server)
.PHONY: api-run
api-run:
	$(CONTAINER) run --rm --name mibici-api -p 8000:8000 \
		-e MLFLOW_TRACKING_URI=http://$(CONTAINER_HOST):$(MLFLOW_PORT) $(API_IMAGE)

## Todos los experimentos y el registro de champions (scripts/run_experiments.sh)
.PHONY: experiments
experiments:
	bash scripts/run_experiments.sh

## Corrida oficial desde un clon limpio: ingesta -> experimentos -> registro
.PHONY: all
all: ingest experiments
