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
ENV_MANAGER ?= uv
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


## Levantar el servidor de tracking (SQLite + artefactos en ./mlartifacts)
.PHONY: mlflow-server
mlflow-server:
	mlflow server --backend-store-uri sqlite:///mlflow.db --port $(MLFLOW_PORT)

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

## Pipeline completo: datos -> features -> experimento
.PHONY: all
all: data features train
