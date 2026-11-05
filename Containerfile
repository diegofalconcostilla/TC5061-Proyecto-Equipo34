# Imagen de la API de MiBici (OCI: funciona con podman build y con docker build).
#
#   make api-models                        # copia los champions del Model Registry a models/api/
#   podman build -t mibici-api:1.1 .
#   podman run --rm -p 8000:8000 mibici-api:1.1
#
# Si models/api/ está vacío, la API descarga los modelos de MLFLOW_TRACKING_URI (ver README).
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MIBICI_MODELS_DIR=/app/models/api \
    MLFLOW_DISABLE_AGENT_HINT=1

# Dependencias primero: la capa se reutiliza mientras no cambie requirements-api.txt.
COPY requirements-api.txt pyproject.toml README.md LICENSE ./
COPY mibici ./mibici
RUN pip install -r requirements-api.txt

# Modelos descargados del registry (models/.gitkeep existe siempre, así que el COPY no falla).
COPY models ./models

# Sin privilegios de root dentro del contenedor.
RUN useradd --create-home --uid 10001 api
USER api

EXPOSE 8000

# Podman solo usa HEALTHCHECK con --format docker; Docker lo usa siempre.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

CMD ["uvicorn", "mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
