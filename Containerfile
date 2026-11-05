FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements-api.txt requirements.txt pyproject.toml README.md LICENSE ./
COPY mibici ./mibici

RUN pip install --upgrade pip && \
    pip install -r requirements-api.txt

EXPOSE 8000

CMD ["uvicorn", "mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
