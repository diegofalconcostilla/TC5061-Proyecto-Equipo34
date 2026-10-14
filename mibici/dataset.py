"""Ingesta reproducible y versionada de los datos abiertos de MiBici.

Uso desde la terminal (sin pasos manuales):

    python -m mibici.dataset ingest            # descarga, limpia, guarda Parquet y manifiesto
    python -m mibici.dataset verify            # comprueba los archivos locales contra el manifiesto

Pasos de ``ingest``:

1. Localiza en el portal de datos abiertos la URL de cada mes del rango configurado y de la
   nomenclatura de estaciones (o reutiliza las URLs registradas en el manifiesto).
2. Descarga los CSV crudos a ``data/raw/``.
3. Normaliza codificación (UTF-8 o Latin-1), nombres de columnas y tipos de fecha; elimina
   los datos personales y los viajes inválidos.
4. Guarda un Parquet por mes en ``data/interim/trips/`` y las estaciones en
   ``data/interim/stations.parquet``.
5. Escribe ``data/manifest.json`` (versionado en Git) con el hash SHA-256 y el número de
   registros de cada archivo, y una ``data_version`` que se registra en MLflow.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from loguru import logger
import pandas as pd
import requests
from tqdm import tqdm
import typer

from mibici.config import (
    DATA_MANIFEST_PATH,
    DATE_END,
    DATE_START,
    HTTP_TIMEOUT_SECONDS,
    MAX_TRIP_MINUTES,
    MIBICI_BASE_URL,
    MIBICI_OPEN_DATA_URL,
    MIN_TRIP_MINUTES,
    PERSONAL_DATA_COLUMNS,
    PROJ_ROOT,
    RAW_DATA_DIR,
    STATIONS_FILE_NAME,
    STATIONS_INTERIM_PATH,
    TRIPS_INTERIM_DIR,
)

app = typer.Typer(help="Ingesta reproducible de los datos abiertos de MiBici.")

# Versión de la lógica de limpieza. Si cambian las reglas, se incrementa y cambia data_version.
PIPELINE_VERSION = "1.0.0"

# Reglas (expresión regular sobre el nombre normalizado -> nombre canónico). Se usan palabras
# clave porque el encabezado varía entre meses: "Año_de_nacimiento" llega en UTF-8, en
# Latin-1 o corrupto (p. ej. "A}äe_nacimiento" en 2024-02).
TRIP_COLUMN_RULES = [
    (r"viaje.*id", "viaje_id"),
    (r"usuario", "usuario_id"),
    (r"g.?nero|sexo", "genero"),
    (r"nacimiento|edad", "anio_nacimiento"),
    (r"inicio", "inicio_viaje"),
    (r"fin", "fin_viaje"),
    (r"origen", "origen_id"),
    (r"destino", "destino_id"),
]
STATION_COLUMN_ALIASES = {
    "id": "station_id",
    "name": "name",
    "latitude": "latitude",
    "longitude": "longitude",
    "dpcapacity": "capacity",
}
TRIP_REQUIRED = ["viaje_id", "inicio_viaje", "fin_viaje", "origen_id", "destino_id"]
CHUNK_BYTES = 1 << 20
# El portal responde 406 al User-Agent por defecto de requests.
HTTP_HEADERS = {"User-Agent": "mibici-equipo34-ingesta/1.0", "Accept": "text/html,text/csv,*/*"}


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def month_range(start: str, end: str) -> list[str]:
    """Lista de meses ``AAAA_MM`` entre ``start`` y ``end`` (formato ``AAAA-MM``), inclusive."""
    periods = pd.period_range(start=start, end=end, freq="M")
    return [p.strftime("%Y_%m") for p in periods]


def sha256_file(path: Path) -> str:
    """Hash SHA-256 del contenido de un archivo, leído por bloques."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_name(name: str) -> str:
    """Normaliza un nombre de columna: corrige mojibake, quita acentos y usa snake_case."""
    name = fix_mojibake(str(name)).strip().strip('"').replace("﻿", "")
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return name


def fix_mojibake(text: str) -> str:
    """Repara texto UTF-8 que fue decodificado como Latin-1 (p. ej. ``AÃ±o`` -> ``Año``)."""
    if "Ã" in text or "Â" in text:
        try:
            return text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            return text
    return text


def detect_encoding(path: Path) -> str:
    """Devuelve ``utf-8`` si el archivo decodifica como UTF-8 y ``latin-1`` en caso contrario."""
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        data.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def relative(path: Path) -> str:
    """Ruta relativa a la raíz del proyecto con separador ``/`` (portátil entre sistemas)."""
    return path.resolve().relative_to(PROJ_ROOT).as_posix()


# ---------------------------------------------------------------------------
# Descubrimiento y descarga
# ---------------------------------------------------------------------------
def discover_urls() -> dict[str, str]:
    """Lee el portal de datos abiertos y devuelve ``{nombre_archivo: url}``.

    Las URLs del portal incluyen un identificador numérico que no se puede inferir, por lo
    que se extraen de los enlaces de la página.
    """
    logger.info(f"Consultando el portal {MIBICI_OPEN_DATA_URL}")
    response = requests.get(
        MIBICI_OPEN_DATA_URL, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    urls: dict[str, str] = {}
    pattern = r'href="([^"]*/((?:datos_abiertos_\d{4}_\d{2}|nomenclatura_\d{4}_\d{2})[^"/]*\.csv))"'
    for href, filename in re.findall(pattern, response.text):
        match = re.match(r"(datos_abiertos_\d{4}_\d{2}|nomenclatura_\d{4}_\d{2})", filename)
        key = f"{match.group(1)}.csv"
        url = href if href.startswith("http") else f"{MIBICI_BASE_URL}{href}"
        urls.setdefault(key, url)
    logger.info(f"{len(urls)} archivos encontrados en el portal")
    return urls


def download(url: str, dest: Path, force: bool = False) -> Path:
    """Descarga ``url`` a ``dest`` por streaming; no vuelve a descargar si ya existe."""
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(
        url, headers=HTTP_HEADERS, stream=True, timeout=HTTP_TIMEOUT_SECONDS
    ) as response:
        response.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in response.iter_content(CHUNK_BYTES):
                fh.write(block)
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# Normalización y limpieza
# ---------------------------------------------------------------------------
def read_raw_csv(path: Path) -> tuple[pd.DataFrame, str]:
    """Lee un CSV crudo de MiBici como texto, detectando la codificación."""
    encoding = detect_encoding(path)
    df = pd.read_csv(
        path,
        encoding=encoding,
        dtype=str,
        na_values=["NULL", "null", ""],
        keep_default_na=False,
    )
    return df, encoding


def normalize_trip_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Renombra las columnas de viajes a los nombres canónicos del proyecto."""
    renamed = {}
    for col in df.columns:
        norm = normalize_name(col)
        target = next((canon for rx, canon in TRIP_COLUMN_RULES if re.search(rx, norm)), None)
        if target is None:
            raise ValueError(f"Columna de viajes no reconocida: {col!r} (normalizada: {norm!r})")
        if target in renamed.values():
            raise ValueError(f"Dos columnas se mapean a {target!r}: revisar {col!r}")
        renamed[col] = target
    df = df.rename(columns=renamed)
    missing = set(TRIP_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(missing)}")
    return df


def clean_trips(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Tipifica, elimina datos personales y filtra viajes inválidos.

    Reglas: fechas parseables, estaciones numéricas, fin posterior al inicio y duración
    entre ``MIN_TRIP_MINUTES`` y ``MAX_TRIP_MINUTES``; se eliminan duplicados por
    ``viaje_id``.
    """
    stats = {"rows_raw": len(df)}
    df = normalize_trip_columns(df)
    df = df.drop(columns=[c for c in PERSONAL_DATA_COLUMNS if c in df.columns])

    out = pd.DataFrame(
        {
            "viaje_id": pd.to_numeric(df["viaje_id"], errors="coerce").astype("Int64"),
            "inicio_viaje": pd.to_datetime(df["inicio_viaje"], errors="coerce", format="mixed"),
            "fin_viaje": pd.to_datetime(df["fin_viaje"], errors="coerce", format="mixed"),
            "origen_id": pd.to_numeric(df["origen_id"], errors="coerce").astype("Int32"),
            "destino_id": pd.to_numeric(df["destino_id"], errors="coerce").astype("Int32"),
        }
    )
    out = out.dropna()
    stats["dropped_unparseable"] = stats["rows_raw"] - len(out)

    duration = (out["fin_viaje"] - out["inicio_viaje"]).dt.total_seconds() / 60
    valid = duration.between(MIN_TRIP_MINUTES, MAX_TRIP_MINUTES)
    stats["dropped_duration"] = int((~valid).sum())
    out = out.loc[valid].assign(duracion_min=duration[valid].astype("float32"))

    before = len(out)
    out = out.drop_duplicates(subset="viaje_id")
    stats["dropped_duplicates"] = before - len(out)

    out = out.sort_values("inicio_viaje").reset_index(drop=True)
    out["viaje_id"] = out["viaje_id"].astype("int64")
    out["origen_id"] = out["origen_id"].astype("int32")
    out["destino_id"] = out["destino_id"].astype("int32")
    stats["rows_clean"] = len(out)
    return out, stats


def clean_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza la nomenclatura de estaciones y extrae el municipio del código."""
    df = df.rename(columns={c: STATION_COLUMN_ALIASES.get(normalize_name(c), c) for c in df})
    out = pd.DataFrame(
        {
            "station_id": pd.to_numeric(df["station_id"], errors="coerce").astype("Int32"),
            "name": df["name"].map(lambda s: fix_mojibake(re.sub(r"\s+", " ", str(s)).strip())),
            "latitude": pd.to_numeric(df["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(df["longitude"], errors="coerce"),
            "capacity": pd.to_numeric(df["capacity"], errors="coerce").astype("Int16"),
        }
    ).dropna()
    # El nombre comienza con un código como "(GDL-001)"; el prefijo identifica al municipio.
    out["municipio"] = out["name"].str.extract(r"^\(([A-Z]+)-", expand=False).fillna("NA")
    out = out.drop_duplicates("station_id").sort_values("station_id").reset_index(drop=True)
    return out.astype({"station_id": "int32", "capacity": "int16"})


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Guarda un DataFrame en Parquet (compresión zstd, sin índice)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    return path


# ---------------------------------------------------------------------------
# Manifiesto y versión de datos
# ---------------------------------------------------------------------------
def compute_data_version(entries: list[dict]) -> str:
    """Hash corto que identifica exactamente los datos fuente y las reglas de limpieza."""
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "min_trip_minutes": MIN_TRIP_MINUTES,
        "max_trip_minutes": MAX_TRIP_MINUTES,
        "files": sorted((e["file"], e["raw_sha256"], e["rows_clean"]) for e in entries),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def load_manifest(path: Path = DATA_MANIFEST_PATH) -> dict:
    """Carga el manifiesto de datos; devuelve un diccionario vacío si no existe."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def get_data_version(path: Path = DATA_MANIFEST_PATH) -> str:
    """Versión de datos registrada en el manifiesto (se usa como parámetro en MLflow)."""
    manifest = load_manifest(path)
    if not manifest:
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero: python -m mibici.dataset ingest"
        )
    return manifest["data_version"]


# ---------------------------------------------------------------------------
# Carga para los demás módulos
# ---------------------------------------------------------------------------
def load_trips(start: str = DATE_START, end: str = DATE_END) -> pd.DataFrame:
    """Carga los viajes limpios de los meses ``start``..``end`` desde Parquet."""
    paths = [TRIPS_INTERIM_DIR / f"trips_{m}.parquet" for m in month_range(start, end)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Faltan {len(missing)} meses ({missing[:3]}...). Ejecuta: python -m mibici.dataset ingest"
        )
    return pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)


def load_stations() -> pd.DataFrame:
    """Carga la nomenclatura de estaciones limpia."""
    return pd.read_parquet(STATIONS_INTERIM_PATH)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@app.command()
def ingest(
    start: str = typer.Option(DATE_START, help="Primer mes (AAAA-MM)."),
    end: str = typer.Option(DATE_END, help="Último mes (AAAA-MM)."),
    force: bool = typer.Option(False, help="Volver a descargar y procesar todo."),
):
    """Descarga, normaliza, guarda en Parquet y versiona los datos de MiBici."""
    previous = load_manifest()
    known_urls = {e["file"]: e["source_url"] for e in previous.get("files", [])}
    months = month_range(start, end)
    wanted = [f"datos_abiertos_{m}.csv" for m in months] + [STATIONS_FILE_NAME]

    if any(name not in known_urls for name in wanted):
        portal = discover_urls()
        for name in wanted:
            if name not in known_urls:
                if name not in portal:
                    raise RuntimeError(f"El portal de MiBici no publica {name}")
                known_urls[name] = portal[name]

    entries: list[dict] = []
    station_ids: set[int] = set()
    for month in tqdm(months, desc="Meses"):
        name = f"datos_abiertos_{month}.csv"
        raw_path = download(known_urls[name], RAW_DATA_DIR / name, force=force)
        clean_path = TRIPS_INTERIM_DIR / f"trips_{month}.parquet"
        raw, encoding = read_raw_csv(raw_path)
        trips, stats = clean_trips(raw)
        write_parquet(trips, clean_path)
        station_ids.update(trips["origen_id"].unique().tolist())
        station_ids.update(trips["destino_id"].unique().tolist())
        entries.append(
            {
                "file": name,
                "kind": "trips",
                "month": month.replace("_", "-"),
                "source_url": known_urls[name],
                "encoding": encoding,
                "raw_path": relative(raw_path),
                "raw_bytes": raw_path.stat().st_size,
                "raw_sha256": sha256_file(raw_path),
                "clean_path": relative(clean_path),
                "clean_sha256": sha256_file(clean_path),
                **stats,
            }
        )
        logger.info(f"{name}: {stats['rows_raw']:,} crudos -> {stats['rows_clean']:,} limpios")

    raw_path = download(known_urls[STATIONS_FILE_NAME], RAW_DATA_DIR / STATIONS_FILE_NAME, force)
    raw_stations, encoding = read_raw_csv(raw_path)
    stations = clean_stations(raw_stations)
    write_parquet(stations, STATIONS_INTERIM_PATH)
    entries.append(
        {
            "file": STATIONS_FILE_NAME,
            "kind": "stations",
            "source_url": known_urls[STATIONS_FILE_NAME],
            "encoding": encoding,
            "raw_path": relative(raw_path),
            "raw_bytes": raw_path.stat().st_size,
            "raw_sha256": sha256_file(raw_path),
            "clean_path": relative(STATIONS_INTERIM_PATH),
            "clean_sha256": sha256_file(STATIONS_INTERIM_PATH),
            "rows_raw": len(raw_stations),
            "rows_clean": len(stations),
        }
    )

    trip_entries = [e for e in entries if e["kind"] == "trips"]
    manifest = {
        "dataset": "MiBici - datos abiertos",
        "source": MIBICI_OPEN_DATA_URL,
        "pipeline_version": PIPELINE_VERSION,
        "date_start": start,
        "date_end": end,
        "data_version": compute_data_version(entries),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "totals": {
            "months": len(trip_entries),
            "trips_raw": sum(e["rows_raw"] for e in trip_entries),
            "trips_clean": sum(e["rows_clean"] for e in trip_entries),
            "stations_catalog": len(stations),
            "stations_with_trips": len(station_ids),
        },
        "files": entries,
    }
    if previous.get("data_version") == manifest["data_version"]:
        manifest["generated_at"] = previous["generated_at"]  # mismo contenido, mismo manifiesto
    DATA_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.success(
        f"Ingesta completa: {manifest['totals']['trips_clean']:,} viajes, "
        f"{len(stations)} estaciones, data_version={manifest['data_version']}"
    )


@app.command()
def verify():
    """Comprueba que los archivos locales coinciden con los hashes del manifiesto."""
    manifest = load_manifest()
    if not manifest:
        raise typer.Exit(code=1)
    problems = 0
    for entry in tqdm(manifest["files"], desc="Verificando"):
        for key in ("raw", "clean"):
            path = PROJ_ROOT / entry[f"{key}_path"]
            if not path.exists():
                logger.error(f"Falta {path}")
                problems += 1
            elif sha256_file(path) != entry[f"{key}_sha256"]:
                logger.error(f"Hash distinto en {path}")
                problems += 1
    if problems:
        logger.error(f"{problems} problemas encontrados")
        raise typer.Exit(code=1)
    logger.success(f"Datos íntegros. data_version={manifest['data_version']}")


if __name__ == "__main__":
    app()
