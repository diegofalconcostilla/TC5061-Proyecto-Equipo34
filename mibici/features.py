"""Feature engineering para las tres preguntas del proyecto MiBici.

Cada función documenta qué calcula y por qué es pertinente para la pregunta que atiende.
Regla general contra la fuga de información: toda variable usada para predecir el periodo
``t`` se calcula solo con datos disponibles antes de que inicie ``t`` (desplazamientos con
``shift`` de al menos un periodo y ventanas móviles sobre la serie ya desplazada). Los
perfiles de P3 se calculan solo con el periodo de entrenamiento.

Uso desde la terminal:

    python -m mibici.features            # construye los datasets de P1, P2 y P3 en data/processed/
"""

from __future__ import annotations

from datetime import date, timedelta

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici.config import (
    P1_DATASET_PATH,
    P1_LAGS_DAYS,
    P1_ROLLING_DAYS,
    P2_BLOCK_HOURS,
    P2_BLOCKS_PER_DAY,
    P2_DATASET_PATH,
    P2_FIRST_HOUR,
    P2_RISK_CAPACITY_FRACTION,
    P2_ROLLING_WEEKS,
    P3_DATASET_PATH,
    P3_MIN_DEPARTURES,
    P3_TEST_DATASET_PATH,
    TEST_START,
    VALID_START,
)

app = typer.Typer(help="Construye los datasets de modelado de P1, P2 y P3.")


# ---------------------------------------------------------------------------
# Calendario (P1 y P2)
# ---------------------------------------------------------------------------
def easter_sunday(year: int) -> date:
    """Domingo de Pascua (algoritmo gregoriano anónimo).

    Se usa para ubicar Jueves y Viernes Santo, días de muy baja demanda en Guadalajara.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return date(year, month, day + 1)


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Fecha del ``n``-ésimo ``weekday`` (0 = lunes) de un mes."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


def mexican_holidays(years: list[int]) -> set[date]:
    """Días de descanso obligatorio en México más Jueves y Viernes Santo.

    Pertinencia (P1, P2): en días festivos desaparecen los viajes de traslado al trabajo,
    lo que cambia el volumen y el patrón horario de uso de cada estación.
    """
    days: set[date] = set()
    for y in years:
        easter = easter_sunday(y)
        days |= {
            date(y, 1, 1),
            nth_weekday(y, 2, 0, 1),  # Día de la Constitución
            nth_weekday(y, 3, 0, 3),  # Natalicio de Benito Juárez
            date(y, 5, 1),
            date(y, 9, 16),
            nth_weekday(y, 11, 0, 3),  # Revolución Mexicana
            date(y, 12, 25),
            easter - timedelta(days=3),  # Jueves Santo
            easter - timedelta(days=2),  # Viernes Santo
        }
        if y % 6 == 0:
            days.add(date(y, 10, 1))  # Transmisión del Poder Ejecutivo Federal (2024, 2030)
    return days


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Agrega día de la semana, mes, fin de semana y festivo a partir de ``date_col``.

    Pertinencia (P1, P2): la demanda de MiBici tiene una estacionalidad semanal fuerte
    (traslados laborales entre semana) y una anual (vacaciones escolares, lluvias). Estas
    variables se conocen de antemano, por lo que no generan fuga de información.
    """
    dates = pd.to_datetime(df[date_col])
    holidays = mexican_holidays(sorted(dates.dt.year.unique().tolist()))
    return df.assign(
        dow=dates.dt.dayofweek.astype("int8"),
        month=dates.dt.month.astype("int8"),
        is_weekend=(dates.dt.dayofweek >= 5).astype("int8"),
        is_holiday=dates.dt.date.isin(holidays).astype("int8"),
    )


# ---------------------------------------------------------------------------
# Utilidades de series por estación (P1 y P2)
# ---------------------------------------------------------------------------
def station_grid(
    trips: pd.DataFrame, stations: pd.DataFrame, periods: pd.DatetimeIndex
) -> pd.DataFrame:
    """Rejilla completa estación x periodo desde el primer viaje observado de cada estación.

    Pertinencia (P1, P2): un periodo sin viajes es un cero real, no un dato faltante; sin la
    rejilla, los lags y promedios móviles saltarían periodos y estarían sesgados. Se descartan
    los periodos previos a la apertura de cada estación para no inventar ceros. Solo se
    incluyen estaciones del catálogo con capacidad positiva (las retiradas, marcadas "ex-",
    tienen capacidad 0), porque P2 necesita la capacidad para definir el riesgo.
    """
    stations = stations[stations["capacity"] > 0]
    first_seen = (
        trips.groupby("origen_id")["inicio_viaje"].min().dt.normalize().rename("first_seen")
    )
    ids = stations[["station_id"]].merge(
        first_seen, left_on="station_id", right_index=True, how="inner"
    )
    grid = ids.merge(pd.DataFrame({"period": periods}), how="cross")
    grid = grid[grid["period"] >= grid["first_seen"]].drop(columns="first_seen")
    return grid.reset_index(drop=True)


def add_lag_features(
    df: pd.DataFrame,
    target: str,
    lags: list[int],
    windows: list[int],
    min_shift: int,
    group: str = "station_id",
) -> pd.DataFrame:
    """Agrega lags y promedios móviles de ``target`` por estación, sin fuga temporal.

    ``df`` debe estar ordenado por ``group`` y periodo, con periodos consecutivos. Las
    ventanas móviles se calculan sobre la serie desplazada ``min_shift`` periodos, de modo que
    solo usan información anterior al horizonte de predicción.

    Pertinencia (P1, P2): la demanda reciente de una estación (ayer, hace una semana) es el
    mejor predictor de su demanda próxima; los promedios móviles suavizan el ruido diario.
    """
    if any(lag < min_shift for lag in lags):
        raise ValueError(f"Todos los lags deben ser >= {min_shift} para evitar fuga")
    grouped = df.groupby(group, sort=False)[target]
    new_cols = {f"{target}_lag{lag}": grouped.shift(lag) for lag in lags}
    if windows:
        shifted = grouped.shift(min_shift)
        for w in windows:
            new_cols[f"{target}_roll{w}_mean"] = (
                shifted.groupby(df[group], sort=False)
                .rolling(w, min_periods=max(1, w // 2))
                .mean()
                .reset_index(level=0, drop=True)
            )
    return df.assign(**new_cols)


def assign_split(dates: pd.Series) -> pd.Series:
    """Etiqueta cada fila como ``train``, ``valid`` o ``test`` según los cortes de config.py.

    Pertinencia (P1, P2, P3): la partición es temporal (no aleatoria) para que el modelo se
    evalúe siempre en fechas posteriores a las que vio al entrenar, como ocurrirá en
    operación.
    """
    dates = pd.to_datetime(dates)
    split = np.where(
        dates < pd.Timestamp(VALID_START),
        "train",
        np.where(dates < pd.Timestamp(TEST_START), "valid", "test"),
    )
    return pd.Series(split, index=dates.index)


def add_station_static_features(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Agrega capacidad, coordenadas y municipio de cada estación.

    Pertinencia (P1, P2): la capacidad acota cuántas bicicletas puede ofrecer o recibir una
    estación y la ubicación distingue zonas céntricas de periféricas. Son atributos estáticos
    que no dependen del tiempo.
    """
    cols = ["station_id", "capacity", "latitude", "longitude", "municipio"]
    return df.merge(stations[cols], on="station_id", how="left")


# ---------------------------------------------------------------------------
# P1: viajes iniciados por estación y día (regresión)
# ---------------------------------------------------------------------------
def build_p1_dataset(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Dataset estación-día con objetivo ``departures`` (viajes iniciados ese día).

    Granularidad: estación x día. Horizonte: 1 día (al cierre del día ``d-1`` se predice el
    día ``d``). Variables: calendario, atributos de la estación, lags de 1, 2, 7 y 14 días y
    promedios móviles de 7 y 28 días de salidas, y lags de llegadas; todo desplazado al
    menos 1 día.
    """
    trips = trips.assign(date=trips["inicio_viaje"].dt.normalize())
    days = pd.date_range(trips["date"].min(), trips["date"].max(), freq="D")
    grid = station_grid(trips, stations, days).rename(columns={"period": "date"})

    dep = trips.groupby(["origen_id", "date"]).size().rename("departures")
    dep.index.names = ["station_id", "date"]
    arr = (
        trips.assign(date=trips["fin_viaje"].dt.normalize())
        .groupby(["destino_id", "date"])
        .size()
        .rename("arrivals")
    )
    arr.index.names = ["station_id", "date"]
    df = grid.join(dep, on=["station_id", "date"]).join(arr, on=["station_id", "date"])
    df[["departures", "arrivals"]] = df[["departures", "arrivals"]].fillna(0).astype("int32")
    df = df.sort_values(["station_id", "date"]).reset_index(drop=True)

    df = add_lag_features(df, "departures", P1_LAGS_DAYS, P1_ROLLING_DAYS, min_shift=1)
    df = add_lag_features(df, "arrivals", [1, 7], [7], min_shift=1)
    df = add_calendar_features(df, "date")
    df = add_station_static_features(df, stations)
    df["split"] = assign_split(df["date"])
    # Se descartan las primeras filas de cada estación, sin historia suficiente para los lags.
    return df.dropna(subset=[f"departures_lag{max(P1_LAGS_DAYS)}"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# P2: riesgo de quedarse sin bicicletas o sin anclajes (clasificación)
# ---------------------------------------------------------------------------
def time_block(ts: pd.Series) -> pd.Series:
    """Índice del bloque de ``P2_BLOCK_HOURS`` horas del día (0 = 06:00-09:00).

    Los viajes de madrugada (antes de las 06:00, muy pocos) se asignan al primer bloque.
    """
    hours = ts.dt.hour.clip(lower=P2_FIRST_HOUR)
    return ((hours - P2_FIRST_HOUR) // P2_BLOCK_HOURS).astype("int8")


def block_start(ts: pd.Series) -> pd.Series:
    """Asigna cada viaje al inicio de su bloque de 3 horas.

    Los viajes anteriores a las 06:00 quedan fuera de los bloques
    analizados por P2, que cubren de las 06:00 a las 24:00.
    """
    hours = ts.dt.hour
    valid = hours >= P2_FIRST_HOUR

    block = (hours.clip(lower=P2_FIRST_HOUR) - P2_FIRST_HOUR) // P2_BLOCK_HOURS

    start = ts.dt.normalize() + pd.to_timedelta(
        P2_FIRST_HOUR + block * P2_BLOCK_HOURS,
        unit="h",
    )

    return start.where(valid)


def risk_label(net_flow: pd.Series, capacity: pd.Series) -> pd.DataFrame:
    """Aproxima el riesgo de desabasto a partir del flujo neto del bloque y la capacidad.

    MiBici no publica el inventario por estación. Se usa el flujo neto
    ``llegadas - salidas`` dentro del bloque: si en 3 horas salen al menos
    ``P2_RISK_CAPACITY_FRACTION`` x capacidad bicicletas más de las que llegan, la estación
    tiende a vaciarse (``risk_empty``); si llegan esa cantidad de más, tiende a llenarse y
    quedarse sin anclajes (``risk_full``). Partiendo de una ocupación típica cercana al 50 %,
    un desbalance de media capacidad basta para agotar bicicletas o anclajes si nadie
    rebalancea.
    """
    threshold = np.ceil(P2_RISK_CAPACITY_FRACTION * capacity.astype("float64"))
    risk_empty = (net_flow <= -threshold).astype("int8")
    risk_full = (net_flow >= threshold).astype("int8")
    return pd.DataFrame(
        {"risk_empty": risk_empty, "risk_full": risk_full, "risk": risk_empty | risk_full}
    )


def build_p2_dataset(trips: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Dataset estación-bloque de 3 horas con objetivo binario ``risk``.

    Granularidad: estación x bloque de 3 horas (6 bloques de 06:00 a 24:00). Horizonte: el
    siguiente bloque (al inicio del bloque ``b`` se predice si estará en riesgo durante ``b``).
    Variables: bloque, calendario, capacidad y ubicación; flujo neto, salidas y riesgo del
    bloque anterior, del mismo bloque ayer y hace una semana; y promedio de |flujo neto| del
    mismo bloque en las 4 semanas previas.
    """
    days = pd.date_range(
        trips["inicio_viaje"].min().normalize(), trips["inicio_viaje"].max().normalize()
    )
    offsets = pd.to_timedelta(
        [P2_FIRST_HOUR + b * P2_BLOCK_HOURS for b in range(P2_BLOCKS_PER_DAY)], unit="h"
    )
    blocks = pd.DatetimeIndex([d + o for d in days for o in offsets])
    grid = station_grid(trips, stations, blocks).rename(columns={"period": "block_ts"})

    n_dep = trips.groupby(["origen_id", block_start(trips["inicio_viaje"])]).size()
    n_arr = trips.groupby(["destino_id", block_start(trips["fin_viaje"])]).size()
    n_dep.index.names = n_arr.index.names = ["station_id", "block_ts"]
    df = grid.join(n_dep.rename("departures"), on=["station_id", "block_ts"]).join(
        n_arr.rename("arrivals"), on=["station_id", "block_ts"]
    )
    df[["departures", "arrivals"]] = df[["departures", "arrivals"]].fillna(0).astype("int32")
    df["net_flow"] = df["arrivals"] - df["departures"]
    df = df.sort_values(["station_id", "block_ts"]).reset_index(drop=True)
    df = add_station_static_features(df, stations)
    df = pd.concat([df, risk_label(df["net_flow"], df["capacity"])], axis=1)

    per_day, per_week = P2_BLOCKS_PER_DAY, 7 * P2_BLOCKS_PER_DAY
    df = add_lag_features(df, "net_flow", [1, per_day, per_week], [], min_shift=1)
    df = add_lag_features(df, "departures", [1, per_day, per_week], [], min_shift=1)
    df = add_lag_features(df, "risk", [per_day, per_week], [], min_shift=1)
    # Promedio del mismo bloque en las semanas previas (lags semanales; excluye la actual).
    abs_net = df["net_flow"].abs().groupby(df["station_id"], sort=False)
    weekly = [abs_net.shift(per_week * w) for w in range(1, P2_ROLLING_WEEKS + 1)]
    df[f"abs_net_flow_same_block_{P2_ROLLING_WEEKS}w_mean"] = pd.concat(weekly, axis=1).mean(
        axis=1
    )
    df["block"] = time_block(df["block_ts"])
    df["date"] = df["block_ts"].dt.normalize()
    df = add_calendar_features(df, "date")
    df["split"] = assign_split(df["date"])
    return df.dropna(subset=[f"net_flow_lag{per_week}"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# P3: perfiles de uso por estación (clustering)
# ---------------------------------------------------------------------------
P3_PROFILE_PREFIXES = ("dep_wd_h", "dep_we_h")
P3_SCALAR_FEATURES = [
    "log_daily_departures",
    "net_flow_ratio",
    "weekend_share",
    "median_duration_min",
]


def build_p3_dataset(
    trips: pd.DataFrame, stations: pd.DataFrame, split: str = "train"
) -> pd.DataFrame:
    """Perfil de uso de cada estación calculado con un periodo (por defecto, entrenamiento).

    Variables por estación:

    - ``dep_wd_hHH`` / ``dep_we_hHH``: proporción de salidas en cada hora para días hábiles y
      fines de semana (48 columnas). Distinguen estaciones de origen matutino (zonas
      residenciales), de destino laboral y de uso recreativo de fin de semana.
    - ``log_daily_departures``: volumen medio diario (escala log); separa estaciones
      troncales de estaciones de baja demanda.
    - ``net_flow_ratio``: (llegadas - salidas) / (llegadas + salidas); estaciones que
      sistemáticamente reciben o despachan bicicletas.
    - ``weekend_share``: proporción de salidas en fin de semana.
    - ``median_duration_min``: duración mediana de los viajes que salen de la estación.

    Pertinencia (P3): agrupar estaciones por estos patrones permite identificar zonas con
    comportamiento similar para planear la expansión. Los grupos se ajustan solo con el
    periodo ``train``; los perfiles de ``test`` sirven para medir si los grupos se mantienen
    estables en el tiempo.
    """
    train = trips[assign_split(trips["inicio_viaje"]) == split]
    hours = train["inicio_viaje"].dt.hour
    is_we = train["inicio_viaje"].dt.dayofweek >= 5

    def hourly_share(mask: pd.Series, tag: str) -> pd.DataFrame:
        counts = train[mask].groupby(["origen_id", hours[mask]]).size().unstack(fill_value=0)
        counts = counts.reindex(columns=range(24), fill_value=0)
        share = counts.div(counts.sum(axis=1), axis=0).fillna(0)
        share.columns = [f"dep_{tag}_h{h:02d}" for h in range(24)]
        return share

    profile = hourly_share(~is_we, "wd").join(hourly_share(is_we, "we"), how="outer").fillna(0)
    n_days = train["inicio_viaje"].dt.normalize().nunique()
    departures = train.groupby("origen_id").size()
    arrivals = train.groupby("destino_id").size()
    flows = pd.concat([departures.rename("dep"), arrivals.rename("arr")], axis=1).fillna(0)
    profile["log_daily_departures"] = np.log1p(departures / n_days)
    profile["net_flow_ratio"] = (flows["arr"] - flows["dep"]) / (flows["arr"] + flows["dep"])
    profile["weekend_share"] = train[is_we].groupby("origen_id").size() / departures
    profile["median_duration_min"] = train.groupby("origen_id")["duracion_min"].median()
    profile = profile.fillna(0)
    profile["train_departures"] = departures
    profile.index.name = "station_id"

    out = stations.merge(profile.reset_index(), on="station_id", how="inner")
    # Estaciones con muy pocos viajes en entrenamiento tienen perfiles ruidosos.
    return out[out["train_departures"] >= P3_MIN_DEPARTURES].reset_index(drop=True)


def p3_feature_columns(df: pd.DataFrame) -> list[str]:
    """Columnas de perfil de uso que se usan para el clustering de P3."""
    hourly = [c for c in df.columns if c.startswith(P3_PROFILE_PREFIXES)]
    return hourly + P3_SCALAR_FEATURES


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@app.command()
def main():
    """Construye y guarda en Parquet los datasets de P1, P2 y P3."""
    from mibici.dataset import load_stations, load_trips

    trips, stations = load_trips(), load_stations()
    logger.info(f"{len(trips):,} viajes y {len(stations)} estaciones cargados")
    for name, builder, path in [
        ("P1", build_p1_dataset, P1_DATASET_PATH),
        ("P2", build_p2_dataset, P2_DATASET_PATH),
        ("P3 (train)", build_p3_dataset, P3_DATASET_PATH),
        ("P3 (test)", lambda t, s: build_p3_dataset(t, s, split="test"), P3_TEST_DATASET_PATH),
    ]:
        df = builder(trips, stations)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.success(f"{name}: {len(df):,} filas -> {path.name}")


if __name__ == "__main__":
    app()
