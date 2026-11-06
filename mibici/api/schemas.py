"""Esquemas de entrada y salida de la API (Pydantic).

Cada pregunta tiene su propio registro con las variables de la firma del modelo ``champion``
(``P1_FEATURES``, ``P2_FEATURES`` y ``p3_feature_columns``), sus tipos y sus rangos válidos.
Las variables que no pertenecen al modelo se rechazan (``extra="forbid"``) para que un error de
escritura en el nombre de una columna no pase desapercibido.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from mibici.config import P2_BLOCKS_PER_DAY

MAX_RECORDS = 1000  # registros por petición; lotes mayores van por make predict (por lotes)
PROFILE_SUM_TOLERANCE = 0.01

Count = Field(ge=0, description="Conteo o promedio de viajes (no negativo).")
Flag = Field(ge=0, le=1, description="Indicador binario: 0 o 1.")
Share = Field(ge=0, le=1, description="Fracción entre 0 y 1.")


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StationCalendar(Record):
    """Variables de calendario y de la estación que comparten P1 y P2."""

    dow: int = Field(ge=0, le=6, description="Día de la semana: lunes = 0 ... domingo = 6.")
    month: int = Field(ge=1, le=12, description="Mes del año.")
    is_weekend: int = Flag
    is_holiday: int = Flag
    capacity: int = Field(gt=0, le=200, description="Anclajes de la estación.")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class P1Record(StationCalendar):
    """Una estación-día: se predicen los viajes que saldrán de la estación ese día."""

    departures_lag1: float = Count
    departures_lag2: float = Count
    departures_lag7: float = Count
    departures_lag14: float = Count
    departures_roll7_mean: float = Count
    departures_roll28_mean: float = Count
    arrivals_lag1: float = Count
    arrivals_lag7: float = Count
    arrivals_roll7_mean: float = Count


class P2Record(StationCalendar):
    """Una estación y un bloque de 3 horas: se predice si habrá riesgo de desbalance."""

    block: int = Field(
        ge=0, le=P2_BLOCKS_PER_DAY - 1, description="Bloque de 3 h desde las 06:00 (0 a 5)."
    )
    net_flow_lag1: float = Field(description="Llegadas - salidas del bloque anterior.")
    net_flow_lag6: float = Field(description="Llegadas - salidas de hace 6 bloques (un día).")
    net_flow_lag42: float = Field(description="Llegadas - salidas de hace 42 bloques (semana).")
    departures_lag1: float = Count
    departures_lag6: float = Count
    departures_lag42: float = Count
    risk_lag6: float = Share
    risk_lag42: float = Share
    abs_net_flow_same_block_4w_mean: float = Count


HOURS = range(24)
WEEKDAY_PROFILE = [f"dep_wd_h{h:02d}" for h in HOURS]
WEEKEND_PROFILE = [f"dep_we_h{h:02d}" for h in HOURS]


class P3Base(Record):
    """Perfil de uso de una estación: se predice su grupo (cluster)."""

    log_daily_departures: float = Field(ge=0, description="log1p de las salidas diarias.")
    net_flow_ratio: float = Field(ge=-1, le=1, description="(llegadas - salidas) / total.")
    weekend_share: float = Share
    median_duration_min: float = Field(gt=0, description="Duración mediana de los viajes.")

    @model_validator(mode="after")
    def profiles_sum_to_one(self):
        """Cada perfil horario (hábil y fin de semana) reparte el 100 % de las salidas."""
        for label, columns in (("dep_wd_h*", WEEKDAY_PROFILE), ("dep_we_h*", WEEKEND_PROFILE)):
            total = sum(getattr(self, c) for c in columns)
            if abs(total - 1) > PROFILE_SUM_TOLERANCE:
                raise ValueError(f"las 24 columnas {label} deben sumar 1, suman {total:.4f}")
        return self


P3Record = create_model(
    "P3Record",
    __base__=P3Base,
    __doc__=P3Base.__doc__,
    **{c: (float, Field(ge=0, le=1)) for c in WEEKDAY_PROFILE + WEEKEND_PROFILE},
)


def request_model(name: str, record: type[Record]) -> type[BaseModel]:
    return create_model(
        name,
        records=(list[record], Field(min_length=1, max_length=MAX_RECORDS)),
    )


P1Request = request_model("P1Request", P1Record)
P2Request = request_model("P2Request", P2Record)
P3Request = request_model("P3Request", P3Record)


class PredictionResponse(BaseModel):
    question: Literal["p1", "p2", "p3"]
    model: str = Field(description="Nombre del modelo en el Model Registry.")
    alias: str
    version: str = Field(description="Versión del Model Registry que respondió.")
    source: Literal["registry", "packaged"] = Field(
        description="registry: descargado del servidor MLflow; packaged: copiado en la imagen."
    )
    total_predictions: int = Field(ge=0)


class P1Response(PredictionResponse):
    predictions: list[Annotated[float, Field(ge=0)]] = Field(
        description="Viajes esperados que salen de la estación ese día."
    )


class P2Response(PredictionResponse):
    predictions: list[Literal[0, 1]] = Field(description="1 = riesgo de desbalance.")


class P3Response(PredictionResponse):
    predictions: list[Annotated[int, Field(ge=0)]] = Field(
        description="Grupo de uso asignado a cada estación."
    )
