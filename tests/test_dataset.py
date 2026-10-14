import pandas as pd

from mibici.dataset import clean_trips, fix_mojibake, month_range, normalize_trip_columns


def test_month_range_inclusive():
    assert month_range("2025-11", "2026-02") == ["2025_11", "2025_12", "2026_01", "2026_02"]


def test_fix_mojibake():
    assert fix_mojibake("AÃ±o") == "Año"
    assert fix_mojibake("Año") == "Año"


def test_headers_with_different_encodings_map_to_same_columns():
    headers = [
        "Año_de_nacimiento",  # UTF-8
        "AÃ±o_de_nacimiento",  # UTF-8 leído como Latin-1
        "A}äe_nacimiento",  # encabezado corrupto de 2024-02
    ]
    for h in headers:
        df = pd.DataFrame(
            columns=["Viaje_Id", "Usuario_Id", "Genero", h, "Inicio_del_viaje",
                     "Fin_del_viaje", "Origen_Id", "Destino_Id"]
        )
        assert "anio_nacimiento" in normalize_trip_columns(df).columns


def test_clean_trips_drops_personal_data_and_invalid_rows():
    raw = pd.DataFrame(
        {
            "Viaje_Id": ["1", "2", "3", "3"],
            "Usuario_Id": ["10", "11", "12", "12"],
            "Genero": ["M", "F", None, None],
            "Año_de_nacimiento": ["1990", "1985", "2000", "2000"],
            "Inicio_del_viaje": ["2025-01-01 08:00:00", "2025-01-01 08:00:00",
                                 "2025-01-01 09:00:00", "2025-01-01 09:00:00"],
            "Fin_del_viaje": ["2025-01-01 08:15:00", "2025-01-01 08:00:20",
                              "2025-01-01 09:10:00", "2025-01-01 09:10:00"],
            "Origen_Id": ["5", "5", "6", "6"],
            "Destino_Id": ["7", "7", "8", "8"],
        }
    )
    clean, stats = clean_trips(raw)
    assert not {"usuario_id", "genero", "anio_nacimiento"} & set(clean.columns)
    assert clean["viaje_id"].tolist() == [1, 3]  # el 2 dura 20 s; el 3 está duplicado
    assert stats["dropped_duration"] == 1
    assert stats["dropped_duplicates"] == 1
