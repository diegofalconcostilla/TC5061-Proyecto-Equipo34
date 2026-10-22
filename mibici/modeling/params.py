"""Hiperparámetros desde la línea de comandos: ``-P params="learning_rate=0.1,max_iter=300"``.

Los valores por defecto de cada modelo viven en ``config.py`` (``P1_MODEL_PARAMS``, etc.); aquí
solo se aplican los cambios pedidos, convertidos al tipo del valor por defecto.
"""

from __future__ import annotations

import itertools


def _convert(key: str, text: str, default):
    text = text.strip()
    if text.lower() in ("none", "null"):
        return None
    if isinstance(default, bool):
        if text.lower() in ("true", "1", "yes"):
            return True
        if text.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"{key} espera true o false, no {text!r}")
    if isinstance(default, (int, float)) or default is None:
        for cast in (int, float):
            try:
                return cast(text)
            except ValueError:
                pass
        if default is not None:
            raise ValueError(f"{key} espera un número, no {text!r}")
    return text


def parse_overrides(text: str) -> dict[str, str]:
    """``"a=1,b=x"`` -> ``{"a": "1", "b": "x"}``; texto vacío -> ``{}``."""
    overrides = {}
    for item in filter(None, (part.strip() for part in text.split(","))):
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"Formato inválido {item!r}; se espera clave=valor")
        overrides[key.strip()] = value
    return overrides


def expand_grid(text: str) -> list[str]:
    """Malla ``"a=1|2;b=x|y"`` -> una cadena ``params`` por combinación (``"a=1,b=x"``, ...)."""
    axes = []
    for item in filter(None, (part.strip() for part in text.split(";"))):
        key, sep, values = item.partition("=")
        options = [v.strip() for v in values.split("|") if v.strip()]
        if not sep or not key.strip() or not options:
            raise ValueError(f"Formato inválido {item!r}; se espera clave=v1|v2")
        axes.append([f"{key.strip()}={v}" for v in options])
    if not axes:
        raise ValueError("La malla está vacía; ejemplo: 'learning_rate=0.03|0.1;max_iter=300|500'")
    return [",".join(combo) for combo in itertools.product(*axes)]


def resolve_params(defaults: dict, text: str = "") -> dict:
    """Valores por defecto con los cambios de ``text`` aplicados; rechaza claves desconocidas."""
    overrides = parse_overrides(text)
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        valid = ", ".join(sorted(defaults)) or "ninguno"
        raise ValueError(f"Hiperparámetros desconocidos: {', '.join(unknown)}. Válidos: {valid}")
    return defaults | {k: _convert(k, v, defaults[k]) for k, v in overrides.items()}
