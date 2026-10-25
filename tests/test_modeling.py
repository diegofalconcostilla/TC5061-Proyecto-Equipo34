import pytest

from mibici.config import P1_MODEL_PARAMS, P2_MODEL_PARAMS, P3_MODEL_PARAMS
from mibici.modeling.common import run_name_with_changes
from mibici.modeling.params import cli_text, expand_grid, resolve_params
from mibici.modeling.search import combine
from mibici.modeling.train import MODELS

DEFAULTS = {"learning_rate": 0.05, "max_iter": 500, "early_stopping": True, "class_weight": None}


def test_empty_params_keep_defaults():
    assert resolve_params(DEFAULTS, "") == DEFAULTS


def test_params_are_cast_to_the_default_type():
    hp = resolve_params(DEFAULTS, "learning_rate=0.1, max_iter=300,early_stopping=false")
    assert hp["learning_rate"] == 0.1
    assert hp["max_iter"] == 300 and isinstance(hp["max_iter"], int)
    assert hp["early_stopping"] is False


def test_none_and_text_values():
    assert resolve_params({"class_weight": "balanced"}, "class_weight=none")["class_weight"] is None
    assert resolve_params(DEFAULTS, "class_weight=balanced")["class_weight"] == "balanced"


def test_unknown_params_are_rejected():
    with pytest.raises(ValueError, match="learnig_rate"):
        resolve_params(DEFAULTS, "learnig_rate=0.1")


def test_invalid_number_is_rejected():
    with pytest.raises(ValueError, match="max_iter"):
        resolve_params(DEFAULTS, "max_iter=mucho")


def test_run_name_lists_only_changed_params():
    hp = resolve_params(DEFAULTS, "max_iter=300")
    assert run_name_with_changes("p1-hgb", hp, DEFAULTS) == "p1-hgb[max_iter=300]"
    assert run_name_with_changes("p1-hgb", DEFAULTS, DEFAULTS) == "p1-hgb"


@pytest.mark.parametrize(
    "question, params", [("p1", P1_MODEL_PARAMS), ("p2", P2_MODEL_PARAMS), ("p3", P3_MODEL_PARAMS)]
)
def test_model_defaults_come_from_config(question, params):
    for name in params:
        assert name in MODELS[question], f"{name} está en config.py pero no en la tabla"
        assert MODELS[question][name].defaults is params[name]
    for name, spec in MODELS[question].items():
        assert spec.stage in ("baseline", "candidate")
        if spec.defaults:
            assert name in params, f"{question}/{name} define defaults fuera de config.py"


def test_grid_expands_to_every_combination():
    assert expand_grid("learning_rate=0.03|0.1; max_iter=300|500") == [
        "learning_rate=0.03,max_iter=300",
        "learning_rate=0.03,max_iter=500",
        "learning_rate=0.1,max_iter=300",
        "learning_rate=0.1,max_iter=500",
    ]


@pytest.mark.parametrize("grid", ["", "learning_rate", "learning_rate=", "=0.1|0.2"])
def test_invalid_grid_is_rejected(grid):
    with pytest.raises(ValueError):
        expand_grid(grid)


def test_combination_overrides_fixed_params():
    hp = resolve_params(DEFAULTS, combine("max_iter=100,learning_rate=0.2", "learning_rate=0.1"))
    assert hp["max_iter"] == 100 and hp["learning_rate"] == 0.1


def test_cli_text_quita_comillas_de_valor_vacio():
    # En Windows, mlflow run pasa un parámetro vacío como "" literal.
    assert cli_text('""') == ""
    assert cli_text("''") == ""
    assert cli_text('"alpha=0.1"') == "alpha=0.1"
    assert cli_text("alpha=0.1") == "alpha=0.1"
