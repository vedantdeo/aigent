"""`facts.formulas`: each library formula's arithmetic, its refusals, and the library's shape."""

from __future__ import annotations

import inspect

import pytest

from aigent.facts.formulas import FORMULAS


@pytest.mark.parametrize(
    ("name", "inputs", "expected"),
    [
        pytest.param("growth", {"current": 115.0, "previous": 100.0}, 0.15, id="growth"),
        pytest.param("growth", {"current": 80.0, "previous": 100.0}, -0.2, id="a fall"),
        pytest.param(
            "growth", {"current": -50.0, "previous": -100.0}, 0.5, id="a loss halved is growth"
        ),
        pytest.param("ratio", {"numerator": 800.0, "denominator": 1000.0}, 0.8, id="ratio"),
        pytest.param("difference", {"minuend": 10.0, "subtrahend": 4.0}, 6.0, id="difference"),
        pytest.param("total", {"first": 10.0, "second": 4.0}, 14.0, id="total"),
    ],
)
def test_each_formula_computes(name: str, inputs: dict[str, float], expected: float) -> None:
    assert FORMULAS[name].compute(**inputs) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("name", "inputs"),
    [
        pytest.param("growth", {"current": 1.0, "previous": 0.0}, id="growth from zero"),
        pytest.param("ratio", {"numerator": 1.0, "denominator": 0.0}, id="a zero denominator"),
    ],
)
def test_a_zero_base_is_refused(name: str, inputs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="is zero"):
        FORMULAS[name].compute(**inputs)


@pytest.mark.parametrize("name", sorted(FORMULAS))
def test_a_formula_s_variables_are_its_function_s_parameters(name: str) -> None:
    formula = FORMULAS[name]
    assert formula.name == name
    assert tuple(inspect.signature(formula.compute).parameters) == formula.variables
    assert formula.unit in ("pure", "same") and formula.description


def test_the_library_is_held_in_name_order() -> None:
    assert list(FORMULAS) == sorted(FORMULAS)
