import pytest
from src.evaluate import evaluate, ALL_SCENARIOS, SUBSET, udp


def test_full_evaluation_matches_esa():
    x = udp.example()
    actual = evaluate(x, ALL_SCENARIOS, fix=False)
    expected = udp.fitness(x)[0]
    assert actual == pytest.approx(expected, rel=0, abs=1e-9)


def test_subset_has_one_scenario_per_map():
    assert SUBSET == [(m, 0) for m in range(6)]