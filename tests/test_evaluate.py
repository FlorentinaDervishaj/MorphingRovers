"""Kontrollon qe evaluate() jep te njejtin rezultat si udp.fitness() i ESA-s.
I ngadalshem (~1 min), sepse ngarkon simulatorin dhe vlereson 2 here me 30 skenare."""
import pytest
from src.evaluate import evaluate, ALL_SCENARIOS, SUBSET, udp


def test_full_evaluation_matches_esa():
    x = udp.example()
    actual = evaluate(x, ALL_SCENARIOS, fix=False)
    expected = udp.fitness(x)[0]
    # ESA mbledh ne float32 (~7 shifra te sakta), ne ne float64, prandaj lejojme 1e-5
    assert actual == pytest.approx(expected, abs=1e-5)


def test_subset_has_one_scenario_per_map():
    assert len(SUBSET) == 6
    assert sorted(m for m, _ in SUBSET) == list(range(6))