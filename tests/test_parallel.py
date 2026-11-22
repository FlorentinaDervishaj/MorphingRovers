"""Kontrollon qe vleresimi paralel jep te njejtat rezultate si ai i zakonshem.
I ngadalshem (~1 min): çdo worker ngarkon simulatorin."""
import numpy as np
from src import chromosome as ch
from src.evaluate import evaluate, udp, SUBSET
from src.parallel import ParallelEvaluator


def test_parallel_matches_sequential():
    base = np.asarray(udp.example(), dtype=float)
    rng = np.random.default_rng(0)
    xs = [base, ch.set_masks(base, ch.get_masks(base) + rng.normal(0, 0.1, ch.NUM_MASK_PARAMS))]
    sequential = [evaluate(x, SUBSET) for x in xs]
    with ParallelEvaluator(workers=2) as pe:
        parallel = pe.map(xs, SUBSET)
    assert np.allclose(sequential, parallel)
