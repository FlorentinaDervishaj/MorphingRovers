"""Testet per src/chromosome.py. Ekzekuto nga folderi kryesor: python -m pytest"""
import os
import numpy as np
import pytest
from src import chromosome as ch

EXAMPLE = os.path.join("data", "spoc2", "morphing", "example_rover.npy")


def test_lengths_add_up():
    assert ch.NUM_MASK_PARAMS == 484
    assert ch.NUM_NN_PARAMS == 18642
    assert ch.TOTAL_LENGTH == 19133
    assert ch.MASK_CENTRES == [60, 181, 302, 423]


def test_split_merge_roundtrip():
    x = np.random.default_rng(0).uniform(-100, 100, ch.TOTAL_LENGTH)
    p = ch.split(x)
    assert p["masks"].shape == (4, 11, 11)
    assert p["biases"].size == 146 and p["weights"].size == 18496 and p["integers"].size == 7
    assert np.array_equal(ch.merge(**p), x)


def test_set_masks_changes_only_masks():
    x = np.random.default_rng(1).uniform(-100, 100, ch.TOTAL_LENGTH)
    y = ch.set_masks(x, np.zeros(484))
    assert np.all(y[:484] == 0)
    assert np.array_equal(y[484:], x[484:])
    assert not np.all(x[:484] == 0)         


def test_repair_respects_bounds():
    x = np.full(ch.TOTAL_LENGTH, 500.0)
    x[ch.MASK_CENTRES] = -5
    y = ch.repair(x)
    lb, ub = ch.bounds()
    assert np.all(y >= lb) and np.all(y <= ub)
    assert np.all(y[ch.INTEGERS] == np.round(y[ch.INTEGERS]))


def test_wrong_length_raises():
    with pytest.raises(ValueError):
        ch.split(np.zeros(100))


@pytest.mark.skipif(not os.path.exists(EXAMPLE), reason="example_rover.npy mungon")
def test_example_rover_is_valid():
    x = np.load(EXAMPLE)
    assert x.shape == (ch.TOTAL_LENGTH,)
    assert np.array_equal(ch.merge(**ch.split(x)), x)


def test_output_layer_indices():
    assert ch.OUTPUT_LAYER_IDX.size == 82
    assert list(ch.OUTPUT_BIAS_IDX) == [628, 629]
    assert ch.OUTPUT_WEIGHT_IDX[0] == 19046 and ch.OUTPUT_WEIGHT_IDX[-1] == 19125
    assert ch.free_indices("masks+output").size == 566
