"""Kontrollon me kodin e ESA-s qe indekset e shtreses se daljes jane te sakta.
I ngadalshem (~20 s), sepse ngarkon simulatorin."""
import numpy as np
from src import chromosome as ch
import evaluation_code as ec


def test_output_layer_matches_esa_network():
    x = np.asarray(ec.udp.example(), dtype=float).copy()
    x[ch.OUTPUT_BIAS_IDX] = [11.0, 22.0]
    x[ch.OUTPUT_WEIGHT_IDX] = np.arange(80) / 100.0
    net = ec.Rover(x).Control
    assert np.allclose(net.output.bias.numpy(), [11.0, 22.0])
    assert np.allclose(net.output.weight.numpy().ravel(), np.arange(80) / 100.0, atol=1e-6)
