import numpy as np
from simulator import simulate_gbm_paths


def test_simulate_shapes_and_reproducibility():
    S1 = simulate_gbm_paths(100.0, 0.01, 0.2, 1.0, 12, 100, seed=123, antithetic=False)
    S2 = simulate_gbm_paths(100.0, 0.01, 0.2, 1.0, 12, 100, seed=123, antithetic=False)
    assert S1.shape == (100, 13)
    assert np.allclose(S1, S2)


def test_antithetic_pairs():
    S = simulate_gbm_paths(100.0, 0.01, 0.2, 1.0, 12, 101, seed=42, antithetic=True)
    # when antithetic used, number of returned paths equals requested
    assert S.shape[0] == 101
    # basic sanity: positive prices
    assert (S > 0).all()

