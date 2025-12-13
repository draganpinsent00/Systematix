import numpy as np
from simulator import simulate_merton_paths, simulate_kou_paths


def test_merton_shapes():
    S = simulate_merton_paths(100.0, 0.01, 0.2, 0.5, -0.1, 0.25, 1.0, 12, 100, seed=42)
    assert S.shape == (100, 13)
    assert np.all(S > 0)


def test_kou_shapes():
    S = simulate_kou_paths(100.0, 0.01, 0.2, 0.5, 0.3, 1.5, 0.5, 1.0, 12, 100, seed=42)
    assert S.shape == (100, 13)
    assert np.all(S > 0)

