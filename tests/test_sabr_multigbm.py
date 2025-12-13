import numpy as np
from simulator import simulate_sabr_paths


def test_sabr_shape():
    F = simulate_sabr_paths(100.0, 0.2, 0.5, -0.3, 0.1, 1.0, 12, 50, seed=42)
    assert F.shape == (50, 13)
    assert np.all(F >= 0)
