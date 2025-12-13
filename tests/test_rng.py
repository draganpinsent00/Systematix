import numpy as np
from rng import get_rng


def test_numpy_rng_reproducible():
    r1 = get_rng('numpy', seed=123)
    a1 = r1.standard_normal((100, 3))
    r2 = get_rng('numpy', seed=123)
    a2 = r2.standard_normal((100, 3))
    assert np.allclose(a1, a2)


def test_numpy_rng_different_seeds():
    r1 = get_rng('numpy', seed=1)
    r2 = get_rng('numpy', seed=2)
    a1 = r1.standard_normal((50,))
    a2 = r2.standard_normal((50,))
    # probabilistic but seeds differ — arrays should not be identical
    assert not np.allclose(a1, a2)
