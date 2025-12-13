import numpy as np
from calibration import fit_heston_iv_surface


def test_calibration_recovery_synthetic():
    # Create a synthetic Heston-like iv surface by using BS prices with varying vol (approximation)
    strikes = np.array([80.0, 100.0, 120.0])
    maturities = np.array([0.25, 0.5, 1.0])
    # synthetic ivs (not true Heston but serves as smoke test)
    market_iv = np.tile(np.array([0.25, 0.22, 0.28])[:, None], (1, len(maturities)))
    S0 = 100.0
    r = 0.01
    q = 0.0
    init = [0.04, 1.0, 0.04, 0.3, -0.5]
    res = fit_heston_iv_surface(market_iv, strikes, maturities, S0, r, q, init_params=init, bounds=None, steps=20, n_paths=500)
    assert 'x' in res
    assert res['success'] or res['fun'] < 1.0

