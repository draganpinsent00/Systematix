import numpy as np
from pricing import price_american_lsm, bs_price
from payoffs import EuropeanCall


def test_lsm_vs_bs_for_non_dividend_call():
    # For non-dividend GBM, American call == European call; LSM should be close to BS
    S0 = 100.0
    K = 100.0
    r = 0.01
    sigma = 0.2
    T = 1.0
    steps = 50
    n_paths = 4000

    payoff_func = lambda S: np.maximum(S - K, 0.0)
    res = price_american_lsm(payoff_func, S0, r, sigma, T, steps, n_paths, seed=42, antithetic=True, degree=2)
    bs = bs_price(S0, K, r, sigma, T, option='call')
    assert abs(res['price'] - bs) < 0.5

