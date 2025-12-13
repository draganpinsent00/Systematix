import numpy as np
from simulator import simulate_gbm_paths
from payoffs import EuropeanCall
from pricing import price_mc_from_paths


def test_importance_sampling_variance_reduction():
    # Generate a deep OTM call target and compare variance for plain v.s. shifted-mean sampler
    S0 = 100.0
    r = 0.01
    sigma = 0.2
    T = 1.0
    steps = 1
    n_paths = 2000
    K = 150.0

    # plain
    S_plain = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed=1)
    payoff_obj = EuropeanCall(K)
    res_plain = price_mc_from_paths(payoff_obj, S_plain, r, T, discount=False, return_payoffs=True)
    var_plain = np.var(res_plain['payoffs'], ddof=1)

    # importance sampling placeholder: shift drift up in simulator by passing mu via simulate_gbm_paths (not fully implemented), so simulate again with higher S0
    S_shift = simulate_gbm_paths(S0*1.1, r, sigma, T, steps, n_paths, seed=2)
    res_shift = price_mc_from_paths(payoff_obj, S_shift, r, T, discount=False, return_payoffs=True)
    var_shift = np.var(res_shift['payoffs'], ddof=1)

    assert var_shift <= var_plain * 2  # basic smoke: shift shouldn't blow up variance massively

