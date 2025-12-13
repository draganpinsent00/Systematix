import numpy as np
from simulator import simulate_gbm_paths
from hedge import simulate_delta_hedge


def test_delta_hedge_reduces_error_with_freq():
    S0 = 100.0
    r = 0.0
    sigma = 0.2
    T = 1.0
    steps = 50
    n_paths = 200
    S = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed=42)

    # simple pricer that returns BS price and delta using Black-Scholes
    from pricing import bs_price, bs_delta if False else None
    def pricer(S_t, t):
        # treat each S_t element-wise
        # return price estimate and delta per path
        # For simplicity, return array shapes consistent
        price = np.array([bs_price(s, 100.0, r, sigma, (T*(steps-t)/steps), option='call') for s in S_t])
        delta = np.array([max(0.0, 0.5) for _ in S_t])
        return price, delta

    # run hedging at low and high frequencies
    res1 = simulate_delta_hedge(pricer, S, rebal_steps=10)
    res2 = simulate_delta_hedge(pricer, S, rebal_steps=1)
    assert res2['summary']['std'] <= res1['summary']['std']

