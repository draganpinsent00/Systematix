import numpy as np
from greeks import compute_greeks_fd
from payoffs import EuropeanCall


def test_greeks_fd_vs_bs():
    S0 = 100.0
    K = 100.0
    r = 0.01
    sigma = 0.2
    T = 0.5
    # price_fn closure for BS
    def price_fn(S):
        from pricing import bs_price
        return bs_price(S, K, r, sigma, T, option='call')
    g = compute_greeks_fd(price_fn, S0)
    assert abs(g['delta'] - 0.546) < 0.05
    assert 'gamma' in g

