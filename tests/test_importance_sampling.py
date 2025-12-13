import numpy as np
from pricing import price_mc_importance_sampling
from payoffs import EuropeanCall


def test_importance_sampling_runs():
    payoff = EuropeanCall(100.0)
    res = price_mc_importance_sampling(payoff, 100.0, 0.01, 0.2, 1.0, 12, 500, theta=0.1, seed=42)
    assert 'price' in res and 'stderr' in res and 'ess' in res
    assert isinstance(res['price'], float)
    assert res['n'] == 500

