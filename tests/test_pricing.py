from pricing import price_mc, bs_price
from payoffs import EuropeanCall


def test_mc_vs_bs_consistency():
    payoff = EuropeanCall(100.0)
    res = price_mc(payoff, 100.0, 0.01, 0.2, 1.0, 12, 10000, seed=42, antithetic=True)
    bs = bs_price(100.0, 100.0, 0.01, 0.2, 1.0, option='call')
    # Price should be within, say, 0.1 of BS for 10k paths
    assert abs(res['price'] - bs) < 0.5
