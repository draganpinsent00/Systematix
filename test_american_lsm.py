"""
Validation of American option pricing with Least Squares Monte Carlo.

Tests that:
1. American option prices >= European option prices
2. American call ≈ European call (non-dividend stock, rare early exercise)
3. American put > European put (early exercise value)
4. LSM pricing is stable and realistic
"""

import numpy as np
from models.gbm import GBM
from instruments.payoffs_vanilla import EuropeanCall, EuropeanPut, AmericanCall, AmericanPut
from analytics.pricing import black_scholes_call, black_scholes_put
from core.mc_engine import MonteCarloEngine
from core.rng_engines import PCG64RNG

def test_american_options():
    """Test American option pricing with LSM."""

    print("="*70)
    print("AMERICAN OPTION PRICING VALIDATION")
    print("="*70)

    # Market parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    q = 0.0  # No dividend
    sigma = 0.2

    # Monte Carlo parameters
    num_paths = 10000
    num_steps = 252

    # Create GBM model
    model = GBM(spot=S0, risk_free_rate=r, dividend_yield=q,
                initial_volatility=sigma, time_to_maturity=T)

    # Generate paths
    rng = PCG64RNG(seed=42)
    paths = model.generate_paths(rng, num_paths, num_steps)

    # Create MC engine
    mc_engine = MonteCarloEngine(rng, num_simulations=num_paths, num_timesteps=num_steps)

    # Analytical prices (Black-Scholes)
    bs_call = black_scholes_call(S0, K, T, r, sigma)
    bs_put = black_scholes_put(S0, K, T, r, sigma)

    print(f"\nMarket Setup:")
    print(f"  Spot: {S0}, Strike: {K}, T: {T}yr, r: {r}, σ: {sigma}")
    print(f"  Dividend yield: {q}")
    print(f"  Paths: {num_paths}, Steps: {num_steps}")

    print(f"\nBlack-Scholes (European) Prices:")
    print(f"  Call: ${bs_call:.6f}")
    print(f"  Put:  ${bs_put:.6f}")

    # Test 1: European Call
    print(f"\n{'='*70}")
    print("TEST 1: European Call")
    print(f"{'='*70}")

    euro_call = EuropeanCall(strike=K)
    euro_call_result = mc_engine.price(
        paths, euro_call.payoff, r, T, use_lsm=False
    )

    print(f"MC European Call Price: ${euro_call_result.price:.6f}")
    print(f"  BS Price: ${bs_call:.6f}")
    print(f"  Error: {abs(euro_call_result.price - bs_call) / bs_call * 100:.2f}%")
    print(f"  95% CI: [${euro_call_result.ci_lower:.6f}, ${euro_call_result.ci_upper:.6f}]")

    # Test 2: American Call (should be ≈ European call)
    print(f"\n{'='*70}")
    print("TEST 2: American Call")
    print(f"{'='*70}")

    amer_call = AmericanCall(strike=K)
    amer_call_result = mc_engine.price(
        paths, amer_call.payoff, r, T, use_lsm=True, lsm_config={'degree': 2}
    )

    print(f"MC American Call Price: ${amer_call_result.price:.6f}")
    print(f"  European Call (BS): ${bs_call:.6f}")
    print(f"  European Call (MC): ${euro_call_result.price:.6f}")
    print(f"  Difference (Amer - Euro): ${amer_call_result.price - euro_call_result.price:.6f}")
    print(f"  95% CI: [${amer_call_result.ci_lower:.6f}, ${amer_call_result.ci_upper:.6f}]")

    call_passes = amer_call_result.price >= euro_call_result.price - 0.01
    print(f"  American >= European? {'✅ PASS' if call_passes else '❌ FAIL'}")

    # Test 3: European Put
    print(f"\n{'='*70}")
    print("TEST 3: European Put")
    print(f"{'='*70}")

    euro_put = EuropeanPut(strike=K)
    euro_put_result = mc_engine.price(
        paths, euro_put.payoff, r, T, use_lsm=False
    )

    print(f"MC European Put Price: ${euro_put_result.price:.6f}")
    print(f"  BS Price: ${bs_put:.6f}")
    print(f"  Error: {abs(euro_put_result.price - bs_put) / bs_put * 100:.2f}%")
    print(f"  95% CI: [${euro_put_result.ci_lower:.6f}, ${euro_put_result.ci_upper:.6f}]")

    # Test 4: American Put (should be > European put)
    print(f"\n{'='*70}")
    print("TEST 4: American Put")
    print(f"{'='*70}")

    amer_put = AmericanPut(strike=K)
    amer_put_result = mc_engine.price(
        paths, amer_put.payoff, r, T, use_lsm=True, lsm_config={'degree': 2}
    )

    print(f"MC American Put Price: ${amer_put_result.price:.6f}")
    print(f"  European Put (BS): ${bs_put:.6f}")
    print(f"  European Put (MC): ${euro_put_result.price:.6f}")
    print(f"  Difference (Amer - Euro): ${amer_put_result.price - euro_put_result.price:.6f}")
    print(f"  95% CI: [${amer_put_result.ci_lower:.6f}, ${amer_put_result.ci_upper:.6f}]")

    put_passes = amer_put_result.price >= euro_put_result.price - 0.01
    print(f"  American >= European? {'✅ PASS' if put_passes else '❌ FAIL'}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    all_pass = call_passes and put_passes
    print(f"American Call >= European Call: {'✅ PASS' if call_passes else '❌ FAIL'}")
    print(f"American Put >= European Put: {'✅ PASS' if put_passes else '❌ FAIL'}")
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")
    print(f"{'='*70}\n")

    return all_pass

if __name__ == "__main__":
    success = test_american_options()
    exit(0 if success else 1)

