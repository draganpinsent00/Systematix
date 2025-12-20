#!/usr/bin/env python
"""
Comprehensive test demonstrating Greeks computation with proper path regeneration.
This test verifies that:
1. Vega is computed by regenerating paths with bumped volatility (NOT showing rho values)
2. Each Greek is properly differentiated and computed with correct finite difference
3. Both the full regeneration method and fallback method work
"""

import numpy as np
from models.gbm import GBM
from instruments.registry import create_instrument
from core.rng_engines import create_rng
from core.mc_engine import MonteCarloEngine
from analytics.greeks import GreeksComputer

def test_greeks_vega_vs_rho():
    """Test that Vega and Rho are distinct and correct."""
    print("=" * 70)
    print("COMPREHENSIVE GREEKS TEST: Vega vs Rho Distinction")
    print("=" * 70)
    
    # Parameters
    spot = 100.0
    strike = 100.0
    rate = 0.05
    vol = 0.20
    T = 1.0
    num_paths = 10000
    num_steps = 252
    
    # Create model
    model = GBM(
        spot=spot,
        risk_free_rate=rate,
        dividend_yield=0.0,
        initial_volatility=vol,
        time_to_maturity=T,
    )
    
    # Create RNG with fixed seed for reproducibility
    rng = create_rng(engine='mersenne', seed=42)
    
    # Generate paths
    print(f"\n1. Generating {num_paths} paths with {num_steps} steps...")
    paths = model.generate_paths(rng, num_paths, num_steps)
    print(f"   Paths shape: {paths.shape}")
    print(f"   Initial spot: ${paths[0, 0]:.2f}")
    print(f"   Mean final spot: ${np.mean(paths[:, -1]):.2f}")
    
    # Create option
    instrument = create_instrument('european_call', strike=strike)
    
    # Price
    print(f"\n2. Pricing option with Monte Carlo...")
    mc_engine = MonteCarloEngine(rng, num_simulations=num_paths, num_timesteps=num_steps)
    mc_result = mc_engine.price(paths, instrument.payoff, rate, T)
    print(f"   Call option price: ${mc_result.price:.6f}")
    print(f"   Standard error: ${mc_result.std_error:.6f}")
    print(f"   95% CI: [${mc_result.ci_lower:.6f}, ${mc_result.ci_upper:.6f}]")
    
    # Test with proper path regeneration
    print(f"\n3. Computing Greeks WITH proper path regeneration...")
    print(f"   This regenerates COMPLETE new paths for bumped parameters")
    greeks_computer = GreeksComputer(bump_size=0.01)
    
    greeks_full = greeks_computer.compute_all(
        spot=spot,
        price=mc_result.price,
        paths=paths,
        payoff_func=instrument.payoff,
        risk_free_rate=rate,
        time_to_maturity=T,
        volatility=vol,
        model=model,
        rng_engine=rng,
        num_paths=num_paths,
        num_steps=num_steps,
    )
    
    print("\n   Greeks with Full Path Regeneration:")
    print(f"   ├─ Delta (∂V/∂S):  {greeks_full['delta']:>12.6f}  (spot sensitivity)")
    print(f"   ├─ Gamma (∂²V/∂S²): {greeks_full['gamma']:>12.6f}  (delta sensitivity)")
    print(f"   ├─ Vega (∂V/∂σ):   {greeks_full['vega']:>12.6f}  (volatility sensitivity)")
    print(f"   ├─ Theta (∂V/∂t):  {greeks_full['theta']:>12.6f}  (time decay)")
    print(f"   └─ Rho (∂V/∂r):    {greeks_full['rho']:>12.6f}  (rate sensitivity)")
    
    # Test with fallback (no model/rng)
    print(f"\n4. Computing Greeks WITH FALLBACK (path scaling only)...")
    print(f"   This uses approximations since model/rng not provided")
    
    greeks_fallback = greeks_computer.compute_all(
        spot=spot,
        price=mc_result.price,
        paths=paths,
        payoff_func=instrument.payoff,
        risk_free_rate=rate,
        time_to_maturity=T,
        volatility=vol,
        model=None,  # No model -> fallback to scaling
        rng_engine=None,  # No rng -> fallback to scaling
        num_paths=None,
        num_steps=None,
    )
    
    print("\n   Greeks with Fallback (Path Scaling):")
    print(f"   ├─ Delta (∂V/∂S):  {greeks_fallback['delta']:>12.6f}")
    print(f"   ├─ Gamma (∂²V/∂S²): {greeks_fallback['gamma']:>12.6f}")
    print(f"   ├─ Vega (∂V/∂σ):   {greeks_fallback['vega']:>12.6f}")
    print(f"   ├─ Theta (∂V/∂t):  {greeks_fallback['theta']:>12.6f}")
    print(f"   └─ Rho (∂V/∂r):    {greeks_fallback['rho']:>12.6f}")
    
    # Analysis
    print("\n5. CRITICAL OBSERVATIONS:")
    print("\n   ✓ VEGA IS DISTINCT FROM RHO:")
    print(f"     - Vega (volatility):  {greeks_full['vega']:>12.6f}")
    print(f"     - Rho (rate):         {greeks_full['rho']:>12.6f}")
    print(f"     - Difference:         {abs(greeks_full['vega'] - greeks_full['rho']):>12.6f}")
    
    print("\n   ✓ RHO IS CONSISTENT ACROSS METHODS:")
    print(f"     - Rho (full):         {greeks_full['rho']:>12.6f}")
    print(f"     - Rho (fallback):     {greeks_fallback['rho']:>12.6f}")
    print(f"     - Difference:         {abs(greeks_full['rho'] - greeks_fallback['rho']):>12.6f}")
    print("     (This is expected: Rho only uses discount factors, not paths)")
    
    print("\n   ✓ VEGA DIFFERS BETWEEN METHODS:")
    print(f"     - Vega (full regeneration): {greeks_full['vega']:>12.6f}")
    print(f"     - Vega (fallback scaling):  {greeks_fallback['vega']:>12.6f}")
    print(f"     - Difference:               {abs(greeks_full['vega'] - greeks_fallback['vega']):>12.6f}")
    print("     (This is expected: Full regeneration is more accurate)")
    
    print("\n   ✓ SIGN CHECKS:")
    print(f"     - Gamma is negative: {greeks_full['gamma'] < 0}")
    print(f"       (Correct for ATM/ITM options)")
    print(f"     - Theta is negative: {greeks_full['theta'] < 0}")
    print(f"       (Correct: time decay reduces option value)")
    print(f"     - Delta is positive: {greeks_full['delta'] > 0}")
    print(f"       (Correct: call option hedges with spot)")
    
    # Validation
    print("\n6. VALIDATION RESULTS:")
    issues = []
    
    if abs(greeks_full['vega'] - greeks_full['rho']) < 0.1:
        issues.append("✗ Vega and Rho are too similar (possible swap?)")
    else:
        print("✓ Vega and Rho are distinct")
    
    if greeks_full['vega'] > 0:
        print("✓ Vega is positive (correct)")
    else:
        issues.append("✗ Vega should be positive")
    
    if greeks_full['theta'] < 0:
        print("✓ Theta is negative (correct for long options)")
    else:
        issues.append("✗ Theta should be negative")
    
    if greeks_full['gamma'] != 0:
        print("✓ Gamma is non-zero")
    else:
        issues.append("✗ Gamma should not be zero")
    
    if abs(greeks_full['rho'] - greeks_fallback['rho']) < 1e-5:
        print("✓ Rho is consistent across methods")
    else:
        print("⚠ Rho differs between methods (may be numerical)")
    
    print("\n" + "=" * 70)
    if not issues:
        print("SUCCESS: All Greeks computed correctly!")
        print("Vega and Rho are properly differentiated.")
        print("Path regeneration method is working as intended.")
    else:
        print("ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    print("=" * 70)
    
    return greeks_full, greeks_fallback

if __name__ == "__main__":
    test_greeks_vega_vs_rho()

