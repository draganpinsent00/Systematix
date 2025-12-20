#!/usr/bin/env python
"""Quick test of Greeks computation with proper path regeneration."""

import numpy as np
from models.gbm import GBM
from instruments.registry import create_instrument
from core.rng_engines import create_rng
from core.mc_engine import MonteCarloEngine
from analytics.greeks import GreeksComputer

# Parameters
spot = 100.0
strike = 100.0
rate = 0.05
vol = 0.20
T = 1.0
num_paths = 5000
num_steps = 252

print("Testing Greeks computation with proper path regeneration")
print("=" * 60)

# Create model
model = GBM(
    spot=spot,
    risk_free_rate=rate,
    dividend_yield=0.0,
    initial_volatility=vol,
    time_to_maturity=T,
)

# Create RNG
rng = create_rng(engine='mersenne', seed=42)

# Generate paths
print("Generating initial paths...")
paths = model.generate_paths(rng, num_paths, num_steps)
print(f"Paths shape: {paths.shape}")

# Create option
instrument = create_instrument('european_call', strike=strike)

# Price
mc_engine = MonteCarloEngine(rng, num_simulations=num_paths, num_timesteps=num_steps)
mc_result = mc_engine.price(paths, instrument.payoff, rate, T)
print(f"Option price: ${mc_result.price:.6f}")

# Compute Greeks WITH path regeneration
print("\nComputing Greeks with path regeneration...")
greeks_computer = GreeksComputer(bump_size=0.01)

greeks = greeks_computer.compute_all(
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

print("\nGreeks (with path regeneration):")
print(f"Delta:  {greeks['delta']:.6f}")
print(f"Gamma:  {greeks['gamma']:.6f}")
print(f"Vega:   {greeks['vega']:.6f}")
print(f"Theta:  {greeks['theta']:.6f}")
print(f"Rho:    {greeks['rho']:.6f}")

# Compute Greeks WITHOUT path regeneration (fallback)
print("\nComputing Greeks without path regeneration (fallback)...")
greeks_fallback = greeks_computer.compute_all(
    spot=spot,
    price=mc_result.price,
    paths=paths,
    payoff_func=instrument.payoff,
    risk_free_rate=rate,
    time_to_maturity=T,
    volatility=vol,
)

print("\nGreeks (fallback with path scaling):")
print(f"Delta:  {greeks_fallback['delta']:.6f}")
print(f"Gamma:  {greeks_fallback['gamma']:.6f}")
print(f"Vega:   {greeks_fallback['vega']:.6f}")
print(f"Theta:  {greeks_fallback['theta']:.6f}")
print(f"Rho:    {greeks_fallback['rho']:.6f}")

print("\n" + "=" * 60)
print("Test completed successfully!")

