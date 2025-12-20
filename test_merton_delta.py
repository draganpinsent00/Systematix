"""
Deep dive into why Merton Delta > 1.

If Delta > 1, something is very wrong. Let me manually compute Delta
and other Greeks for Merton to see what's happening.
"""

import numpy as np
from models.merton_jump import MertonJump
from core.rng_engines import PCG64RNG

S = 100.0
K = 100.0
T = 1.0
r = 0.05
q = 0.0

# Create Merton
merton = MertonJump(
    spot=S,
    risk_free_rate=r,
    dividend_yield=q,
    initial_volatility=0.2,
    time_to_maturity=T,
    lambda_=0.5,
    mu_j=0.0,
    sigma_j=0.2
)

num_paths = 100000
num_steps = 252

# Base price
rng_base = PCG64RNG(seed=42)
paths_base = merton.generate_paths(rng_base, num_paths, num_steps)
payoff_base = np.maximum(paths_base[:, -1] - K, 0)
price_base = np.mean(payoff_base) * np.exp(-r * T)

print(f"Base call price: {price_base:.6f}")

# Manual Delta calculation using FINITE DIFFERENCE on SPOT ONLY
# (not regenerating paths, just scaling them)
bump = 0.01 * S  # 1% bump

paths_up = paths_base.copy()
paths_up *= (S + bump) / S

paths_down = paths_base.copy()
paths_down *= (S - bump) / S

payoff_up = np.maximum(paths_up[:, -1] - K, 0)
payoff_down = np.maximum(paths_down[:, -1] - K, 0)

price_up = np.mean(payoff_up) * np.exp(-r * T)
price_down = np.mean(payoff_down) * np.exp(-r * T)

delta_simple = (price_up - price_down) / (2 * bump)

print(f"\nDelta calculation (simple finite difference on spot):")
print(f"  Price at S + {bump:.2f}: {price_up:.6f}")
print(f"  Price at S - {bump:.2f}: {price_down:.6f}")
print(f"  Delta = (P_up - P_down) / (2*bump) = {delta_simple:.6f}")

# Now with path regeneration (as the Greeks code does)
merton_up = MertonJump(
    spot=S + bump,
    risk_free_rate=r,
    dividend_yield=q,
    initial_volatility=0.2,
    time_to_maturity=T,
    lambda_=0.5,
    mu_j=0.0,
    sigma_j=0.2
)

merton_down = MertonJump(
    spot=S - bump,
    risk_free_rate=r,
    dividend_yield=q,
    initial_volatility=0.2,
    time_to_maturity=T,
    lambda_=0.5,
    mu_j=0.0,
    sigma_j=0.2
)

rng_up = PCG64RNG(seed=42)
paths_up_regen = merton_up.generate_paths(rng_up, num_paths, num_steps)
payoff_up_regen = np.maximum(paths_up_regen[:, -1] - K, 0)
price_up_regen = np.mean(payoff_up_regen) * np.exp(-r * T)

rng_down = PCG64RNG(seed=42)
paths_down_regen = merton_down.generate_paths(rng_down, num_paths, num_steps)
payoff_down_regen = np.maximum(paths_down_regen[:, -1] - K, 0)
price_down_regen = np.mean(payoff_down_regen) * np.exp(-r * T)

delta_regen = (price_up_regen - price_down_regen) / (2 * bump)

print(f"\nDelta calculation (with path regeneration):")
print(f"  Price at S + {bump:.2f}: {price_up_regen:.6f}")
print(f"  Price at S - {bump:.2f}: {price_down_regen:.6f}")
print(f"  Delta = {delta_regen:.6f}")

# Check the paths themselves
print(f"\nPath statistics:")
print(f"  Base mean final: {np.mean(paths_base[:, -1]):.2f}")
print(f"  Up mean final: {np.mean(paths_up_regen[:, -1]):.2f}")
print(f"  Down mean final: {np.mean(paths_down_regen[:, -1]):.2f}")

# Sanity check: for an ATM call, Delta should be around 0.5
# For deep ITM, Delta -> 1
# For deep OTM, Delta -> 0

percent_itm_base = np.mean(paths_base[:, -1] > K) * 100
percent_itm_up = np.mean(paths_up_regen[:, -1] > K) * 100
percent_itm_down = np.mean(paths_down_regen[:, -1] > K) * 100

print(f"\nITM percentage:")
print(f"  Base: {percent_itm_base:.1f}%")
print(f"  Up: {percent_itm_up:.1f}%")
print(f"  Down: {percent_itm_down:.1f}%")

# The delta based on ITM percentage should be roughly right
delta_itm = (percent_itm_up - percent_itm_down) / 100 / (2 * bump / S)
print(f"  Delta (from ITM %): {delta_itm:.6f}")

print(f"\n⚠️  Check: Is Delta > 1 because the bump is too large?")
print(f"  Bump size: {bump / S * 100:.2f}% of spot")
print(f"  With {num_paths} paths, this might not be enough for accurate FD")

# Try with a slightly larger sample and no path regeneration
print(f"\nRetesting with path scaling (no regeneration):")
print(f"  Delta (scaling): {delta_simple:.6f}")

