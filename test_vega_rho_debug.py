"""
Debug Vega and Rho computation for Heston and Merton models.

Issue: Heston Vega is 0.158 (too small)
Issue: Merton Gamma/Delta/Rho are unusual

Let me manually compute Vega for Heston to see what's happening.
"""

import numpy as np
from models.heston import Heston
from models.merton_jump import MertonJump
from core.rng_engines import PCG64RNG
from analytics.greeks import GreeksComputer

S = 100.0
K = 100.0
T = 1.0
r = 0.05
q = 0.0

# Heston test
heston = Heston(
    spot=S,
    risk_free_rate=r,
    dividend_yield=q,
    initial_volatility=0.2,
    time_to_maturity=T,
    kappa=2.0,
    theta=0.04,
    sigma=0.3,
    rho=-0.5
)

# Merton test
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

num_paths = 10000
num_steps = 252
rng_seed = 42

print("="*70)
print("HESTON VEGA ANALYSIS")
print("="*70)

# Generate base paths
rng_base = PCG64RNG(seed=rng_seed)
paths_base = heston.generate_paths(rng_base, num_paths, num_steps)
payoff_base = np.maximum(paths_base[:, -1] - K, 0)
price_base = np.mean(payoff_base) * np.exp(-r * T)

print(f"\nBase price: {price_base:.6f}")

# Manually compute Vega using _compute_vega_fd logic
bump = 1e-4
vol_base = 0.2

# Up vol
rng_up = np.random.default_rng(rng_seed)
heston_up = Heston(
    spot=S, risk_free_rate=r, dividend_yield=q,
    initial_volatility=vol_base + bump, time_to_maturity=T,
    kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5
)
paths_up = heston_up.generate_paths(rng_up, num_paths, num_steps)
payoff_up = np.maximum(paths_up[:, -1] - K, 0)
price_up = np.mean(payoff_up) * np.exp(-r * T)

# Down vol
rng_down = np.random.default_rng(rng_seed)
heston_down = Heston(
    spot=S, risk_free_rate=r, dividend_yield=q,
    initial_volatility=vol_base - bump, time_to_maturity=T,
    kappa=2.0, theta=0.04, sigma=0.3, rho=-0.5
)
paths_down = heston_down.generate_paths(rng_down, num_paths, num_steps)
payoff_down = np.maximum(paths_down[:, -1] - K, 0)
price_down = np.mean(payoff_down) * np.exp(-r * T)

print(f"\nVega bump: {bump}")
print(f"Price up (vol + bump):   {price_up:.6f}")
print(f"Price down (vol - bump): {price_down:.6f}")

# Vega = dPrice / dVol
# Using central difference: (price_up - price_down) / (2 * bump)
vega_manual = (price_up - price_down) / (2 * bump)

# But the code uses: (price_up - price_down) / (200 * bump)
# That's 100x different!
vega_code = (price_up - price_down) / (200 * bump)

print(f"\nVega computations:")
print(f"  Manual (2*bump): {vega_manual:.6f}")
print(f"  Code (200*bump): {vega_code:.6f}")
print(f"  Ratio: {vega_manual / vega_code:.1f}")

print(f"\n⚠️  AH-HA! The code divides by (200 * bump) instead of (2 * bump)!")
print(f"This is 100x too small, which explains the low Vega!")

print("\n" + "="*70)
print("MERTON RHO ANALYSIS")
print("="*70)

# Generate base paths
rng_base = PCG64RNG(seed=rng_seed)
paths_base = merton.generate_paths(rng_base, num_paths, num_steps)
payoff_base = np.maximum(paths_base[:, -1] - K, 0)
price_base = np.mean(payoff_base) * np.exp(-r * T)

print(f"\nBase price: {price_base:.6f}")

# Manually compute Rho
bump = 1e-4
r_base = 0.05

# Up rate
rng_up = np.random.default_rng(rng_seed)
merton_up = MertonJump(
    spot=S, risk_free_rate=r_base + bump, dividend_yield=q,
    initial_volatility=0.2, time_to_maturity=T,
    lambda_=0.5, mu_j=0.0, sigma_j=0.2
)
paths_up = merton_up.generate_paths(rng_up, num_paths, num_steps)
payoff_up = np.maximum(paths_up[:, -1] - K, 0)
price_up = np.mean(payoff_up) * np.exp(-(r_base + bump) * T)

# Down rate
rng_down = np.random.default_rng(rng_seed)
merton_down = MertonJump(
    spot=S, risk_free_rate=r_base - bump, dividend_yield=q,
    initial_volatility=0.2, time_to_maturity=T,
    lambda_=0.5, mu_j=0.0, sigma_j=0.2
)
paths_down = merton_down.generate_paths(rng_down, num_paths, num_steps)
payoff_down = np.maximum(paths_down[:, -1] - K, 0)
price_down = np.mean(payoff_down) * np.exp(-(r_base - bump) * T)

print(f"\nRate bump: {bump}")
print(f"Price up (r + bump):   {price_up:.6f}")
print(f"Price down (r - bump): {price_down:.6f}")

# Rho = dPrice / dr
rho_manual = (price_up - price_down) / (2 * bump)
rho_code = (price_up - price_down) / (200 * bump)

print(f"\nRho computations:")
print(f"  Manual (2*bump): {rho_manual:.6f}")
print(f"  Code (200*bump): {rho_code:.6f}")
print(f"  Ratio: {rho_manual / rho_code:.1f}")

print(f"\n⚠️  Same issue: code divides by (200 * bump) instead of (2 * bump)!")

