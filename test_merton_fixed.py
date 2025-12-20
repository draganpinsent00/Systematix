"""
Test Merton Greeks with fixed Vega/Rho computation.
"""
import numpy as np
from models.merton_jump import MertonJump
from core.rng_engines import PCG64RNG
from analytics.greeks import GreeksComputer

# Test Merton with larger sample
merton = MertonJump(
    spot=100.0,
    risk_free_rate=0.05,
    dividend_yield=0.0,
    initial_volatility=0.2,
    time_to_maturity=1.0,
    lambda_=0.5,
    mu_j=0.0,
    sigma_j=0.2
)

num_paths = 50000
num_steps = 252
rng = PCG64RNG(seed=42)
paths = merton.generate_paths(rng, num_paths, num_steps)

# Compute call price
call_payoff = np.maximum(paths[:, -1] - 100.0, 0)
call_price = np.mean(call_payoff) * np.exp(-0.05 * 1.0)

# Compute Greeks
gc = GreeksComputer()
greeks = gc.compute_all(
    spot=100.0,
    price=call_price,
    paths=paths,
    payoff_func=lambda p: np.maximum(p[:, -1] - 100.0, 0),
    risk_free_rate=0.05,
    time_to_maturity=1.0,
    volatility=0.2,
    model=merton,
    rng_engine=rng,
    num_paths=num_paths,
    num_steps=num_steps
)

print('Merton Greeks (fixed Vega/Rho):')
print(f'  Price:  {call_price:.6f}')
print(f'  Delta:  {greeks["delta"]:.6f}  (valid: 0-1)')
print(f'  Gamma:  {greeks["gamma"]:.6f}  (valid: >=0)')
print(f'  Vega:   {greeks["vega"]:.6f}  (fixed from 100x error)')
print(f'  Theta:  {greeks["theta"]:.6f}')
print(f'  Rho:    {greeks["rho"]:.6f}  (fixed from 100x error)')

