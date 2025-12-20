"""
Test Bachelier path generation in detail.
"""
import numpy as np
from models.bachelier import Bachelier
from core.rng_engines import PCG64RNG
from scipy.stats import norm

def bachelier_analytical_call(S, K, T, r, sigma):
    """Analytical Bachelier call price."""
    d = (S - K + r * T) / (sigma * np.sqrt(T))
    return (S - K) * norm.cdf(d) + sigma * np.sqrt(T) * norm.pdf(d) * np.exp(-r * T)

S = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2

bach = Bachelier(spot=S, risk_free_rate=r, dividend_yield=0.0, initial_volatility=sigma, time_to_maturity=T)

print(f"Bachelier model parameters:")
print(f"  spot: {bach.spot}")
print(f"  drift (r - q): {bach.drift}")
print(f"  T: {bach.T}")
print(f"  sigma: {bach.sigma}")

# Generate small sample
rng = PCG64RNG(seed=42)
paths = bach.generate_paths(rng, 5, 1)
print(f"\nFirst 5 paths (1 step):")
print(f"  Initial: {paths[:, 0]}")
print(f"  Final:   {paths[:, 1]}")
print(f"  Increments: {paths[:, 1] - paths[:, 0]}")

# The issue is clear now:
# In Bachelier, dS = (r - q)*dt + sigma*sqrt(dt)*dW
# Expected drift over 1 year: (r - q) * 1 = 0.05
# So we expect paths to drift by ~0.05 on average

# But the call payoff max(S_T - K, 0) suggests paths should have
# more positive drift. Let's check with the formula.

# The problem is: when we compute paths[:, t+1] = paths[:, t] + self.drift * dt + ...
# We get a drift of only 0.05 per year when we should get (r - q) * spot * dt

print("\nDetailed check of path generation:")
num_paths = 100000
num_steps = 252
paths = bach.generate_paths(rng, num_paths, num_steps)

final_prices = paths[:, -1]
call_payoffs = np.maximum(final_prices - K, 0)
call_price = np.mean(call_payoffs) * np.exp(-r * T)

theo_call = bachelier_analytical_call(S, K, T, r, sigma)

print(f"\nMean final price: {np.mean(final_prices):.6f}")
print(f"Std final price:  {np.std(final_prices):.6f}")
print(f"Expected drift over T: (r - q) * T = {(r - 0.0) * T:.6f}")
print(f"Expected final price: {S + (r - 0.0) * T:.6f}")

print(f"\nPricing comparison:")
print(f"Theoretical Bachelier call: {theo_call:.6f}")
print(f"MC Bachelier call:          {call_price:.6f}")
print(f"Ratio (theory/MC):          {theo_call/call_price:.6f}")

