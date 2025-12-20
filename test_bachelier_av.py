"""
Test if sqrt(2) factor is from antithetic variates.
"""
import numpy as np
from models.bachelier import Bachelier
from core.rng_engines import PCG64RNG

S = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2

# Test WITH antithetic variates (default)
bach_with_av = Bachelier(spot=S, risk_free_rate=r, dividend_yield=0.0, initial_volatility=sigma, time_to_maturity=T)
rng1 = PCG64RNG(seed=42)
paths_with_av = bach_with_av.generate_paths(rng1, 10000, 252)
std_with_av = np.std(paths_with_av[:, -1])

# Compute call price
call_payoff_av = np.maximum(paths_with_av[:, -1] - K, 0)
call_price_av = np.mean(call_payoff_av) * np.exp(-r * T)

print(f"With Antithetic Variates (default):")
print(f"  Standard deviation of final prices: {std_with_av:.6f}")
print(f"  Call price: {call_price_av:.6f}")
print(f"  Expected std: {sigma * np.sqrt(T):.6f}")

# Now let's manually disable antithetic variates by examining the code
# Actually, I can't easily disable it without modifying the class
# But I can see what the code does:
# It takes Z, then does: Z = np.vstack([Z[:half], -Z[:half]])
# This pairs each Z with -Z, which should reduce variance but not change the mean or std

# The variance reduction from antithetic variates comes from averaging
# But the code doesn't seem to average - it just stacks them

# Actually wait - let me look at the exact transformation
# In the loop:
# paths[:, t + 1] = paths[:, t] + self.drift * dt + self.sigma * sqrt_dt * Z_transformed[:, t]

# With antithetic variates, we get twice as many paths:
# half from Z, half from -Z

# The variance in dS should still be (sigma * sqrt_dt)^2 = sigma^2 * dt
# And the std of dS should be sigma * sqrt(dt)

# So the std of S_T over T timesteps should be sigma * sqrt(T)

# Wait, I think I see it! Let me check the generation loop logic

print("\nDetailed trace of path generation:")
rng = PCG64RNG(seed=42)

# Manually generate to see
num_paths = 100
num_steps = 1
dt = T / num_steps
sqrt_dt = np.sqrt(dt)

Z = rng.standard_normal((num_paths, num_steps))
print(f"Original Z shape: {Z.shape}")
print(f"Original Z mean: {np.mean(Z):.6f}, std: {np.std(Z):.6f}")

# Apply antithetic
half = num_paths // 2
Z_av = np.vstack([Z[:half], -Z[:half]])
num_paths_av = half * 2
print(f"After antithetic Z shape: {Z_av.shape}")
print(f"After antithetic Z mean: {np.mean(Z_av):.6f}, std: {np.std(Z_av):.6f}")

# So antithetic variates don't change the distribution properties
# They just make paths more correlated/paired

# The issue must be elsewhere...

# OH WAIT. Let me check the InnovationTransform!
from core.rng_distributions import InnovationTransform

transform = InnovationTransform("normal", 3.0, use_sobol=False)
Z_test = np.random.standard_normal((100, 1))
Z_trans = transform.transform(Z_test)

print(f"\nInnovationTransform check:")
print(f"Input Z shape: {Z_test.shape}, mean: {np.mean(Z_test):.6f}, std: {np.std(Z_test):.6f}")
print(f"Output Z shape: {Z_trans.shape}, mean: {np.mean(Z_trans):.6f}, std: {np.std(Z_trans):.6f}")

# Is the transform scaling the data?

