"""
Debug the Bachelier MC vs Theory discrepancy more carefully.

The issue: MC gives mean payoff = 0.1068, theory gives E[max(S_T - K, 0)] = 0.0773

For Bachelier: S_T ~ N(S_0 + (r-q)*T, sigma^2*T)

E[max(S_T - K, 0)] = (S_0 + (r-q)*T - K) * N(d) + sigma*sqrt(T) * n(d)

where d = (S_0 - K + (r-q)*T) / (sigma*sqrt(T))
"""

import numpy as np
from scipy.stats import norm

S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
q = 0.0
sigma = 0.2

# Under risk-neutral measure, drift = r (not r - q for dividend yield)
# Actually, with dividend yield q:
# Adjusted forward: F = S_0 * exp((r - q)*T)
# But for Bachelier (arithmetic), the mean is: S_0 + (r - q)*T

forward = S0 + (r - q) * T  # 100.05
print(f"Forward price (Bachelier): {forward:.6f}")

# The option value formula:
d = (S0 - K + (r - q) * T) / (sigma * np.sqrt(T))
print(f"d = {d:.6f}")

N_d = norm.cdf(d)
n_d = norm.pdf(d)

# E[max(S_T - K, 0)] before discounting
option_value_undiscounted = (forward - K) * N_d + sigma * np.sqrt(T) * n_d
print(f"\nE[max(S_T - K, 0)] undiscounted = {option_value_undiscounted:.6f}")

# After discounting
option_value = option_value_undiscounted * np.exp(-r * T)
print(f"E[max(S_T - K, 0)] * exp(-r*T) = {option_value:.6f}")

# Now let's generate paths and compute the empirical expectation
np.random.seed(42)
num_paths = 1000000
num_steps = 252

dt = T / num_steps
sqrt_dt = np.sqrt(dt)

# Generate ALL random numbers first
Z = np.random.randn(num_paths, num_steps)

# Apply antithetic variates
half = num_paths // 2
Z_av = np.vstack([Z[:half], -Z[:half]])
num_paths_actual = half * 2

# Generate paths exactly as in the code
paths = np.zeros((num_paths_actual, num_steps + 1))
paths[:, 0] = S0

for t in range(num_steps):
    paths[:, t + 1] = paths[:, t] + (r - q) * dt + sigma * sqrt_dt * Z_av[:, t]

final_S = paths[:, -1]

# Empirical mean and std at maturity
emp_mean = np.mean(final_S)
emp_std = np.std(final_S)

print(f"\nEmpirical distribution at maturity:")
print(f"  Mean: {emp_mean:.6f} (expected: {forward:.6f})")
print(f"  Std:  {emp_std:.6f} (expected: {sigma * np.sqrt(T):.6f})")

# Now compute E[max(S_T - K, 0)] empirically
payoffs = np.maximum(final_S - K, 0)
emp_option_value_undiscounted = np.mean(payoffs)
emp_option_value = emp_option_value_undiscounted * np.exp(-r * T)

print(f"\nEmpirical option values:")
print(f"  E[max(S_T - K, 0)] undiscounted: {emp_option_value_undiscounted:.6f}")
print(f"  E[max(S_T - K, 0)] discounted:   {emp_option_value:.6f}")

print(f"\nComparison:")
print(f"  Theory undiscounted: {option_value_undiscounted:.6f}")
print(f"  MC undiscounted:     {emp_option_value_undiscounted:.6f}")
print(f"  Ratio: {emp_option_value_undiscounted / option_value_undiscounted:.6f}")

# Let me check if maybe the issue is that the itm percentage doesn't match
itm_theory = N_d
itm_mc = np.mean(final_S > K)

print(f"\nITM percentage:")
print(f"  Theory: {itm_theory*100:.4f}%")
print(f"  MC:     {itm_mc*100:.4f}%")

# And check the intrinsic value contribution
intrinsic = (forward - K) * itm_mc  # Very rough
print(f"\nIntrinsic value (rough):")
print(f"  (forward - K) * P(ITM) = ({forward - K:.6f}) * {itm_mc:.6f} = {intrinsic:.6f}")

# Let me manually compute what the payoff should be
# For a truncated normal distribution
print(f"\nDetailed payoff calculation:")

# The theoretical formula is:
# E[max(S_T - K, 0)] = (F - K) * N(d) + sigma*sqrt(T) * n(d)
# where F = S_0 + (r-q)*T, d = (F - K) / (sigma * sqrt(T))

# But wait, I had d = (S_0 - K + (r-q)*T) / (sigma*sqrt(T))
# which is the same as (F - K) / (sigma*sqrt(T))

# So the formula should give the right answer
# Let me just compute it with different notation to be sure

F = S0 + (r - q) * T  # Forward
d_alt = (F - K) / (sigma * np.sqrt(T))
option_alt = (F - K) * norm.cdf(d_alt) + sigma * np.sqrt(T) * norm.pdf(d_alt)

print(f"  Using F = S0 + (r-q)*T = {F}")
print(f"  d = (F - K) / (sigma*sqrt(T)) = {d_alt}")
print(f"  Option = (F - K)*N(d) + sigma*sqrt(T)*n(d) = {option_alt:.6f}")

# Hmm, this should work. Let me check if the issue is numerical precision
# or if there's something wrong with how paths are generated

print(f"\nDebugging: Are the paths being generated correctly?")
print(f"  First 10 final prices: {final_S[:10]}")
print(f"  Percentiles of final prices:")
for p in [1, 5, 25, 50, 75, 95, 99]:
    print(f"    {p:2d}%: {np.percentile(final_S, p):.4f}")

# Check if paths are normally distributed
from scipy.stats import shapiro
# Use a sample since Shapiro test has limits
sample = final_S[::100]  # Every 100th
stat, pval = shapiro(sample[:5000])
print(f"\nShapiro-Wilk test on sample of paths (p-value): {pval:.6f}")
print(f"  Paths are normally distributed: {pval > 0.05}")

