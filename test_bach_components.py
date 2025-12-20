"""
Identify the exact Bachelier scaling issue by component.
"""
import numpy as np
from scipy.stats import norm

# Manual path generation to isolate the issue
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
q = 0.0
sigma = 0.2
num_paths = 100000
num_steps = 252

# Standard drift
drift = r - q  # 0.05

# MANUAL PATH GENERATION (exactly as in Bachelier.generate_paths)
dt = T / num_steps
sqrt_dt = np.sqrt(dt)

# Generate random normals
rng = np.random.RandomState(42)
Z = rng.randn(num_paths, num_steps)

# Apply antithetic variates (as the code does)
half = num_paths // 2
Z = np.vstack([Z[:half], -Z[:half]])
num_paths_av = half * 2

# No transformation for "normal" distribution
Z_transformed = Z

# Generate paths
paths = np.zeros((num_paths_av, num_steps + 1))
paths[:, 0] = S0

for t in range(num_steps):
    # This is the exact line from Bachelier.generate_paths
    paths[:, t + 1] = paths[:, t] + drift * dt + sigma * sqrt_dt * Z_transformed[:, t]

final_S = paths[:, -1]
call_payoff = np.maximum(final_S - K, 0)
call_price_mc = np.mean(call_payoff) * np.exp(-r * T)

print("MANUAL GENERATION (exact code replication):")
print(f"  Mean final price: {np.mean(final_S):.6f}")
print(f"  Std final price:  {np.std(final_S):.6f}")
print(f"  Call price (MC):  {call_price_mc:.6f}")

# Now compute analytical
def bachelier_call(S, K, T, r, sigma):
    d = (S - K + r * T) / (sigma * np.sqrt(T))
    return (S - K) * norm.cdf(d) + sigma * np.sqrt(T) * norm.pdf(d) * np.exp(-r * T)

call_theo = bachelier_call(S0, K, T, r, sigma)
print(f"  Call price (Theory): {call_theo:.6f}")
print(f"  Ratio (Theory/MC): {call_theo / call_price_mc:.6f}")

# Now let's check each component:
print("\n" + "="*60)
print("COMPONENT ANALYSIS")
print("="*60)

# 1. Check discount factor
print(f"\nDiscount factor: exp(-r*T) = exp(-{r}*{T}) = {np.exp(-r*T):.6f}")

# 2. Check expected drift
print(f"\nExpected drift over T:")
print(f"  (r - q) * T = {(r - q) * T:.6f}")
print(f"  Observed mean drift: {np.mean(final_S - S0):.6f}")

# 3. Check volatility
print(f"\nExpected volatility (std of returns):")
print(f"  sigma * sqrt(T) = {sigma * np.sqrt(T):.6f}")
print(f"  Observed std:     {np.std(final_S):.6f}")

# 4. Check the analytical d value
d_val = (S0 - K + r * T) / (sigma * np.sqrt(T))
print(f"\nAnalytical d parameter:")
print(f"  d = (S - K + r*T) / (sigma * sqrt(T))")
print(f"  d = ({S0} - {K} + {r}*{T}) / ({sigma} * sqrt({T}))")
print(f"  d = {d_val:.6f}")

# 5. Let's check if maybe the issue is in how the formula is set up
# Bachelier formula requires discounting of the volatility term
print(f"\nAnalytical formula components:")
print(f"  (S - K) * N(d) = {(S0 - K) * norm.cdf(d_val):.6f}")
print(f"  sigma*sqrt(T)*n(d)*exp(-r*T) = {sigma * np.sqrt(T) * norm.pdf(d_val) * np.exp(-r*T):.6f}")
print(f"  Total = {(S0 - K) * norm.cdf(d_val) + sigma * np.sqrt(T) * norm.pdf(d_val) * np.exp(-r*T):.6f}")

# Let me compute the undiscounted expectation
print(f"\nUndiscounted option value (T=0 perspective):")
call_undiscounted = (S0 - K) * norm.cdf(d_val) + sigma * np.sqrt(T) * norm.pdf(d_val)
print(f"  {call_undiscounted:.6f}")
print(f"  Discounted (x exp(-r*T)): {call_undiscounted * np.exp(-r*T):.6f}")

# Check what the MC mean payoff is
print(f"\nMC Payoff Analysis:")
print(f"  Mean undiscounted payoff: {np.mean(call_payoff):.6f}")
print(f"  Mean discounted payoff:   {np.mean(call_payoff) * np.exp(-r*T):.6f}")

# Could the issue be that payoffs are being computed at the wrong time?
# In Bachelier, the payoff should be evaluated at maturity S_T
# And then discounted back

print(f"\nPayoff at maturity check:")
itm_count = np.sum(final_S > K)
print(f"  Paths in-the-money: {itm_count} / {num_paths_av} = {itm_count/num_paths_av*100:.2f}%")
print(f"  Expected from theory: {norm.cdf(d_val)*100:.2f}%")

# Check the distribution
print(f"\nPrice distribution check:")
percentiles = [5, 25, 50, 75, 95]
for p in percentiles:
    val = np.percentile(final_S, p)
    print(f"  {p}th percentile: {val:.2f}")

