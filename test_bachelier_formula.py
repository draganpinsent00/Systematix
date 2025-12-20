"""
Verify Bachelier analytical formula and understand the discrepancy.
"""
import numpy as np
from scipy.stats import norm

def bachelier_call_v1(S, K, T, r, sigma):
    """Version 1: Standard formula"""
    d = (S - K + r * T) / (sigma * np.sqrt(T))
    return (S - K) * norm.cdf(d) + sigma * np.sqrt(T) * norm.pdf(d) * np.exp(-r * T)

def bachelier_call_v2(S, K, T, r, sigma):
    """Version 2: Alternative formulation"""
    sqrt_T = np.sqrt(T)
    d = (S - K + r * T) / (sigma * sqrt_T)
    N_d = norm.cdf(d)
    n_d = norm.pdf(d)  # Standard normal density

    # Call = exp(-r*T) * [ (S - K)*N(d) + sigma*sqrt(T)*n(d) ]
    return np.exp(-r * T) * ((S - K) * N_d + sigma * sqrt_T * n_d)

# Test with ATM option
S = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2

call_v1 = bachelier_call_v1(S, K, T, r, sigma)
call_v2 = bachelier_call_v2(S, K, T, r, sigma)

print(f"Bachelier Call Price (S={S}, K={K}, T={T}, r={r}, sigma={sigma}):")
print(f"  V1 (with discount outside): {call_v1:.6f}")
print(f"  V2 (with discount inside):  {call_v2:.6f}")

# Check the formula components
d = (S - K + r * T) / (sigma * np.sqrt(T))
print(f"\nd = {d:.6f}")
print(f"N(d) = {norm.cdf(d):.6f}")
print(f"n(d) = {norm.pdf(d):.6f}")

# Try with discounting check - maybe v1 doesn't properly discount
print(f"\nIntermediate check:")
print(f"(S - K) * N(d) = {(S - K) * norm.cdf(d):.6f}")
print(f"sigma * sqrt(T) * n(d) = {sigma * np.sqrt(T) * norm.pdf(d):.6f}")
print(f"Sum = {(S - K) * norm.cdf(d) + sigma * np.sqrt(T) * norm.pdf(d):.6f}")
print(f"Sum * exp(-r*T) = {((S - K) * norm.cdf(d) + sigma * np.sqrt(T) * norm.pdf(d)) * np.exp(-r * T):.6f}")

# The issue: v1 probably has the exp(-r*T) in the wrong place
# Let's check which one is correct by pure economic argument:
# At T=0, call should be max(S - K, 0) = 0
# At S=K, T=∞, call should approach infinity in value if r > 0

# With r=0 (no discounting), what should ATM call be?
call_v1_r0 = bachelier_call_v1(S, K, T, 0.0, sigma)
call_v2_r0 = bachelier_call_v2(S, K, T, 0.0, sigma)
print(f"\nWith r=0:")
print(f"  V1: {call_v1_r0:.6f}")
print(f"  V2: {call_v2_r0:.6f}")
print(f"  sigma * sqrt(T) / sqrt(2*pi) = {sigma * np.sqrt(T) * norm.pdf(0):.6f}")

# The true formula should be with exp(-rT) multiplying everything
print(f"\nConclusion: V2 seems more correct (discount factor outside)")

