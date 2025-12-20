"""
Detailed investigation of Bachelier volatility scaling.
"""
import numpy as np
from scipy.stats import norm

# The Bachelier formula with sigma = 0.2 gives call = 0.0736
# The MC with sigma = 0.2 gives call = 0.1019

# This suggests the MC is using a HIGHER volatility
# If MC used sigma * 100 = 20.0 instead of sigma = 0.2:

def bachelier_call(S, K, T, r, sigma):
    d = (S - K + r * T) / (sigma * np.sqrt(T))
    return (S - K) * norm.cdf(d) + sigma * np.sqrt(T) * norm.pdf(d) * np.exp(-r * T)

S = 100.0
K = 100.0
T = 1.0
r = 0.05

# Theoretical with sigma = 0.2
theo_02 = bachelier_call(S, K, T, r, 0.2)
print(f"Theoretical with sigma=0.2: {theo_02:.6f}")

# Theoretical with sigma = 20.0 (if misinterpreted as percentage)
theo_20 = bachelier_call(S, K, T, r, 20.0)
print(f"Theoretical with sigma=20.0: {theo_20:.6f}")

# Check the ratio
print(f"Ratio: {theo_20 / theo_02:.6f}")

# What if the issue is that the increments are scaled wrong?
# dS = (r - q) * dt + sigma * sqrt(dt) * Z
# If they accidentally did:
# dS = (r - q) * dt + (sigma * 100) * sqrt(dt) * Z
# That would be the issue!

# Actually, let's think about it differently. The MC gave 0.1019
# Divide by theoretical: 0.1019 / 0.0736 = 1.386

# This is suspiciously close to... hmm, not an obvious ratio.

# Let me compute what sigma would give MC = 0.1019
from scipy.optimize import fsolve

def diff_func(sigma_test):
    return bachelier_call(S, K, T, r, sigma_test) - 0.101885

sigma_implied = fsolve(diff_func, 0.5)[0]
print(f"\nImplied sigma from MC price: {sigma_implied:.6f}")
print(f"Input sigma: 0.2")
print(f"Ratio: {sigma_implied / 0.2:.6f}")

# Let's also check what the standard deviation of final prices should be
print(f"\nExpected standard deviation of S_T:")
print(f"  In Bachelier: sqrt(T) * sigma = {np.sqrt(T) * 0.2:.6f}")
print(f"  From MC: {0.200038:.6f}")

# So the paths are correct (std matches). But the call price is higher.
# This suggests the issue might be in the... PAYOFF function?

# Wait, maybe the issue is simple:
# What if the analytical formula uses percentage volatility (e.g., 0.20 = 20%)
# But the MC is being given percentage volatility too?

# Let me compute the call price if we misunderstand the input:
# If sigma input is meant to be 20% (so 0.20 as a decimal)
# But the formula assumes basis points (so 0.0020)

theo_pct_confusion = bachelier_call(S, K, T, r, 0.002)
print(f"\nTheoretical with sigma=0.002: {theo_pct_confusion:.6f}")

# Nope, that's too small.

# What if the problem is in how dividend_yield interacts with initial_volatility?
# Let me re-read the Bachelier generation...

# Actually I wonder if the issue is the dividend yield parameter being used somewhere!
# If they accidentally treat dividend_yield as another volatility source...

# Or - what if initial_volatility is being multiplied by 100 somewhere?
# That would explain the ratio of ~1.4x

print(f"\nTesting hypothesis: is sigma being multiplied by some factor?")
print(f"Theoretical with sigma=0.2: {theo_02:.6f}")

# What if there's a sqrt(2) factor somewhere?
print(f"Theoretical with sigma=0.2/sqrt(2)={0.2/np.sqrt(2):.4f}: {bachelier_call(S, K, T, r, 0.2/np.sqrt(2)):.6f}")

# What about the reverse?
print(f"Theoretical with sigma=0.2*sqrt(2)={0.2*np.sqrt(2):.4f}: {bachelier_call(S, K, T, r, 0.2*np.sqrt(2)):.6f}")

# Or some discount factor issue?
print(f"\nDiscount factor issue?")
print(f"Theo * exp(r*T) = {theo_02 * np.exp(r * T):.6f}")
print(f"MC value = 0.101885")

# Hmm, that's close! 0.077... * 1.0513 = 0.081... nope.

# Let me just directly check: what adjustment makes them match?
adjustment = 0.101885 / theo_02
print(f"\nDirect adjustment factor needed: {adjustment:.6f}")

