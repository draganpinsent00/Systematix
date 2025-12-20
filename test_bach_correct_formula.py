"""
Verify the correct Bachelier call formula.

The Bachelier model has:
S_T ~ N(S_0 + (r - q)*T, sigma^2 * T)

The call price is:
C = E[max(S_T - K, 0)] * exp(-r*T)
  = [(F - K)*N(d) + sigma*sqrt(T)*n(d)] * exp(-r*T)

where F = S_0 + (r-q)*T (forward price)
and d = (F - K) / (sigma * sqrt(T))
"""

import numpy as np
from scipy.stats import norm

S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
q = 0.0
sigma = 0.2

# Correct formula
F = S0 + (r - q) * T
d = (F - K) / (sigma * np.sqrt(T))

# The call price formula SHOULD be:
# C = E_Q[max(S_T - K, 0)] * exp(-r*T)
# Under Q, S_T ~ N(F, sigma^2*T) where F is forward
# So: E[max(S_T - K, 0)] = (F - K)*N(d) + sigma*sqrt(T)*n(d)
# And: C = [(F - K)*N(d) + sigma*sqrt(T)*n(d)] * exp(-r*T)

call_correct = ((F - K) * norm.cdf(d) + sigma * np.sqrt(T) * norm.pdf(d)) * np.exp(-r * T)

print("CORRECT FORMULA:")
print(f"F = S0 + (r-q)*T = {F}")
print(f"d = (F - K) / (sigma*sqrt(T)) = {d}")
print(f"C = [(F - K)*N(d) + sigma*sqrt(T)*n(d)] * exp(-r*T)")
print(f"C = {call_correct:.6f}")

# Now what I had before
call_wrong = (S0 - K) * norm.cdf(d) + sigma * np.sqrt(T) * norm.pdf(d) * np.exp(-r * T)

print(f"\nINCORRECT FORMULA (what I used in audit):")
print(f"C = (S0 - K)*N(d) + sigma*sqrt(T)*n(d)*exp(-r*T)")
print(f"C = {call_wrong:.6f}")

print(f"\nRatio (correct/incorrect): {call_correct / call_wrong:.6f}")

# Let me think about this more carefully
# In Bachelier, the forward price matters because of the drift
# The expectation of S_T is S0 + (r-q)*T = F
# So (S0 - K) is NOT the forward moneyness

# The correct formulation is:
# E[max(S_T - K, 0)] = ∫_K^∞ (s - K) * pdf(s) ds
# 
# For S_T ~ N(F, sigma^2*T), this integral evaluates to:
# (F - K) * N(d) + sigma*sqrt(T) * n(d)
# where d = (F - K) / (sigma*sqrt(T))

# And since we discount at rate r:
# C = E_Q[max(S_T - K, 0)] * exp(-r*T)

print(f"\n✅ The correct Bachelier call formula is:")
print(f"C = [(F - K)*N(d) + sigma*sqrt(T)*n(d)] * exp(-r*T)")
print(f"where F = S0 + (r-q)*T")
print(f"and d = (F - K) / (sigma*sqrt(T))")

