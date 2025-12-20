"""
Test if the issue with Merton is in the parameter cloning.

Merton.get_required_params() returns:
{
    "spot": ...,
    "risk_free_rate": ...,
    "dividend_yield": ...,
    "initial_volatility": ...,
    "time_to_maturity": ...,
    "merton_lambda": self.lambda_,
    "merton_mu_j": self.mu_j,
    "merton_sigma_j": self.sigma_j,
}

And Merton.__init__() expects:
(spot, risk_free_rate, dividend_yield, initial_volatility, time_to_maturity,
 lambda_=0.5, mu_j=0.0, sigma_j=0.2)

So "merton_lambda" needs to be mapped to "lambda_"
And "merton_mu_j" needs to be mapped to "mu_j"
And "merton_sigma_j" needs to be mapped to "sigma_j"

The _clone_model with prefix stripping should handle this...
Let me verify.
"""

import numpy as np
from models.merton_jump import MertonJump
from analytics.greeks import GreeksComputer
import inspect

# Create a Merton model
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

print("Original Merton:")
print(f"  lambda_: {merton.lambda_}")
print(f"  mu_j: {merton.mu_j}")
print(f"  sigma_j: {merton.sigma_j}")

# Get the params
params = merton.get_required_params()
print(f"\nget_required_params():")
for k, v in params.items():
    print(f"  {k}: {v}")

# Check the __init__ signature
sig = inspect.signature(MertonJump.__init__)
print(f"\nMertonJump.__init__ parameters:")
for name in sig.parameters:
    if name != 'self':
        print(f"  {name}")

# Now test _clone_model
gc = GreeksComputer()

# Try to clone with spot bump
try:
    cloned = gc._clone_model(merton, spot=110.0)
    print(f"\n✅ Cloning with spot bump succeeded!")
    print(f"  Original spot: {merton.spot}")
    print(f"  Cloned spot: {cloned.spot}")
    print(f"  Original lambda_: {merton.lambda_}")
    print(f"  Cloned lambda_: {cloned.lambda_}")
except Exception as e:
    print(f"\n❌ Cloning failed: {e}")

# Try with _with_vol
try:
    merton_vol_bumped = gc._with_vol(merton, 0.25)
    print(f"\n✅ _with_vol succeeded!")
    print(f"  Original sigma: {merton.sigma}")
    print(f"  Bumped sigma: {merton_vol_bumped.sigma}")
except Exception as e:
    print(f"\n❌ _with_vol failed: {e}")

# Try with _with_rate
try:
    merton_rate_bumped = gc._with_rate(merton, 0.06)
    if merton_rate_bumped is not None:
        print(f"\n✅ _with_rate succeeded!")
        print(f"  Original r: {merton.r}")
        print(f"  Bumped r: {merton_rate_bumped.r}")
    else:
        print(f"\n⚠️  _with_rate returned None (model doesn't expose rate)")
except Exception as e:
    print(f"\n❌ _with_rate failed: {e}")

