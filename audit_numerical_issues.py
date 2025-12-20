"""
Audit script for numerical issues in pricing and Greeks.
Tests:
1. Bachelier model pricing scale
2. SABR Greeks computation
3. Vega & Rho magnitudes for Heston and Merton
"""

import numpy as np
from scipy.stats import norm
from models.bachelier import Bachelier
from models.heston import Heston
from models.sabr import SABR
from models.merton_jump import MertonJump
from analytics.greeks import GreeksComputer
from core.rng_engines import PCG64RNG
import warnings

warnings.filterwarnings('ignore')

def bachelier_analytical_call(S, K, T, r, sigma):
    """Analytical Bachelier call price.

    Formula: C = [(F - K)*N(d) + sigma*sqrt(T)*n(d)] * exp(-r*T)
    where F = S + r*T (forward price), d = (F - K) / (sigma*sqrt(T))
    """
    F = S + r * T  # Forward price
    d = (F - K) / (sigma * np.sqrt(T))
    return ((F - K) * norm.cdf(d) + sigma * np.sqrt(T) * norm.pdf(d)) * np.exp(-r * T)

def bachelier_analytical_put(S, K, T, r, sigma):
    """Analytical Bachelier put price."""
    F = S + r * T
    d = (F - K) / (sigma * np.sqrt(T))
    return ((K - F) * norm.cdf(-d) + sigma * np.sqrt(T) * norm.pdf(d)) * np.exp(-r * T)

# ========================
# Issue 1: Bachelier Scale
# ========================
print("\n" + "="*70)
print("ISSUE 1: BACHELIER MODEL PRICING SCALE")
print("="*70)

S = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2  # 20% absolute vol (normal model)

# Analytical
theo_call = bachelier_analytical_call(S, K, T, r, sigma)
theo_put = bachelier_analytical_put(S, K, T, r, sigma)

print(f"\nSpot={S}, Strike={K}, T={T}, r={r}, sigma(ABM)={sigma}")
print(f"Theoretical Bachelier Call: {theo_call:.6f}")
print(f"Theoretical Bachelier Put:  {theo_put:.6f}")

# Monte Carlo
bach_model = Bachelier(
    spot=S,
    risk_free_rate=r,
    dividend_yield=0.0,
    initial_volatility=sigma,
    time_to_maturity=T
)

rng = PCG64RNG(seed=42)
num_paths = 10000
num_steps = 252

paths = bach_model.generate_paths(rng, num_paths, num_steps)
call_payoffs = np.maximum(paths[:, -1] - K, 0)
put_payoffs = np.maximum(K - paths[:, -1], 0)

call_price_mc = np.mean(call_payoffs) * np.exp(-r * T)
put_price_mc = np.mean(put_payoffs) * np.exp(-r * T)

print(f"\nMonte Carlo Call Price:     {call_price_mc:.6f}")
print(f"Monte Carlo Put Price:      {put_price_mc:.6f}")

call_ratio = theo_call / call_price_mc if call_price_mc > 0 else np.inf
put_ratio = theo_put / put_price_mc if put_price_mc > 0 else np.inf

print(f"\nRatio (Theory/MC):")
print(f"  Call: {call_ratio:.4f}")
print(f"  Put:  {put_ratio:.4f}")

if abs(call_ratio - 1.0) > 0.1 or abs(put_ratio - 1.0) > 0.1:
    print("\n⚠️  ISSUE CONFIRMED: Bachelier MC prices are NOT matching theory!")
    print(f"   Expected ratio ~1.0, got Call={call_ratio:.4f}, Put={put_ratio:.4f}")
else:
    print("\n✅ Bachelier pricing is correct (ratio ≈ 1.0)")

# ========================
# Issue 2: SABR Greeks
# ========================
print("\n" + "="*70)
print("ISSUE 2: SABR GREEKS COMPUTATION")
print("="*70)

sabr_model = SABR(
    spot=100.0,
    risk_free_rate=0.05,
    dividend_yield=0.0,
    initial_volatility=0.2,
    time_to_maturity=1.0,
    alpha=0.4,
    beta=0.5,
    nu=0.5,
    rho=-0.5
)

# Generate paths
rng_sabr = PCG64RNG(seed=42)
paths_sabr = sabr_model.generate_paths(rng_sabr, num_paths, num_steps)
call_payoffs_sabr = np.maximum(paths_sabr[:, -1] - 100.0, 0)
price_sabr = np.mean(call_payoffs_sabr) * np.exp(-r * T)

# Test Greeks
gc = GreeksComputer(bump_size=0.01, rng_seed=42)

def sabr_call_payoff(paths):
    return np.maximum(paths[:, -1] - 100.0, 0)

print(f"\nTesting SABR Greeks computation...")
print(f"Base price: {price_sabr:.6f}")

try:
    greeks = gc.compute_all(
        spot=100.0,
        price=price_sabr,
        paths=paths_sabr,
        payoff_func=sabr_call_payoff,
        risk_free_rate=0.05,
        time_to_maturity=1.0,
        volatility=0.2,
        model=sabr_model,
        rng_engine=rng_sabr,
        num_paths=num_paths,
        num_steps=num_steps
    )

    print("\n✅ SABR Greeks computed successfully!")
    print(f"  Delta:  {greeks['delta']:.6f}")
    print(f"  Gamma:  {greeks['gamma']:.6f}")
    print(f"  Vega:   {greeks['vega']:.6f}")
    print(f"  Theta:  {greeks['theta']:.6f}")
    print(f"  Rho:    {greeks['rho']:.6f}")

    # Check for NaN or inf
    for name, val in greeks.items():
        if not np.isfinite(val):
            print(f"\n⚠️  {name.upper()} is {val} (not finite)!")

except Exception as e:
    print(f"\n❌ SABR Greeks computation FAILED!")
    print(f"Error: {type(e).__name__}: {e}")

# ========================
# Issue 3: Vega & Rho Explosions
# ========================
print("\n" + "="*70)
print("ISSUE 3: VEGA & RHO MAGNITUDES (HESTON & MERTON)")
print("="*70)

heston_model = Heston(
    spot=100.0,
    risk_free_rate=0.05,
    dividend_yield=0.0,
    initial_volatility=0.2,
    time_to_maturity=1.0,
    kappa=2.0,
    theta=0.04,
    sigma=0.3,
    rho=-0.5
)

merton_model = MertonJump(
    spot=100.0,
    risk_free_rate=0.05,
    dividend_yield=0.0,
    initial_volatility=0.2,
    time_to_maturity=1.0,
    lambda_=0.5,
    mu_j=0.0,
    sigma_j=0.2
)

for model_name, model in [("Heston", heston_model), ("Merton", merton_model)]:
    print(f"\n{model_name}:")
    print("-" * 50)

    rng_test = PCG64RNG(seed=42)
    paths_test = model.generate_paths(rng_test, num_paths, num_steps)
    call_payoffs_test = np.maximum(paths_test[:, -1] - 100.0, 0)
    price_test = np.mean(call_payoffs_test) * np.exp(-r * T)

    def call_payoff(paths):
        return np.maximum(paths[:, -1] - 100.0, 0)

    try:
        greeks_test = gc.compute_all(
            spot=100.0,
            price=price_test,
            paths=paths_test,
            payoff_func=call_payoff,
            risk_free_rate=0.05,
            time_to_maturity=1.0,
            volatility=0.2,
            model=model,
            rng_engine=rng_test,
            num_paths=num_paths,
            num_steps=num_steps
        )

        print(f"Base price: {price_test:.6f}")
        print(f"  Delta: {greeks_test['delta']:>12.6f}  | Reasonable? {0.3 < greeks_test['delta'] < 0.7}")
        print(f"  Gamma: {greeks_test['gamma']:>12.6f}  | Reasonable? {0 < greeks_test['gamma'] < 0.05}")
        print(f"  Vega:  {greeks_test['vega']:>12.6f}  | Reasonable? {1 < greeks_test['vega'] < 100}")
        print(f"  Theta: {greeks_test['theta']:>12.6f}  | Reasonable? {-100 < greeks_test['theta'] < 100}")
        print(f"  Rho:   {greeks_test['rho']:>12.6f}  | Reasonable? {-100 < greeks_test['rho'] < 100}")

        # Check for explosions
        if abs(greeks_test['vega']) > 1000 or not np.isfinite(greeks_test['vega']):
            print(f"\n  ⚠️  VEGA EXPLOSION: {greeks_test['vega']:.6f}")
        if abs(greeks_test['rho']) > 1000 or not np.isfinite(greeks_test['rho']):
            print(f"\n  ⚠️  RHO EXPLOSION: {greeks_test['rho']:.6f}")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*70)
print("AUDIT COMPLETE")
print("="*70 + "\n")

