from simulator import simulate_gbm_paths, simulate_heston_paths, simulate_merton_paths, simulate_kou_paths
from payoffs import EuropeanCall
from pricing import price_mc, price_heston, price_merton, price_kou
from greeks import compute_greeks_mc

print("Testing models...")

# Test GBM
print("\n1. GBM:")
call = EuropeanCall(100.0)
res_gbm = price_mc(call, 100.0, 0.01, 0.2, 1.0, 12, 2000, seed=42)
print(f"   Price: {res_gbm['price']:.6f}, SE: {res_gbm['stderr']:.6f}")

# Test Heston
print("\n2. Heston:")
res_heston = price_heston(call, 100.0, 0.01, 0.04, 1.5, 0.04, 0.3, -0.7, 1.0, 12, 2000, seed=42)
print(f"   Price: {res_heston['price']:.6f}, SE: {res_heston['stderr']:.6f}")

# Test Merton
print("\n3. Merton:")
res_merton = price_merton(call, 100.0, 0.01, 0.2, 1.0, 12, 2000, lambda_jump=0.1, seed=42)
print(f"   Price: {res_merton['price']:.6f}, SE: {res_merton['stderr']:.6f}")

# Test Kou
print("\n4. Kou:")
res_kou = price_kou(call, 100.0, 0.01, 0.2, 1.0, 12, 2000, lambda_jump=0.1, seed=42)
print(f"   Price: {res_kou['price']:.6f}, SE: {res_kou['stderr']:.6f}")

# Test Greeks
print("\n5. Greeks (GBM):")
greeks = compute_greeks_mc(call, 'gbm', 100.0, 0.01, 0.2, 1.0, 12, 2000, seed=42)
print(f"   Delta: {greeks['delta']:.6f}")
print(f"   Gamma: {greeks['gamma']:.6g}")
print(f"   Vega: {greeks['vega']:.6f}")

print("\n✅ All models working!")

