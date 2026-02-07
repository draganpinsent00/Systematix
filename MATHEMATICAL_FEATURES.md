# Systematix — Monte Carlo Pricing Engine

A modular, extensible Monte Carlo simulation platform for pricing derivatives with multiple stochastic models, variance reduction techniques, and risk analytics.

---

## Features Overview

### 1. Stochastic Differential Equations (SDEs)

#### Geometric Brownian Motion (GBM)
**SDE:**
```
dS_t = μ S_t dt + σ S_t dW_t
```
- **μ**: drift (risk-free rate)
- **σ**: volatility
- **W_t**: Wiener process

**Discretization (Euler-Maruyama):**
```
S_{t+Δt} = S_t exp((μ - σ²/2)Δt + σ√Δt Z)
```
where `Z ~ N(0,1)`

---

#### Heston Model
**SDE System:**
```
dS_t = r S_t dt + √v_t S_t dW_t^S
dv_t = κ(θ - v_t) dt + ξ √v_t dW_t^v
```
- **v_t**: stochastic variance
- **κ**: mean reversion speed
- **θ**: long-term variance
- **ξ**: volatility of volatility
- **ρ**: correlation between `dW^S` and `dW^v`

**Discretization (Full Truncation):**
```
S_{t+Δt} = S_t exp((r - v_t/2)Δt + √(v_t Δt) Z_S)
v_{t+Δt} = max(v_t + κ(θ - v_t)Δt + ξ√(v_t Δt)(ρ Z_S + √(1-ρ²) Z_v), 0)
```

---

#### 3/2 Heston Model
**SDE System:**
```
dS_t = r S_t dt + √v_t S_t dW_t^S
dv_t = κ(θ - v_t) dt + ξ v_t^(3/2) dW_t^v
```
Higher volatility of volatility sensitivity compared to Heston.

---

#### Merton Jump Diffusion
**SDE:**
```
dS_t = (μ - λk) S_t dt + σ S_t dW_t + (J - 1) S_t dN_t
```
- **λ**: jump intensity (arrivals per year)
- **k**: `E[J - 1]`, expected relative jump size
- **J**: jump size, `log(J) ~ N(μ_J, σ_J²)`
- **N_t**: Poisson process with intensity λ

**Discretization:**
```
S_{t+Δt} = S_t exp((μ - σ²/2 - λk)Δt + σ√Δt Z) ∏ J_i
```
where the number of jumps ~ Poisson(λΔt)

---

#### Kou Jump Diffusion
**SDE:**
```
dS_t = (μ - λk) S_t dt + σ S_t dW_t + (J - 1) S_t dN_t
```
**Jump Distribution (Double Exponential):**
```
J - 1 = { Y with prob p    (Y ~ Exp(η₁), upward jumps)
        {-Z with prob 1-p  (Z ~ Exp(η₂), downward jumps)
```

---

#### SABR Model
**SDE System:**
```
dF_t = α_t F_t^β dW_t^F
dα_t = ν α_t dW_t^α
```
- **F_t**: forward price
- **α_t**: stochastic volatility
- **β**: CEV exponent (0 = normal, 1 = lognormal)
- **ν**: vol-of-vol
- **ρ**: correlation

Used primarily for implied volatility surfaces in interest rate/FX markets.

---

#### Bachelier Model
**SDE:**
```
dS_t = μ dt + σ dW_t
```
Normal (additive noise) model. Used for negative rate environments or short-dated options.

**Discretization:**
```
S_{t+Δt} = S_t + μΔt + σ√Δt Z
```

---

#### Shifted Lognormal Model
**SDE:**
```
d(S_t + α) = μ(S_t + α) dt + σ(S_t + α) dW_t
```
where **α** is a shift parameter allowing negative rates while preserving lognormal dynamics.

---

#### Local Volatility Model
**SDE:**
```
dS_t = r S_t dt + σ(S_t, t) S_t dW_t
```
where **σ(S, t)** is a deterministic volatility surface calibrated to market data.

---

#### Regime-Switching Model
**SDE with Markov Chain:**
```
dS_t = μ(Z_t) S_t dt + σ(Z_t) S_t dW_t
```
where **Z_t** is a Markov chain representing economic regimes (e.g., bull/bear markets).

---

#### Multi-Asset GBM
**SDE System (n assets):**
```
dS_i(t) = μ_i S_i(t) dt + σ_i S_i(t) dW_i(t)
```
with **correlation matrix** Σ defining `Cov(dW_i, dW_j) = ρ_{ij} dt`

**Discretization:**
```
S_i(t+Δt) = S_i(t) exp((μ_i - σ_i²/2)Δt + σ_i√Δt Z_i)
```
where **Z ~ N(0, Σ)** via Cholesky decomposition.

---

### 2. Option Types & Payoffs

#### Vanilla Options

**European Call:**
```
V_T = max(S_T - K, 0)
```

**European Put:**
```
V_T = max(K - S_T, 0)
```

**American Call:**
```
V_t = max(S_t - K, E[e^(-rΔt) V_{t+Δt}])
```
Priced using Longstaff-Schwartz LSM algorithm.

**American Put:**
```
V_t = max(K - S_t, E[e^(-rΔt) V_{t+Δt}])
```

---

#### Path-Dependent Exotics

**Asian Options (Arithmetic Average):**
```
S_avg = (1/n) ∑_{i=1}^n S_{t_i}
V_T^call = max(S_avg - K, 0)
V_T^put = max(K - S_avg, 0)
```

**Asian Options (Geometric Average):**
```
S_geom = (∏_{i=1}^n S_{t_i})^(1/n)
V_T = max(S_geom - K, 0)
```

**Barrier Options:**

*Up-and-Out Call:*
```
V_T = { max(S_T - K, 0)  if max(S_t) < H for all t
      { 0                otherwise
```

*Down-and-Out Put:*
```
V_T = { max(K - S_T, 0)  if min(S_t) > H for all t
      { 0                otherwise
```

*Up-and-In Call:*
```
V_T = { max(S_T - K, 0)  if max(S_t) ≥ H at some t
      { 0                otherwise
```

*Down-and-In Put:*
```
V_T = { max(K - S_T, 0)  if min(S_t) ≤ H at some t
      { 0                otherwise
```

**Lookback Options:**

*Floating Strike Call:*
```
V_T = S_T - min_{t∈[0,T]} S_t
```

*Floating Strike Put:*
```
V_T = max_{t∈[0,T]} S_t - S_T
```

*Fixed Strike Call:*
```
V_T = max(max_{t∈[0,T]} S_t - K, 0)
```

*Fixed Strike Put:*
```
V_T = max(K - min_{t∈[0,T]} S_t, 0)
```

**Digital/Binary Options:**
```
V_T^call = { Q  if S_T > K
           { 0  otherwise

V_T^put = { Q  if S_T < K
          { 0  otherwise
```
where **Q** is the cash payout.

**Cliquet Options:**
```
V_T = ∑_{i=1}^n max(min(S_{t_i}/S_{t_{i-1}} - 1, cap), floor)
```
Sum of capped and floored periodic returns.

---

#### Multi-Asset Options

**Basket Options:**
```
S_basket = ∑_{i=1}^n w_i S_i(T)
V_T^call = max(S_basket - K, 0)
V_T^put = max(K - S_basket, 0)
```

**Rainbow Options (Best-of Call):**
```
V_T = max(max_{i=1,...,n} S_i(T) - K, 0)
```

**Rainbow Options (Worst-of Put):**
```
V_T = max(K - min_{i=1,...,n} S_i(T), 0)
```

**Spread Options:**
```
V_T = max(S_1(T) - S_2(T) - K, 0)
```

**Exchange Options:**
```
V_T = max(S_1(T) - S_2(T), 0)
```

**Quanto Options:**
Options on foreign assets settled in domestic currency at a fixed exchange rate.

---

#### Exotic Structures

**Bermudan Options:**
Exercise allowed only at discrete dates `t_1, t_2, ..., t_n`:
```
V_{t_i} = max(h(S_{t_i}), e^(-rΔt) E[V_{t_{i+1}}])
```

**Persian Options (Work in Progress):**
Path-dependent options with payoffs dependent on multiple barrier levels and path characteristics.

---

#### Fixed Income & FX

**Caps/Floors:**
```
Caplet_i = N × τ_i × max(L_i - K, 0)
Floorlet_i = N × τ_i × max(K - L_i, 0)
```
where **L_i** is the LIBOR rate, **N** is notional, **τ_i** is accrual period.

**Swaptions:**
Option to enter into an interest rate swap:
```
V = max(∑_{i=1}^n τ_i D(t_i) (S - K), 0)
```
where **S** is the swap rate, **K** is the strike, **D(t_i)** is the discount factor.

**FX Options:**
```
V_T^call = max(X_T - K, 0)
V_T^put = max(K - X_T, 0)
```
where **X_T** is the exchange rate at maturity.

---

### 3. Random Number Generation (RNG)

#### Pseudo-Random Generators

**Mersenne Twister (MT19937):**
```
Period: 2^19937 - 1
```
- Widely used, high-quality uniform randomness
- Equidistribution in 623 dimensions

**PCG64 (Permuted Congruential Generator):**
```
state = state × multiplier + increment
output = permute(state)
```
- Excellent statistical properties
- Fast generation
- Small state size

**XorShift128+:**
```
s1 = s0
s0 = s1
s1 ^= s1 << 23
s1 ^= s1 >> 17
s1 ^= s0
s1 ^= s0 >> 26
return s0 + s1
```
- Extremely fast
- Good quality for Monte Carlo
- Fails some statistical tests

**Philox4x32:**
```
Counter-based PRNG
Key schedule + bijective function
```
- Excellent for parallel/GPU computing
- Reproducible across platforms
- Cryptographically secure variant available

**Middle-Square Weyl Sequence:**
```
x = x²
x = middle_bits(x) + weyl_sequence
```
- Fast, simple
- Reproducible
- Suitable for simulations

---

#### Quasi-Random Sequences

**Sobol Sequence:**
Low-discrepancy sequence for improved Monte Carlo convergence:
```
Standard error ~ O((log n)^d / n) vs O(1/√n) for pseudo-random
```
- Better coverage of multi-dimensional space
- Faster convergence for smooth integrands
- Deterministic

**Usage:**
Best for low-dimensional problems (d < 10-20) with smooth payoffs.

---

#### Distributions

**Standard Normal (Box-Muller Transform):**
```
Z₁ = √(-2 ln U₁) cos(2π U₂)
Z₂ = √(-2 ln U₁) sin(2π U₂)
```
where `U₁, U₂ ~ Uniform(0,1)`

**Student-t Distribution:**
```
Z ~ t(ν) via inverse CDF of t-distribution
```
- **ν**: degrees of freedom
- Heavier tails than normal
- Used for modeling extreme events

**Uniform Distribution:**
```
U ~ Uniform(0, 1)
```
Base distribution for all transformations.

---

### 4. Greeks Computation

Greeks are computed using **finite difference methods** with **common random numbers (CRN)** for variance reduction.

#### Delta (∂V/∂S)
**Central Difference:**
```
Δ = (V(S + εS) - V(S - εS)) / (2εS)
```
- **ε**: bump size (typically 0.01 or 1%)
- **CRN**: Use same random paths for both bumps

**Interpretation:** Change in option value per unit change in spot price.

---

#### Gamma (∂²V/∂S²)
**Central Difference:**
```
Γ = (V(S + εS) - 2V(S) + V(S - εS)) / (εS)²
```

**Interpretation:** Rate of change of delta; convexity of option value.

**Properties:**
- Always positive for long options
- Maximum near at-the-money
- Approaches zero deep in/out of the money

---

#### Vega (∂V/∂σ)
**Central Difference:**
```
ν = (V(σ + εσ) - V(σ - εσ)) / (2εσ)
```
- **ε**: typically 0.0001 or 0.01 (1 vol point)

**Interpretation:** Change in option value per 1% change in volatility.

**Properties:**
- Always positive for long options
- Maximum near at-the-money
- Larger for longer maturities

---

#### Theta (∂V/∂t)
**Forward Difference:**
```
Θ = (V(T - εt) - V(T)) / εt
```
- **εt**: time bump (typically 1/365 for daily theta)

**Interpretation:** Change in option value per day (time decay).

**Properties:**
- Usually negative for long options (time decay)
- Can be positive for deep ITM puts

---

#### Rho (∂V/∂r)
**Central Difference:**
```
ρ = (V(r + εr) - V(r - εr)) / (2εr)
```
- **εr**: typically 0.0001 (1 basis point)

**Interpretation:** Change in option value per 1% change in interest rate.

**Properties:**
- Positive for calls, negative for puts
- Larger for longer maturities
- Generally small for short-dated options

---

#### Common Random Numbers (CRN)

For all Greeks:
```
Same seed → Same paths → Reduced variance in differences
```

**Implementation:**
1. Generate base paths with seed
2. For each bump, regenerate paths with same seed
3. Compute difference

**Benefit:** Variance of Δ̂ scales as `O(ε²)` instead of `O(1/n)`

---

### 5. Variance Reduction Techniques

#### Antithetic Variates
For each path with random variable `Z`, generate antithetic path with `-Z`:
```
V_avg = (V(Z) + V(-Z)) / 2
```

**Variance Reduction:**
```
Var(V_avg) = (Var(V) / 2) × (1 + Corr(V(Z), V(-Z)))
```

For monotonic payoffs: `Corr < 0` → variance reduction

**Theoretical Efficiency:**
Up to 50% variance reduction for symmetric payoffs.

---

#### Control Variates
Use known benchmark (e.g., Black-Scholes price `V_BS`):
```
V_CV = V_MC + β(V_BS - V_BS_MC)

β = Cov(V_MC, V_BS) / Var(V_BS)
```

**Variance:**
```
Var(V_CV) = Var(V_MC) × (1 - ρ²)
```
where **ρ** is the correlation between `V_MC` and `V_BS`.

**Best Case:** ρ = 1 → perfect control variate → zero variance

---

#### Moment Matching
Adjust sampled paths to match theoretical moments:
```
Z_adj = (Z - mean(Z)) / std(Z)
```

Forces sample mean = 0 and sample variance = 1 exactly.

**Application:**
- Ensures drift consistency
- Reduces bias in small samples

---

#### Importance Sampling
Generate samples from distribution `g(x)` instead of `f(x)`:
```
E_f[h(X)] = E_g[h(X) × f(X)/g(X)]
```

**Optimal:** `g*(x) ∝ |h(x)| f(x)`

**Use Case:** Deep out-of-the-money options

---

#### Stratified Sampling
Divide sample space into strata and sample proportionally:
```
V = ∑_{i=1}^k w_i V_i
```
where **w_i** is the weight of stratum i.

**Variance Reduction:**
```
Var(V_strat) ≤ Var(V_MC)
```

---

### 6. Risk Analytics

#### Value at Risk (VaR)
**Definition:**
```
VaR_α = -Percentile(ΔP, α)
```
where **α** is the confidence level (e.g., 95%, 99%).

**Interpretation:** Maximum expected loss over time period at confidence level α.

**Example:**
```
VaR_95 = $1M means: 95% confident loss ≤ $1M
```

---

#### Conditional Value at Risk (CVaR / Expected Shortfall)
**Definition:**
```
CVaR_α = E[Loss | Loss > VaR_α]
```

**Interpretation:** Expected loss given that loss exceeds VaR.

**Properties:**
- Always ≥ VaR
- Coherent risk measure
- Captures tail risk

---

#### Maximum Drawdown
**Definition:**
```
MDD = max_{t∈[0,T]} (max_{s∈[0,t]} P_s - P_t)
```

**Interpretation:** Largest peak-to-trough decline.

---

#### Sharpe Ratio
**Definition:**
```
SR = (E[R] - r_f) / σ_R
```
where **R** is return, **r_f** is risk-free rate, **σ_R** is return volatility.

---

#### Convergence Diagnostics

**Standard Error:**
```
SE = σ_MC / √n
```
where **σ_MC** is sample standard deviation, **n** is number of paths.

**95% Confidence Interval:**
```
CI = [μ̂ - 1.96 × SE, μ̂ + 1.96 × SE]
```

**Convergence Rate:**
```
Error ~ O(1/√n)         [standard Monte Carlo]
Error ~ O((log n)^d/n)  [quasi-Monte Carlo]
```

**Stopping Criterion:**
```
Stop when: SE / |μ̂| < tolerance
```

---

### 7. Numerical Methods

#### Black-Scholes Analytical Formula

**Call Price:**
```
C = S₀ Φ(d₁) - K e^(-rT) Φ(d₂)

d₁ = (ln(S₀/K) + (r + σ²/2)T) / (σ√T)
d₂ = d₁ - σ√T
```

where **Φ** is the standard normal CDF.

**Put Price:**
```
P = K e^(-rT) Φ(-d₂) - S₀ Φ(-d₁)
```

**Put-Call Parity:**
```
C - P = S₀ - K e^(-rT)
```

---

#### Bachelier Model (Normal)

**Call Price:**
```
C = (S - K) Φ(d) + σ√T φ(d)

d = (S - K) / (σ√T)
```

where **φ** is the standard normal PDF.

---

#### Longstaff-Schwartz (LSM) for American Options

**Algorithm:**
1. Generate Monte Carlo paths forward
2. Work backward from maturity
3. At each exercise time `t_i`:
   - Identify in-the-money paths
   - Regress discounted continuation value on basis functions
   - Compare exercise value vs continuation value
   - Exercise if immediate payoff > continuation

**Continuation Value Regression:**
```
C(t, S) ≈ β₀ + β₁S + β₂S² + β₃S³ + ...
```

**Basis Functions:**
- Simple: `{1, S, S²}`
- Laguerre polynomials: `L_0(S), L_1(S), L_2(S), ...`
- Hermite polynomials

**Decision Rule:**
```
Exercise at t if: h(S_t) > Ĉ(t, S_t)
```
where **h** is intrinsic value, **Ĉ** is estimated continuation.

---

#### Heston Semi-Closed Form (Fourier Transform)

**Call Price via Characteristic Function:**
```
C = S₀ P₁ - K e^(-rT) P₂

P_j = 1/2 + (1/π) ∫₀^∞ Re[e^(-iφ ln(K)) f_j(φ) / (iφ)] dφ
```

where **f_j** is the characteristic function of log(S_T).

**Used for:** Calibration, benchmarking Monte Carlo

---

#### Implied Volatility via Newton-Raphson

**Goal:** Solve `V_market = V_BS(σ)` for σ

**Iteration:**
```
σ_{n+1} = σ_n - (V_BS(σ_n) - V_market) / Vega(σ_n)
```

**Initial Guess:**
```
σ_0 = √(2π / T) × V_market / S₀
```

---

### 8. Model Calibration

#### Heston Calibration

**Objective:** Minimize squared error between model and market prices:
```
min_{θ} ∑_i w_i (V_market^i - V_model^i(θ))²
```

**Parameters to Calibrate:** `θ = {v₀, κ, θ, ξ, ρ}`

**Constraints:**
- Feller condition: `2κθ > ξ²` (ensures positive variance)
- `ρ ∈ [-1, 1]`
- All parameters > 0

**Methods:**
- Levenberg-Marquardt
- Differential Evolution
- Particle Swarm Optimization

---

#### SABR Calibration

**Objective:** Fit SABR implied volatility smile:
```
σ_imp(K, T; α, β, ρ, ν)
```

**Hagan's Approximation:**
```
σ_BS ≈ α × [various terms involving β, ρ, ν, F, K]
```

**Common:** Fix β = 0.5 or 1.0, calibrate `{α, ρ, ν}`

---

#### Local Volatility Surface

**Dupire's Formula:**
```
σ_loc²(K, T) = (∂C/∂T + rK ∂C/∂K) / (K² ∂²C/∂K²/2)
```

**Construction:**
1. Interpolate market option prices
2. Compute derivatives numerically
3. Apply Dupire formula
4. Smooth/regularize surface

---

### 9. Implementation Details

#### Path Generation

**Euler-Maruyama (GBM):**
```python
for i in range(n_steps):
    dW = sqrt(dt) * Z[i]
    S[i+1] = S[i] * exp((r - 0.5*sigma**2)*dt + sigma*dW)
```

**Milstein (higher order):**
```python
S[i+1] = S[i] * (1 + r*dt + sigma*dW + 0.5*sigma**2*(dW**2 - dt))
```

---

#### Cholesky Decomposition (Multi-Asset)

**Correlation to Covariance:**
```
Σ = diag(σ) × Ρ × diag(σ)
```

**Cholesky:**
```
Σ = L L^T
```

**Correlated Brownian Motions:**
```
W = L × Z
```
where **Z** is vector of independent N(0,1) variables.

---

#### Discounting

**Present Value:**
```
PV = E[e^(-rT) V_T]
```

**Continuous Compounding:**
```
DF(T) = e^(-rT)
```

**Discrete Compounding:**
```
DF(T) = 1 / (1 + r)^T
```

---

#### Early Exercise Boundary (American Options)

**Optimal Stopping:**
```
V(t, S) = max(h(S), C(t, S))
```
where:
- **h(S)**: intrinsic value (exercise now)
- **C(t, S)**: continuation value (hold)

**Critical Price:**
```
S* = {S : h(S) = C(t, S)}
```

Exercise if `S > S*` (call) or `S < S*` (put)

---

### 10. Validation & Testing

#### Unit Tests

**Model Correctness:**
- GBM → Black-Scholes limit
- Heston → GBM when ξ = 0
- Jump models → GBM when λ = 0

**Greek Signs:**
- Delta: 0 < Δ_call < 1, -1 < Δ_put < 0
- Gamma: Γ > 0
- Vega: ν > 0
- Theta: Θ_call < 0 (usually)
- Rho: ρ_call > 0, ρ_put < 0

**Put-Call Parity:**
```
C - P = S - K e^(-rT)
```

---

#### Convergence Tests

**Monte Carlo Error:**
```
Error ~ O(1/√n)
```

**Verification:**
- Plot price vs √(1/n)
- Should be linear

**Quasi-Monte Carlo:**
- Error ~ O((log n)^d / n)
- Faster convergence

---

#### Benchmark Comparisons

**European Options:**
- Monte Carlo vs Black-Scholes
- Tolerance: < 0.1% for 100k paths

**American Options:**
- LSM vs Binomial Tree
- LSM vs FDM (Finite Difference)

**Exotics:**
- Monte Carlo vs Semi-Analytical (when available)
- Cross-validation with external libraries

---

### 11. Performance Optimization

#### Vectorization

**NumPy Vectorized Operations:**
```python
# Bad: loop
for i in range(n):
    S[i+1] = S[i] * exp(drift*dt + vol*dW[i])

# Good: vectorized
S = S0 * np.cumprod(np.exp(drift*dt + vol*dW))
```

**Speedup:** 10-100x

---

#### Memory Management

**Chunking Large Simulations:**
```python
for chunk in range(n_chunks):
    paths = generate_paths(n_paths_per_chunk)
    payoffs = compute_payoff(paths)
    results.append(payoffs.mean())
return np.mean(results)
```

**Benefit:** Handle millions of paths without memory overflow

---

#### Parallel Processing

**Path Independence:**
```python
from multiprocessing import Pool

def price_chunk(seed):
    return monte_carlo_price(seed)

with Pool() as p:
    results = p.map(price_chunk, seeds)
```

**Speedup:** Near-linear with CPU cores

---

#### GPU Acceleration (Future)

**CuPy / JAX:**
```python
import cupy as cp

# Generate paths on GPU
dW = cp.random.randn(n_paths, n_steps)
S = S0 * cp.exp(cp.cumsum((r - 0.5*sigma**2)*dt + sigma*sqrt(dt)*dW, axis=1))
```

**Speedup:** 10-100x for large-scale simulations

---

## Architecture Principles

### 1. Modularity
- **Registry Pattern:** Models and options registered dynamically
- **Factory Functions:** `build_model()`, `create_instrument()`
- **Loose Coupling:** RNG, model, pricing engine are independent

### 2. Extensibility
- **Add New Model:** Inherit from `BaseModel`, implement `generate_paths()`
- **Add New Option:** Inherit from `BaseInstrument`, implement `payoff()`
- **No Core Changes Required**

### 3. Type Safety
- **100% Type Hints:** All functions annotated
- **Runtime Validation:** Pydantic schemas for configuration
- **Static Analysis:** MyPy compatible

### 4. Reproducibility
- **Seed Control:** Every RNG has explicit seed
- **Deterministic:** Same inputs → same outputs
- **Logging:** All parameters logged for auditing

### 5. Numerical Stability
- **Underflow Protection:** `exp(x)` clamped for large |x|
- **Positive Definiteness:** Correlation matrices checked
- **Variance Positivity:** Heston/SABR enforce v > 0

---

## Usage Examples

### Example 1: European Call on GBM
```python
from systematix import build_model, create_instrument, MonteCarloEngine

# Market data
market = {"spot": 100, "rate": 0.05, "dividend": 0.0, "time_to_maturity": 1.0}

# Model
model = build_model("gbm", market, volatility=0.2)

# Option
option = create_instrument("european_call", strike=100, maturity=1.0)

# Pricing
mc = MonteCarloEngine(num_paths=100000, num_timesteps=252)
result = mc.price(model, option, market)

print(f"Price: {result['price']:.4f}")
print(f"Std Error: {result['std_error']:.4f}")
```

---

### Example 2: American Put on Heston
```python
model = build_model(
    "heston", 
    market,
    v0=0.04,
    kappa=2.0,
    theta=0.04,
    xi=0.3,
    rho=-0.7
)

option = create_instrument("american_put", strike=100, maturity=1.0)

mc = MonteCarloEngine(num_paths=50000, num_timesteps=100)
result = mc.price(model, option, market)
```

---

### Example 3: Greeks for Barrier Option
```python
from systematix.analytics import GreeksComputer

option = create_instrument("up_and_out_call", strike=100, barrier=120, maturity=1.0)

greeks = GreeksComputer(model, option, market)
all_greeks = greeks.compute_all()

print(f"Delta: {all_greeks['delta']:.4f}")
print(f"Gamma: {all_greeks['gamma']:.4f}")
print(f"Vega: {all_greeks['vega']:.4f}")
```

---

### Example 4: Multi-Asset Basket
```python
model = build_model(
    "multi_asset",
    market,
    spots=[100, 110, 95],
    vols=[0.2, 0.25, 0.18],
    correlation_matrix=[[1.0, 0.5, 0.3],
                        [0.5, 1.0, 0.4],
                        [0.3, 0.4, 1.0]]
)

option = create_instrument(
    "basket_call",
    strike=100,
    weights=[0.4, 0.4, 0.2],
    maturity=1.0
)

result = mc.price(model, option, market)
```

---

### Example 5: Variance Reduction
```python
mc = MonteCarloEngine(
    num_paths=50000,
    num_timesteps=252,
    variance_reduction="antithetic"  # or "control_variates"
)

result = mc.price(model, option, market)

print(f"Variance Reduction: {result['variance_reduction_factor']:.2f}x")
```

---

## References

### Books
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*. Springer.
- Shreve, S. (2004). *Stochastic Calculus for Finance II: Continuous-Time Models*. Springer.
- Joshi, M. (2008). *C++ Design Patterns and Derivatives Pricing*. Cambridge University Press.

### Papers
- Longstaff, F. & Schwartz, E. (2001). "Valuing American Options by Simulation: A Simple Least-Squares Approach." *Review of Financial Studies*, 14(1), 113-147.
- Heston, S. (1993). "A Closed-Form Solution for Options with Stochastic Volatility." *Review of Financial Studies*, 6(2), 327-343.
- Hagan, P. et al. (2002). "Managing Smile Risk." *Wilmott Magazine*.

### Online Resources
- QuantLib: https://www.quantlib.org/
- Quantitative Finance Stack Exchange: https://quant.stackexchange.com/

---

## License

This software is provided for educational and research purposes.

