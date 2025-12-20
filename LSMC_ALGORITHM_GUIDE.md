# LEAST SQUARES MONTE CARLO (LSMC) ALGORITHM DETAILS

## Overview
Longstaff-Schwartz method (2001) for pricing American options using Monte Carlo simulation.

**Key Idea**: Use regression to estimate continuation values, enabling optimal early exercise decisions.

---

## Algorithm

### Inputs
- `paths`: Simulated spot price paths (num_paths × (num_steps + 1))
- `intrinsic_payoff`: Function to compute immediate exercise value
- `dt`: Time step size
- `r`: Risk-free rate
- `degree`: Polynomial degree for regression (default: 2)

### Outputs
- `price`: American option value (mean across paths)
- `cashflows`: Optimal discounted payoff per path

---

## Backward Induction

### Step 0: Initialize at Maturity
```
cashflow[i] = intrinsic_payoff(S_T[i]) for each path i
```
At maturity, the option holder exercises if in-the-money.

### Step 1-T-1: Backward Induction
At each time step `t` from `T-1` down to `1`:

#### 1.1: Compute Intrinsic Value
```
intrinsic[t, i] = intrinsic_payoff(S_t[i]) for each path i
```
This is the immediate payoff if we exercise at time `t`.

#### 1.2: Identify In-The-Money Paths
```
itm_mask = (intrinsic[t] > 0)
num_itm = sum(itm_mask)
```
Only use ITM paths for regression (reduces noise).

#### 1.3: Regression Setup
Build polynomial basis for ITM paths:
```
X_itm = [1, S_t^(itm), (S_t^(itm))^2, ..., (S_t^(itm))^degree]
y_itm = discount * cashflow[itm]
```

Where:
- `X_itm`: Design matrix (num_itm × (degree+1))
- `y_itm`: Discounted continuation values for ITM paths
- `discount = exp(-r * dt)`

#### 1.4: Regression
Solve least-squares problem:
```
X_itm @ coeffs = y_itm  (in least-squares sense)
coeffs = (X_itm^T X_itm)^(-1) X_itm^T y_itm
```

This estimates the relationship between spot price and continuation value.

#### 1.5: Continuation Value Estimation
Evaluate fitted polynomial at ALL paths (ITM + OTM):
```
X_all = [1, S_t, S_t^2, ..., S_t^degree]
continuation_fit = X_all @ coeffs
continuation_fit = max(continuation_fit, 0)  # Ensure non-negative
```

#### 1.6: Exercise Decision
Compare immediate payoff vs. estimated continuation:
```
exercise_now[i] = (intrinsic[t, i] > continuation_fit[i])
```

If the immediate payoff exceeds the estimated value of continuing, exercise now.

#### 1.7: Update Cashflow
```
for each path i:
    if exercise_now[i]:
        cashflow[i] = intrinsic[t, i]
    else:
        cashflow[i] = discount * cashflow[i]
```

---

## Basis Functions

The polynomial basis determines what continuation value relationships we can capture:

### Degree 1 (Linear)
```
φ(S) = [1, S]
```
Linear relationship between spot and continuation.

### Degree 2 (Quadratic) - RECOMMENDED
```
φ(S) = [1, S, S²]
```
Allows for non-linear continuation value (convexity).
**Best balance of accuracy and efficiency.**

### Degree 3 (Cubic)
```
φ(S) = [1, S, S², S³]
```
Captures more complex continuation patterns.
May overfit with fewer ITM paths.

---

## Why ITM-Only Regression?

**Problem**: OTM paths have zero payoff, which biases the regression.

**Solution**: Regress only on ITM paths where:
- Intrinsic value > 0
- Exercise decision is meaningful
- Continuation value matters

**Benefit**: Cleaner estimate of true continuation value function.

---

## Intuition

### Early Exercise Value
The option value = maximum of:
1. **Immediate Exercise**: intrinsic value now
2. **Continue Holding**: discounted future value

### Regression's Role
We can't know the exact future value, so we:
1. Use paths to simulate futures
2. Use regression to smooth/estimate the continuation value
3. The polynomial basis allows for reasonable approximations

### Why It Works
- Large sample size: enough paths for stable regression
- Backward induction: uses realized future values (not forecasted)
- Regression: captures relationship between spot and continuation

---

## Example: American Put

### Setup
- S₀ = 100, K = 110, T = 1yr, r = 0.05, σ = 0.2
- 10,000 paths, 252 steps

### Maturity (t = T)
```
cashflow = max(110 - S_T, 0)
```
Exercise if S_T < 110.

### Backward Step (t = 1 month)
```
intrinsic = max(110 - S_{t}, 0)
itm_paths = {i : S_{t,i} < 110}  // In-the-money

// Regression on ITM paths
X_itm = [1, S_{t,itm}, S_{t,itm}²]
y_itm = exp(-0.05/12) * cashflow_{t+1,itm}

coeffs = solve_lstsq(X_itm, y_itm)

// Continuation value for all paths
continuation = [1, S_t, S_t²] @ coeffs

// Exercise decision
exercise = (intrinsic > continuation)

// Update
cashflow[i] = intrinsic[i] if exercise[i] else discount * cashflow[i]
```

American puts often exercise early (high value in-the-money).

---

## Numerical Stability

### Issues Handled

1. **Singular Regression Matrix**
   - Solution: Use least-squares solver with regularization
   - Fallback: Use mean continuation value

2. **Negative Continuation Values**
   - Can occur with noisy regression
   - Solution: Enforce `continuation = max(continuation, 0)`

3. **Insufficient ITM Paths**
   - Need at least degree+1 paths to fit polynomial
   - Solution: Skip regression if too few ITM paths

4. **Extreme Spot Values**
   - High powers (S³) can overflow
   - Solution: Normalize or use lower degree

---

## Convergence Properties

### Convergence Rate
- **Bias**: O(1/√N) where N = number of paths
- **Variance**: Decreases with more paths
- **Overall**: Standard Monte Carlo rate: O(1/√N)

### Practical Guidance
- **10,000 paths**: Good for most applications
- **50,000 paths**: High precision
- **1,000 paths**: Fast but noisier
- **100,000 paths**: Research/publication quality

---

## Variance Reduction

### Common Random Numbers (CRN)
Use same random seed for Greeks computation:
```python
rng_up = np.random.default_rng(seed=42)
rng_down = np.random.default_rng(seed=42)
```
This reduces variance of finite-difference estimates.

### Antithetic Variates
Pair paths with opposite shocks:
```python
half = num_paths // 2
Z = np.vstack([Z[:half], -Z[:half]])
```
Reduces path generation variance.

### Importance Sampling
Can be combined with LSMC for further variance reduction
(not implemented in current code).

---

## Comparison to Binomial Trees

| Aspect | LSMC | Binomial Tree |
|--------|------|---------------|
| **Scalability** | Excellent (high dimensions) | Poor (curse of dimensionality) |
| **Accuracy** | Good (polynomial fit) | Excellent (exact) |
| **Speed** | Fast | Slow for many steps |
| **Implementation** | Moderate | Simple |
| **Multi-asset** | Easy | Very hard |

LSMC is preferred for:
- American options on multi-asset products
- Non-Markovian payoffs
- Exotic options

---

## References

- Longstaff, F. A., & Schwartz, E. S. (2001). "Valuing American options by simulation: a simple least-squares approach." *The Review of Financial Studies*, 14(1), 113-147.

---

## Summary

LSMC is a powerful, practical algorithm for American option pricing that:
1. ✅ Works in any dimension
2. ✅ Produces accurate prices
3. ✅ Handles complex payoffs
4. ✅ Scales to large simulations
5. ✅ Enables variance reduction techniques

The key insight is using regression to estimate continuation values, enabling optimal early exercise decisions in a Monte Carlo framework.

