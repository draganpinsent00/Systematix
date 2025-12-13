# Systematix Pro — Multi-Model Options Pricing Dashboard

## Overview

Systematix Pro is a production-ready options pricing platform with support for multiple underlying models, custom payoff functions, Monte Carlo Greeks computation, and professional 3-column layout for research and trading workflows.

### Supported Models

1. **GBM** — Geometric Brownian Motion (classic equity model)
2. **Heston** — Stochastic volatility (equity)
3. **Merton** — Jump-diffusion (equity with jumps)
4. **Kou** — Double exponential jump process (asset with asymmetric jumps)
5. **G2++** — Two-factor Gaussian interest-rate model

### Key Features

- **Custom Payoff Functions**: Write arbitrary Python payoff(S) in the UI
- **Monte Carlo Greeks**: Delta, Gamma, Vega, Rho, Theta with confidence intervals (Common Random Numbers method)
- **Model-Specific Parameters**: Full control over calibrated and user-tuned inputs for each model
- **Path Visualization**: Terminal distribution + optional sample paths
- **History Tracking**: All simulations logged with downloadable CSV
- **Professional 3-Column Layout**:
  - **Left (1/4 width)**: Model selection, custom payoff, tunable parameters
  - **Middle (1/2 width)**: Pricing results, Greeks, visualizations
  - **Right (1/4 width)**: Simulation history with export

---

## Installation & Setup

### Requirements
- Python 3.9+
- Streamlit, NumPy, Pandas, Plotly, SciPy

### Installation

```bash
# Create virtual environment
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install streamlit scipy
```

### Running the Dashboard

```bash
# New professional dashboard (recommended)
py -3 -m streamlit run dashboard_v2.py

# Legacy dashboard (deprecated)
py -3 -m streamlit run dashboard.py
```

The app opens in your browser at `http://localhost:8501`

---

## Model Documentation

### 1. GBM — Geometric Brownian Motion

**Parameters:**
- `S0` (Spot): Initial asset price
- `r` (Risk-free rate): Drift of the asset
- `σ` (Volatility): Annualized asset volatility
- `T` (Maturity): Time to expiration (years)

**Dynamics:** `dS/S = r dt + σ dW`

**Use case:** Simple equity option pricing; baseline comparisons.

---

### 2. Heston — Stochastic Volatility Model

**Parameters:**
- `v0` (Initial variance): Starting volatility² level
- `κ` (Kappa, mean-reversion speed): How fast vol reverts to long-run mean
- `θ` (Theta, long-run variance): Long-run average variance
- `ξ` (Xi, vol-of-vol): Volatility of volatility
- `ρ` (Rho, correlation): Correlation between spot and vol increments

**Dynamics:**
```
dS/S = r dt + √v dW_S
dv = κ(θ - v) dt + ξ√v dW_v,  corr(dW_S, dW_v) = ρ
```

**Use case:** Equity options with stochastic volatility; smile/skew modeling.

**Typical params (equity):**
- v0 = 0.04, κ = 1.5, θ = 0.04, ξ = 0.3, ρ = -0.7

---

### 3. Merton — Jump-Diffusion Model

**Parameters:**
- `σ` (Volatility): Continuous diffusion volatility
- `λ` (Jump intensity): Poisson rate per year
- `μ_J` (Jump mean log-return): Mean of log-jump size
- `σ_J` (Jump volatility): Volatility of jump size

**Dynamics:** `dS/S = (r - λk) dt + σ dW + (J-1) dN_t`
where `J = exp(μ_J + σ_J Z)`, `N_t` ~ Poisson(λt)

**Use case:** Equity options with tail risk; event-driven pricing.

**Typical params (equity):**
- λ = 0.1/year, μ_J = 0.0, σ_J = 0.3

---

### 4. Kou — Double Exponential Jump Process

**Parameters:**
- `σ` (Volatility): Continuous component vol
- `λ` (Jump intensity): Poisson rate per year
- `p` (Up jump probability): Probability of upward jump
- `η⁺` (Up jump parameter): Exponential parameter for up jumps
- `η⁻` (Down jump parameter): Exponential parameter for down jumps

**Dynamics:** Diffusion + Poisson jumps with asymmetric exponential sizes
- Up jumps: log(J) ~ Exp(η⁺) with probability p
- Down jumps: log(J) ~ -Exp(η⁻) with probability (1-p)

**Use case:** Asset with asymmetric jump risk; commodity/FX modeling.

**Typical params:**
- λ = 0.1, p = 0.5, η⁺ = 1.0, η⁻ = 2.0

---

### 5. G2++ — Two-Factor Gaussian Interest-Rate Model

**Parameters:**
- `r0` (Initial short rate): Starting interest rate level
- `a` (Mean-reversion factor 1): Reversion speed of first factor
- `b` (Mean-reversion factor 2): Reversion speed of second factor
- `σ` (Vol factor 1): Volatility of first factor
- `η` (Vol factor 2): Volatility of second factor
- `ρ` (Correlation): Correlation between the two factors

**Dynamics:**
```
dr = a(φ(t) - r) dt + σ dW1 + η dW2,  corr(dW1, dW2) = ρ
```

**Use case:** Interest-rate derivative pricing; bond option valuation.

**Typical params (rate markets):**
- r0 = 0.03, a = 0.1, b = 0.1, σ = 0.015, η = 0.025, ρ = 0.8

---

## Custom Payoff Functions

### Overview

The dashboard allows you to write arbitrary Python payoff functions. The function receives an array `S` of spot prices and returns payoff(S).

### Example Payoffs

#### Standard European Call
```python
def custom_payoff(S):
    return np.maximum(S - 100, 0)
```

#### Straddle (Long Call + Long Put)
```python
def custom_payoff(S):
    K = 100
    return np.maximum(S - K, 0) + np.maximum(K - S, 0)
```

#### Barrier (Up-and-Out Call, H=120)
```python
def custom_payoff(S):
    # S is shape (n_paths, steps+1)
    # Check if price ever hit barrier
    hit_barrier = np.any(S > 120, axis=1)  # (n_paths,)
    payoffs = np.maximum(S[:, -1] - 100, 0)
    payoffs[hit_barrier] = 0  # Knock out
    return payoffs
```

#### Digital (Binary) Option
```python
def custom_payoff(S):
    return np.where(S > 100, 1.0, 0.0)
```

### Restrictions
- Only import NumPy (as `np`) via `import numpy as np` is allowed for safety
- Functions must be vectorized (accept array S, return array payoffs)
- Use `np.maximum`, `np.minimum`, `np.where` for conditional logic

---

## Monte Carlo Greeks

Compute Greeks using the **Common Random Numbers (CRN)** finite-difference method:

- **Delta (Δ)**: Price sensitivity to spot; uses path scaling
- **Gamma (Γ)**: Delta sensitivity; second-order finite difference
- **Vega (ν)**: Price sensitivity to volatility; CRN resimulation
- **Rho (ρ)**: Price sensitivity to interest rates; CRN resimulation
- **Theta (Θ)**: Price sensitivity to time decay; CRN resimulation

Each Greek includes a standard-error estimate for confidence intervals.

### How It Works

1. **Base simulation**: Generate n_paths paths under base parameters
2. **Price base**: Compute payoff and option price
3. **Bumped simulations**: Re-simulate with +/- parameter bumps using same random seeds (CRN)
4. **Finite difference**: Greeks = (Price_up - Price_down) / (2 × bump)

**Advantages of CRN:**
- Lower variance than non-CRN estimators
- More stable across different models and payoff structures

---

## History Tracking

Every simulation is logged with:
- Model type
- Option type (Call/Put/Custom)
- Computed price and standard error
- Spot, strike, volatility, time-to-maturity
- Timestamp (implicit in table order)

### Download & Export

Use the **"Download history (CSV)"** button in the right panel to export all simulations for:
- Backtesting and model validation
- Integration with downstream risk systems
- Compliance and audit trails

---

## Workflow Example: Heston European Call

1. **Select Model**: Heston
2. **Set Market Params**:
   - S0 = 100, K = 100, r = 0.02, σ = 0.2, T = 1.0
3. **Tune Heston Params**:
   - v0 = 0.04, κ = 1.5, θ = 0.04, ξ = 0.3, ρ = -0.7
4. **Set Simulation**:
   - Steps = 12, Paths = 5000, Seed = 42
5. **Select Payoff**:
   - Leave custom payoff blank; choose "Call"
6. **Run Simulation**
   - View price & CI in metrics
   - See terminal distribution in viz
7. **Compute Greeks**
   - Delta, gamma, vega, rho, theta with confidence bands
8. **Review History**
   - All runs logged for sensitivity analysis

---

## Common Tasks

### Sensitivity Analysis
- **Vary σ**: Rerun with σ = 0.15, 0.20, 0.25; observe price impact
- **Vary ρ (Heston)**: Sweep ρ from -0.9 to 0; watch vol-spot correlation effect
- **Compare models**: Run same strike/maturity across GBM, Heston, Merton

### Calibration Workflow
1. Price observed market strike prices in each model
2. Adjust parameters to minimize squared price errors
3. Use history export for model comparison metrics

### Risk Management
- Export history of daily pricing runs
- Compute P&L attribution to spot, vol, time, jump intensity
- Validate Greeks for hedging decisions

---

## Technical Notes

### Performance

- **n_paths = 2000, steps = 12**: ~0.5-1 sec per simulation
- **Heston/Merton/Kou**: Slightly slower than GBM due to loop-based jump/vol simulation
- **G2++**: Fast; only interest-rate simulation, not equity-specific

### Variance Reduction

The dashboard uses **antithetic variates** by default (negated paths). For further variance reduction:
- Increase n_paths (most robust)
- Use lower interest rates (less discounting noise)
- Adjust seed to find better Monte Carlo sample

### Monte Carlo Greeks

- Bump sizes (h_S, h_sigma, h_r, h_T) are automatically selected based on parameters
- More paths → lower stderr on Greeks
- For stable gamma: use n_paths ≥ 5000

---

## Troubleshooting

### Simulation slow / memory issue
- Reduce n_paths (try 1000)
- Reduce steps (try 6 for quick tests)

### Custom payoff errors
- Ensure code uses only `np` (NumPy)
- Test with simple payoff first: `def custom_payoff(S): return np.maximum(S - 100, 0)`
- Check that function returns array of same shape as S (or broadcasts correctly)

### Greeks have high stderr
- Increase n_paths to 5000+
- Reduce other parameters (T, σ) to simplify landscape

### G2++ results seem off
- G2++ is a rate model; for equity options use GBM, Heston, Merton, or Kou
- The current dashboard applies G2++ rates directly; full hybrid pricing (stochastic rates + equity) is a future enhancement

---

## Version History

- **v2.0** (Current): Multi-model, custom payoff, professional 3-column layout
- **v1.0** (Legacy): Single-model GBM/Heston, deprecated

---

## Support & Contributing

For issues, feature requests, or contributions:
1. Check existing issues in GitHub
2. Submit a detailed bug report with model, parameters, and error message
3. Contributions: fork, create feature branch, submit PR with tests

---

## License & Attribution

Systematix Pro — Financial Software
Developed as part of systematic options pricing research.

Built with:
- NumPy, SciPy, Pandas, Plotly
- Streamlit for UI
- Custom Monte Carlo & calibration engines

Enjoy responsible quantitative finance! 📈

