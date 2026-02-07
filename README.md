# Systematix - Monte Carlo Options Pricing Platform

A production-grade Python platform for pricing derivatives using Monte Carlo simulation. Provides 7 stochastic models, 50 option types, 5 RNG engines, and professional analytics via a Streamlit dashboard.

## Features

- **7 Stochastic Models**: GBM, Heston, 3/2 Heston, Merton Jump, Kou Jump, SABR, Multi-Asset
- **50 Option Types**: European, Asian, Barrier, Exotic, Compound, Multi-Asset, and more
- **5 RNG Engines**: Mersenne Twister, PCG64, XorShift, Philox, Middle-Square (user-selectable)
- **3 Distribution Types**: Normal, Student-t (heavy-tailed), Sobol (quasi-random)
- **Greeks**: Delta, Gamma, Vega, Theta, Rho via finite difference
- **Risk Analytics**: VaR, CVaR, convergence diagnostics, path statistics
- **Professional UI**: Single-page Streamlit dashboard with Plotly charts

## Requirements

- Python 3.9+
- Dependencies: streamlit, numpy, scipy, pandas, plotly, scikit-learn

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

**Workflow:**
1. Configure market parameters (spot, rate, volatility, time)
2. Select RNG engine and distribution
3. Choose stochastic model and option type
4. Click "Run Pricing"
5. View results: price, confidence intervals, Greeks, diagnostics

## Key Capabilities

- Monte Carlo pricing with confidence intervals
- Dynamic input validation and error handling
- Variance reduction techniques (antithetic variates, control variates)
- Reproducible results via seed-based RNG
- Configuration export for audit trails
- Supports single and multi-asset scenarios

## Documentation

- **PROJECT_SUMMARY.md** - Non-technical overview and capabilities
- **ARCHITECTURE.md** - System design and implementation details


#### Rainbow (2)
- Rainbow Max Call
- Rainbow Min Put

#### Forward-Start (2)
- Forward-Start Call
- Forward-Start Put

#### Path-Dependent (2)
- Cliquet Option
- Variance Swap

**Total: 50 option types**

### Analytics & Risk
- **Pricing** - Confidence intervals, convergence analysis
- **Greeks** - Delta, Gamma, Vega, Theta, Rho via finite difference
- **Risk Metrics** - VaR, CVaR, skewness, kurtosis
- **Diagnostics** - Autocorrelation, path statistics, convergence rates

### Streamlit Dashboard
- **Dynamic UI** - All inputs change based on model and option selection
- **Tab-Based Layout:**
  - **Inputs** - Market, RNG, model, option parameters
  - **Results** - Price, confidence intervals, path visualization
  - **Diagnostics** - Convergence analysis, path statistics
  - **Greeks** - Sensitivity analysis
  - **Risk** - VaR/CVaR, distribution analysis
  - **Scenarios** - Parameter sweep analysis
- **Professional Plotly Charts** - Unified theme, interactive visualization
- **Session State Management** - Clean config tracking, resumable sessions

## Project Structure

```
Systematix/
├── app.py                      # Streamlit entrypoint
├── test_integration.py         # Integration tests
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── config/
│   ├── settings.py            # Global configuration constants
│   └── schemas.py             # Input schemas, registries, UI metadata
│
├── core/
│   ├── rng_engines.py         # 5 RNG implementations
│   ├── rng_distributions.py   # Normal, Student-t, Sobol transforms
│   ├── brownian.py            # BM generation, Brownian bridge
│   ├── mc_engine.py           # MC pricing orchestration
│   ├── variance_reduction.py  # Antithetic, control variates, IS
│   ├── lsm.py                 # Longstaff-Schwartz for American
│   └── numerics.py            # Stability, discretization helpers
│
├── models/
│   ├── base.py                # Abstract stochastic model
│   ├── gbm.py                 # Geometric Brownian Motion
│   ├── heston.py              # Standard Heston
│   ├── heston_3_2.py          # 3/2 Heston variant
│   ├── merton_jump.py         # Merton jump diffusion
│   ├── kou_jump.py            # Kou double exponential
│   ├── sabr.py                # SABR model
│   ├── multi_asset.py         # Multi-asset correlated
│   └── sobol_wrapper.py       # Sobol sequence interface
│
├── instruments/
│   ├── base.py                # Abstract instrument
│   ├── registry.py            # Instrument factory, 50 types
│   ├── payoffs_vanilla.py     # European, digital, gap, American
│   ├── payoffs_exotic.py      # Asian, barrier, lookback, compound
│   ├── payoffs_rates_fx.py    # Multi-asset, rainbow, variance swap
│   └── custom_payoff.py       # User-defined payoff builder
│
├── analytics/
│   ├── pricing.py             # Black-Scholes benchmark
│   ├── greeks.py              # Greeks computation
│   ├── risk.py                # VaR, CVaR, statistics
│   ├── diagnostics.py         # Convergence, path analysis
│   └── calibration.py         # Model calibration (scaffolding)
│
├── visualization/
│   ├── plotly_theme.py        # Professional Plotly styling
│   ├── charts_paths.py        # Path and distribution plots
│   ├── charts_payoffs.py      # Payoff diagrams, P&L, Greeks
│   └── charts_diagnostics.py  # Convergence, VaR, diagnostics
│
├── ui/
│   ├── layout.py              # Page structure, tabs
│   ├── components.py          # Input widgets, messages
│   ├── dynamic_forms.py       # Registry-driven form generation
│   └── state.py               # Session state management
│
└── utils/
    ├── validation.py          # Input validation
    ├── io.py                  # Import/export utilities
    └── logging.py             # Application logging
```

## Installation

### 1. Clone Repository
```bash
cd C:\Users\smcin\PycharmProjects\Systematix
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Running Integration Tests
```bash
python test_integration.py
```

Expected output:
```
╔════════════════════════════════════════════════════════════╗
║          SYSTEMATIX INTEGRATION TEST SUITE                 ║
╚════════════════════════════════════════════════════════════╝

Testing European Call Option
