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
====================================================================
...
✅ Test completed successfully!
...
```

### Running Streamlit Dashboard
```bash
streamlit run app.py
```

Then navigate to `http://localhost:8501` in your browser.

## Quick Example (Python API)

```python
from core.rng_engines import create_rng
from models.gbm import GBM
from instruments.registry import create_instrument
from core.mc_engine import MonteCarloEngine

# Setup
rng = create_rng("mersenne", seed=42)
gbm = GBM(
    spot=100.0,
    risk_free_rate=0.05,
    dividend_yield=0.0,
    initial_volatility=0.20,
    time_to_maturity=1.0,
)

# Generate paths
paths = gbm.generate_paths(
    rng,
    num_paths=10000,
    num_steps=252,
    distribution="normal",
    antithetic_variates=True,
)

# Price European call
option = create_instrument("european_call", strike=100.0)
mc_engine = MonteCarloEngine(rng, num_simulations=10000, num_timesteps=252)
result = mc_engine.price(
    paths,
    option.payoff,
    risk_free_rate=0.05,
    time_to_maturity=1.0,
)

print(f"Price: ${result.price:.6f}")
print(f"95% CI: [${result.ci_lower:.6f}, ${result.ci_upper:.6f}]")
```

## Key Design Principles

1. **Clean Modularity** - Single-responsibility files, easy to extend
2. **Registry-Driven** - Models and instruments added without modifying core
3. **Type Safety** - Type hints throughout for IDE support
4. **Numerical Stability** - Safe operations on extreme values
5. **Reproducibility** - Seed-based deterministic execution
6. **Production Quality** - Docstrings, validation, error handling
7. **Professional UI** - Streamlit with Plotly visualization

## Configuration & Schemas

### Dynamic UI Logic
- **RNG Selection:** Choose engine, seed, distribution type
- **Sobol Toggle:** Hides/shows distribution selector
- **Student-t DF:** Only visible when Student-t selected and Sobol OFF
- **Model Selection:** Auto-loads required parameters
- **Option Selection:** Shows only relevant payoff parameters

### Built-in Validation
- Positive spot prices, volatility, time to maturity
- Correlation matrix positive semi-definite checks
- Min/max ranges on all parameters
- Clear error messages with remediation hints

## Advanced Features

### Variance Reduction
- **Antithetic Variates** - Pair paths for lower variance
- **Control Variates** - Benchmark against analytical price
- **Importance Sampling** - Shift distribution for tail risk

### Monte Carlo Enhancements
- **Brownian Bridge** - Quasi-random path interpolation
- **Convergence Monitoring** - Real-time efficiency tracking
- **Confidence Intervals** - Student-t based estimates

### Greeks Computation
- **Finite Difference** - Bump-based sensitivities
- **Path-Wise** - Direct derivative computations (future)
- **Likelihood Ratio Method** - For jump models (future)

## Testing

Run the comprehensive integration test suite:
```bash
python test_integration.py
```

This tests:
- European call pricing vs Black-Scholes
- Asian options
- Barrier options
- All 5 RNG engines
- Heston stochastic volatility model
- Greeks computation
- Risk metrics

## Performance Characteristics

Typical runtime on modern CPU:
- **10,000 paths, 252 steps** - ~2-3 seconds
- **100,000 paths, 252 steps** - ~20-30 seconds
- **1,000,000 paths, 252 steps** - ~200-300 seconds

Speed varies by:
- Model complexity (GBM < Heston < Jump models)
- RNG engine selection
- Variance reduction techniques
- Number of Greeks computed

## Future Enhancements

- [ ] GPU acceleration (CuPy)
- [ ] Parallel path generation
- [ ] Local volatility models
- [ ] Smile calibration
- [ ] XVA adjustments
- [ ] Real market data integration
- [ ] Portfolio-level analytics

## License

Proprietary - Systematix Trading Systems

## Support

For questions or issues reach out to draganpinsent00@gmail.com or refer to inline documentation and docstrings throughout the codebase.



