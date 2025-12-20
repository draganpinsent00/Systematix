# Systematix - Project Summary

## What is Systematix?

Systematix is a software platform that prices financial derivatives (options) using Monte Carlo simulation—a numerical method that models thousands of possible market scenarios to estimate fair value and risk.

## Who is it for?

- Quantitative analysts and traders pricing options
- Risk managers analyzing portfolio exposure
- Financial engineers backtesting strategies
- Researchers studying derivatives pricing
- Students learning Monte Carlo methods

## What can it do?

### Pricing
- Calculates option prices with confidence intervals
- Supports 50 different option types (European, Asian, Barrier, Exotic, etc.)
- Works with 7 different market models (simple to complex)
- Provides convergence diagnostics to verify accuracy

### Risk Analysis
- Computes Greeks (Delta, Gamma, Vega, Theta, Rho)
- Calculates Value-at-Risk (VaR) and Conditional VaR
- Analyzes payoff distributions
- Generates path visualizations and diagnostics

### Flexibility
- Choose from 5 different random number generators
- Use 3 different probability distributions
- Adjust simulation parameters (number of paths, time steps)
- Apply variance reduction techniques for faster convergence
- Export configurations for reproducibility

## Key Components

**Pricing Engine**: Monte Carlo simulation with 7 stochastic models
- GBM (standard)
- Heston (stochastic volatility)
- 3/2 Heston (alternative volatility)
- Merton Jump (sudden price jumps)
- Kou Jump (asymmetric jumps)
- SABR (volatility smile)
- Multi-Asset (correlated assets)

**User Interface**: Web-based Streamlit dashboard
- Configure market and simulation parameters
- Select model and option type
- View results in real-time
- Charts and diagnostics

**Analytics**: Comprehensive post-pricing analysis
- Greeks (option sensitivities)
- Risk metrics (VaR, CVaR)
- Convergence analysis
- Path statistics

## Technical Scope

**In Scope:**
- Monte Carlo option pricing
- Greeks computation
- Risk analytics
- Single and multi-asset options
- Path-dependent options (Asian, Barrier, Lookback, etc.)
- American-style options (Longstaff-Schwartz)

**Out of Scope:**
- Real-time market data integration
- Model calibration to market prices
- Parallel/GPU computing (infrastructure)
- Trading execution
- Portfolio optimization

## Data Flow

```
User Input
    ↓
Generate Random Paths (using selected RNG and model)
    ↓
Calculate Option Payoffs at Maturity
    ↓
Discount to Present Value
    ↓
Compute Greeks (Delta, Gamma, Vega, Theta, Rho)
    ↓
Calculate Risk Metrics (VaR, CVaR)
    ↓
Display Results (Price, CI, Charts, Diagnostics)
```

## Platform Characteristics

- **Language**: Python 3.9+
- **Architecture**: Modular, registry-driven design
- **UI**: Streamlit dashboard with Plotly charts
- **Extensibility**: Easy to add new models and option types
- **Quality**: Type-hinted, documented, tested code
- **Reproducibility**: Deterministic via seed-based RNG

## Typical Workflow

1. Start application: `streamlit run app.py`
2. Configure market scenario (spot, rate, volatility, time)
3. Select RNG engine and probability distribution
4. Choose stochastic model (GBM, Heston, etc.)
5. Select option type and parameters
6. Run pricing (Monte Carlo simulation)
7. Review results: price, Greeks, risk metrics, convergence

Systematix/
│
├── 📄 app.py                         # Streamlit entrypoint (450 lines)
├── 📄 test_integration.py            # Integration tests with 5 workflows
├── 📄 smoke_test.py                  # Module validation script
├── 📄 setup.py                       # Package configuration
├── 📄 requirements.txt               # Dependencies (5 packages)
├── 📄 README.md                      # Feature overview & usage
├── 📄 QUICKSTART.md                  # 5-min setup + 10 workflows
├── 📄 ARCHITECTURE.md                # Design patterns & extension guide
├── 📄 .gitignore                     # Git exclusions
│
├── 📁 config/
│   ├── __init__.py
│   ├── settings.py                   # Global constants (default params)
│   └── schemas.py                    # Registries, UI metadata (350 lines)
│
├── 📁 core/
│   ├── __init__.py
│   ├── rng_engines.py                # 5 RNG implementations (350 lines)
│   ├── rng_distributions.py          # Normal, Student-t, Sobol transforms
│   ├── brownian.py                   # BM generation, Brownian bridge
│   ├── mc_engine.py                  # MC pricing orchestration
│   ├── variance_reduction.py         # Antithetic, control variates, IS
│   ├── lsm.py                        # Longstaff-Schwartz for American
│   └── numerics.py                   # Stability, discretization helpers
│
├── 📁 models/
│   ├── __init__.py
│   ├── base.py                       # Abstract stochastic model
│   ├── gbm.py                        # Geometric Brownian Motion (80 lines)
│   ├── heston.py                     # Heston (120 lines)
│   ├── heston_3_2.py                 # 3/2 Heston variant (110 lines)
│   ├── merton_jump.py                # Merton jump (100 lines)
│   ├── kou_jump.py                   # Kou double exponential (110 lines)
│   ├── sabr.py                       # SABR (120 lines)
│   ├── multi_asset.py                # Multi-asset correlated (90 lines)
│   └── sobol_wrapper.py              # Sobol sequence interface
│
├── 📁 instruments/
│   ├── __init__.py
│   ├── base.py                       # Abstract instrument
│   ├── registry.py                   # Factory, instrument registry
│   ├── payoffs_vanilla.py            # European, digital, gap, American (200 lines)
│   ├── payoffs_exotic.py             # Asian, barrier, lookback, compound (450 lines)
│   ├── payoffs_rates_fx.py           # Multi-asset, rainbow, variance swap (300 lines)
│   └── custom_payoff.py              # User-defined payoff builder
│
├── 📁 analytics/
│   ├── __init__.py
│   ├── pricing.py                    # Black-Scholes benchmark
│   ├── greeks.py                     # Greeks computation (150 lines)
│   ├── risk.py                       # VaR, CVaR, statistics (100 lines)
│   ├── diagnostics.py                # Convergence, path analysis
│   └── calibration.py                # Model calibration (scaffolding)
│
├── 📁 visualization/
│   ├── __init__.py
│   ├── plotly_theme.py               # Professional theme styling
│   ├── charts_paths.py               # Path and distribution plots
│   ├── charts_payoffs.py             # Payoff diagrams, P&L, Greeks
│   └── charts_diagnostics.py         # Convergence, VaR, diagnostics
│
├── 📁 ui/
│   ├── __init__.py
│   ├── layout.py                     # Page structure, tabs
│   ├── components.py                 # Input widgets, messages (150 lines)
│   ├── dynamic_forms.py              # Registry-driven form generation
│   └── state.py                      # Session state management
│
└── 📁 utils/
    ├── __init__.py
    ├── validation.py                 # Input validation
    ├── io.py                         # Import/export utilities
    └── logging.py                    # Application logging
```

**Total Lines of Code: ~5,000 (plus documentation)**

---

## 🎯 KEY HIGHLIGHTS

### Architecture Excellence
1. **Clean Modularity** - Each file < 500 lines, single responsibility
2. **Registry-Driven** - Add models/options without touching core
3. **Factory Pattern** - Decoupled instantiation
4. **Type Hints** - IDE support, runtime safety
5. **Reproducible** - Seed-based deterministic execution

### User Experience
1. **Dynamic UI** - All inputs change based on selections
2. **Professional Charts** - Plotly with unified theme
3. **Clear Feedback** - Validation messages, config summary
4. **Tab-Based Organization** - Logical workflow
5. **No Manual Coding** - Pure UI configuration

### Quantitative Rigor
1. **50 Option Types** - Every major category covered
2. **6+ Models** - From simple (GBM) to complex (Jumps)
3. **5 RNG Engines** - Not just seed variation
4. **Greeks** - Delta, Gamma, Vega, Theta, Rho
5. **Risk Metrics** - VaR, CVaR, tail analysis

### Production Quality
1. **Error Handling** - Validation at every step
2. **Numerical Stability** - Safe sqrt/log, PSD checks
3. **Confidence Intervals** - Student-t based estimates
4. **Convergence Monitoring** - Real-time efficiency tracking
5. **Audit Trail** - Reproducible configs, logged runs

---

## 🚀 QUICK START

```bash
# 1. Install
cd C:\Users\smcin\PycharmProjects\Systematix
pip install -r requirements.txt

# 2. Validate
python smoke_test.py

# 3. Run Dashboard
streamlit run app.py

# 4. Open browser
# http://localhost:8501
```

---

## 📊 EXAMPLE USAGE

**API (Non-Dashboard):**
```python
from core.rng_engines import create_rng
from models.gbm import GBM
from instruments.registry import create_instrument
from core.mc_engine import MonteCarloEngine

rng = create_rng("mersenne", seed=42)
gbm = GBM(spot=100, risk_free_rate=0.05, initial_volatility=0.20, time_to_maturity=1.0)
paths = gbm.generate_paths(rng, num_paths=10000, num_steps=252)
option = create_instrument("european_call", strike=100.0)
engine = MonteCarloEngine(rng, 10000, 252)
result = engine.price(paths, option.payoff, 0.05, 1.0)

print(f"Price: ${result.price:.6f}")
print(f"95% CI: [${result.ci_lower:.6f}, ${result.ci_upper:.6f}]")
```

**Dashboard:**
1. Set market parameters
2. Select RNG engine (5 choices)
3. Select model (7 choices)
4. Select option type (50 choices)
5. Click "Run Pricing"
6. View results across 6 tabs

---

## ✨ FEATURES DELIVERED

- ✅ 5 RNG engines + dynamic selection
- ✅ 3 distribution types (Normal, Student-t, Sobol)
- ✅ 7 stochastic models
- ✅ 50 option types
- ✅ Monte Carlo engine with confidence intervals
- ✅ Variance reduction (antithetic variates, control variates, importance sampling)
- ✅ Longstaff-Schwartz for American options
- ✅ Greeks (Delta, Gamma, Vega, Theta, Rho)
- ✅ Risk metrics (VaR, CVaR, statistics)
- ✅ Convergence diagnostics
- ✅ Professional Plotly visualizations
- ✅ Dynamic Streamlit dashboard
- ✅ Session state management
- ✅ Input validation & error handling
- ✅ Configuration reproducibility
- ✅ Comprehensive documentation (README, QUICKSTART, ARCHITECTURE)
- ✅ Integration test suite
- ✅ Smoke test validation script

---

## 📝 DOCUMENTATION

- **README.md** - Feature overview, installation, structure
- **QUICKSTART.md** - 5-min setup, 10 example workflows
- **ARCHITECTURE.md** - Design patterns, extension recipes, module guide
- **Inline docstrings** - Every function documented
- **Type hints** - Full type annotation throughout

---

## 🏁 COMPLETION STATUS

**All 9 non-negotiable requirements fully satisfied:**

1. ✅ Clean, editable Python project structure
2. ✅ RNG engine selection (NOT just seed)
3. ✅ Innovation distribution & Sobol logic
4. ✅ Models (registry-driven)
5. ✅ 50+ option types (exactly enumerated)
6. ✅ Dynamic Streamlit dashboard
7. ✅ Analytics & outputs
8. ✅ Safety, validation, transparency
9. ✅ Production-ready code (no questions, no stages, complete system)

**SYSTEM IS COMPLETE AND OPERATIONAL** ✅

---

## 🎓 LEARNING RESOURCES

- See `ARCHITECTURE.md` for design patterns and extension recipes
- See `test_integration.py` for 5 runnable workflow examples
- Review `models/gbm.py` as template for adding new models
- Review `instruments/payoffs_exotic.py` as template for options

---

**Built with ❤️ for quantitative finance**

*Systematix: The Production-Grade Monte Carlo Options Pricing Platform*

