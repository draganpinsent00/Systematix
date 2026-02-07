# Systematix Architecture

## System Overview

Systematix is a modular, extensible Monte Carlo pricing platform with clean separation of concerns across RNG, models, instruments, analytics, and UI layers.

## Module Organization

### config/
- **settings.py** - Global constants
- **schemas.py** - Model and option registries, UI metadata

### core/
- **rng_engines.py** - 5 RNG implementations
- **rng_distributions.py** - Normal, Student-t, Sobol transforms
- **brownian.py** - Path generation
- **mc_engine.py** - Monte Carlo orchestration
- **variance_reduction.py** - Variance reduction techniques
- **lsm.py** - Longstaff-Schwartz for American options
- **numerics.py** - Numerical stability

### models/
7 implementations: GBM, Heston, 3/2 Heston, Merton Jump, Kou Jump, SABR, Multi-Asset

### instruments/
50 option types across 3 payoff files (vanilla, exotic, multi-asset)

### analytics/
- **pricing.py** - Black-Scholes benchmarking
- **greeks.py** - Delta, Gamma, Vega, Theta, Rho
- **risk.py** - VaR, CVaR, statistics
- **diagnostics.py** - Convergence analysis

### visualization/
Plotly charts: paths, payoffs, Greeks, diagnostics

### ui/
- **components.py** - Input widgets
- **dynamic_forms.py** - Registry-driven forms
- **state.py** - Session state

### utils/
- **validation.py** - Input validation
- **io.py** - Config import/export
- **logging.py** - Application logging

## Design Principles

1. **Modularity** - Each file handles one concept
2. **Registry-Driven** - Models and options registered centrally, UI auto-updates
3. **Factory Pattern** - Decoupled instantiation via build_model(), create_instrument()
4. **Type Safety** - 100% type-hinted functions
5. **Configuration as Data** - Immutable, reproducible configs
6. **Clean State** - Streamlit session state without globals

## Data Flow

1. User configures market, RNG, model, option in UI
2. RNG engine creates reproducible stream (user-selected engine + seed)
3. Model generates correlated price paths
4. Instrument computes payoffs from paths
5. MC engine discounts to present value
6. Analytics compute Greeks, risk metrics, diagnostics
7. Charts visualize results

## Key Concepts

**RNG Abstraction**: 5 engines (Mersenne, PCG64, XorShift, Philox, Middle-Square) + 3 distributions (Normal, Student-t, Sobol)

**Model Abstraction**: Common interface (generate_paths), self-contained implementations

**Instrument Abstraction**: Payoff functions take entire path, return scalar per path (enables exotic options)

**MC Engine**: Orchestrates path generation, payoffs, discounting, confidence intervals

## Extension Points

- **New RNG**: Implement RNGEngineBase, register in config/schemas.py
- **New Model**: Implement StochasticModel, register in MODEL_REGISTRY
- **New Option**: Implement Instrument, register in INSTRUMENT_REGISTRY
- **New Greek**: Add method to GreeksComputer
- **New Risk Metric**: Add method to RiskAnalyzer

## Testing

- **smoke_test.py** - Module import validation
- **test_integration.py** - 5 end-to-end workflows
- **validate_imports.py** - Model validation
- Input validation, PSD checks, convergence diagnostics

## Guarantees

- **Reproducibility**: Same seed + config = identical results
- **Type Safety**: All functions type-hinted
- **Numerical Stability**: Safe operations, PSD validation
- **Extensibility**: Add features without modifying core
- **Clean Code**: < 500 lines per file

