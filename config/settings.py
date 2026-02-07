"""
Global configuration and constants.
"""

from dataclasses import dataclass

# MC Configuration
DEFAULT_NUM_SIMULATIONS = 10000
DEFAULT_NUM_TIMESTEPS = 252
DEFAULT_SEED = 42
DEFAULT_RNG_ENGINE = "mersenne"
DEFAULT_DISTRIBUTION = "normal"
DEFAULT_SOBOL = False
DEFAULT_ANTITHETIC = True
DEFAULT_CONTROL_VARIATE = False
DEFAULT_IMPORTANCE_SAMPLING = False

# Greeks Configuration
GREEKS_BUMP = 0.01  # 1% bump for finite difference
PATHWISE_CAPABLE = ["european_call", "european_put", "asian_arithmetic_call", "asian_arithmetic_put"]

# Risk Configuration
VAR_CONFIDENCE = 0.95
CVAR_CONFIDENCE = 0.95

# Numerical
MIN_VARIANCE = 1e-10
MAX_CORRELATION = 0.9999
MIN_TIME_TO_MATURITY = 1/252  # 1 trading day

# Display
NUM_PATHS_PLOT = 100  # paths shown in visualization
DECIMAL_PLACES = 6

