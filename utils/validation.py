"""
Input validation and error handling.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np


def validate_market_params(params: Dict[str, float]) -> Tuple[bool, Optional[str]]:
    """Validate market parameters."""
    if params.get("spot", 0) <= 0:
        return False, "Spot price must be positive"
    if params.get("time_to_maturity", 0) <= 0:
        return False, "Time to maturity must be positive"
    if params.get("initial_volatility", 0) <= 0:
        return False, "Volatility must be positive"
    return True, None


def validate_mc_settings(settings: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate MC settings."""
    if settings.get("num_simulations", 0) < 100:
        return False, "Number of simulations must be >= 100"
    if settings.get("num_timesteps", 0) < 10:
        return False, "Number of time steps must be >= 10"
    return True, None


def validate_option_params(params: Dict[str, float]) -> Tuple[bool, Optional[str]]:
    """Validate option parameters.

    This validation is flexible: many payoffs use different strike keys
    (for example `strike`, `trigger_strike`, `payoff_strike`).
    We consider any parameter whose name contains the substring "strike"
    (case-insensitive) and check that its numeric value is positive.
    """
    if not params:
        return True, None

    # Find keys that represent strikes (e.g., 'strike', 'trigger_strike', 'payoff_strike')
    strike_keys = [k for k in params.keys() if 'strike' in k.lower()]

    # If any strike-like keys are present, validate them
    for k in strike_keys:
        try:
            val = float(params.get(k, 0))
        except Exception:
            return False, f"{k} must be a numeric value"
        if val <= 0:
            # User-friendly label: convert snake_case to Title Case
            label = k.replace('_', ' ').title()
            return False, f"{label} must be positive"

    # If no strike keys are present, assume validation passes here (other validators may catch missing required params)
    return True, None


def validate_correlation_matrix(corr: np.ndarray) -> Tuple[bool, Optional[str]]:
    """Validate correlation matrix is PSD."""
    try:
        eigenvalues = np.linalg.eigvalsh(corr)
        if np.min(eigenvalues) < -1e-6:
            return False, f"Correlation matrix not PSD. Min eigenvalue: {np.min(eigenvalues):.6f}"
        return True, None
    except Exception as e:
        return False, f"Correlation matrix error: {str(e)}"
