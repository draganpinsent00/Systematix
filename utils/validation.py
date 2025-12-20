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
    """Validate option parameters."""
    if params.get("strike", 0) <= 0:
        return False, "Strike must be positive"
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

