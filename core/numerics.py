"""
Numerical stability and discretization helpers.
"""

import numpy as np
from typing import Tuple, Optional


def ensure_positive_variance(sigma: float, min_var: float = 1e-10) -> float:
    """Ensure volatility is above minimum."""
    return max(sigma, np.sqrt(min_var))


def check_psd_correlation(corr: np.ndarray, tol: float = 1e-6) -> Tuple[bool, Optional[str]]:
    """
    Check if correlation matrix is positive semi-definite.

    Args:
        corr: Correlation matrix
        tol: Tolerance for eigenvalue check

    Returns:
        (is_valid, error_message)
    """
    try:
        eigenvalues = np.linalg.eigvalsh(corr)
        min_eigenvalue = np.min(eigenvalues)
        if min_eigenvalue < -tol:
            return False, f"Correlation matrix not PSD. Min eigenvalue: {min_eigenvalue:.6f}"
        return True, None
    except Exception as e:
        return False, f"Correlation matrix error: {str(e)}"


def safe_log(x: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Numerically stable log."""
    x_safe = np.maximum(x, floor)
    return np.log(x_safe)


def safe_sqrt(x: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Numerically stable sqrt."""
    x_safe = np.maximum(x, floor)
    return np.sqrt(x_safe)


def discrete_averaging_schedule(
    num_steps: int,
    averaging_frequency: str = "daily",
) -> np.ndarray:
    """
    Generate averaging dates for Asian options.

    Args:
        num_steps: Total number of time steps
        averaging_frequency: "daily", "weekly", "monthly"

    Returns:
        Array of step indices for averaging
    """
    if averaging_frequency == "daily":
        return np.arange(1, num_steps + 1)
    elif averaging_frequency == "weekly":
        # Assuming 252 trading days per year
        return np.arange(5, num_steps + 1, 5)
    elif averaging_frequency == "monthly":
        return np.arange(21, num_steps + 1, 21)
    else:
        return np.arange(1, num_steps + 1)


def create_time_grid(num_steps: int, time_to_maturity: float) -> np.ndarray:
    """Create time grid."""
    return np.linspace(0, time_to_maturity, num_steps + 1)


def euler_discretization(
    drift: np.ndarray,
    volatility: np.ndarray,
    dW: np.ndarray,
    dt: float,
    current_x: np.ndarray,
) -> np.ndarray:
    """
    Standard Euler discretization step.

    dx = drift * dt + volatility * sqrt(dt) * dW
    """
    sqrt_dt = np.sqrt(dt)
    return current_x + drift * dt + volatility * sqrt_dt * dW


def milstein_discretization(
    drift: np.ndarray,
    volatility: np.ndarray,
    drift_derivative: np.ndarray,
    volatility_derivative: np.ndarray,
    dW: np.ndarray,
    dt: float,
    current_x: np.ndarray,
) -> np.ndarray:
    """
    Milstein discretization (higher order accuracy).

    dx = drift*dt + vol*sqrt(dt)*dW + 0.5*vol'*vol*((dW)^2 - dt)
    """
    sqrt_dt = np.sqrt(dt)
    euler_term = current_x + drift * dt + volatility * sqrt_dt * dW

    # Milstein correction
    dW_squared = dW ** 2
    milstein_correction = (
        0.5 * volatility_derivative * volatility * (dW_squared - dt)
    )

    return euler_term + milstein_correction

