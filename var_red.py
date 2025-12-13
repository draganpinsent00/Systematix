"""Variance reduction helpers.

Provides control variate adjustment and small helpers to create antithetic draws.
"""
from typing import Dict, Any
import numpy as np


def apply_control_variate(payoffs: np.ndarray, control: np.ndarray, control_expectation: float) -> Dict[str, Any]:
    """Adjust payoffs using control variate X with known expectation EX.

    Returns dict with adjusted payoffs and metadata including control coefficient c.
    c = cov(Y, X) / var(X)
    Y_adj = Y - c (X - EX)
    """
    Y = np.asarray(payoffs).astype(float)
    X = np.asarray(control).astype(float)
    n = Y.shape[0]
    if n <= 1:
        return {'adjusted': Y, 'c': 0.0, 'mean': float(np.mean(Y))}
    cov = np.cov(Y, X, ddof=1)[0, 1]
    varX = float(np.var(X, ddof=1))
    c = cov / varX if varX > 0.0 else 0.0
    Y_adj = Y - c * (X - control_expectation)
    out = {
        'adjusted': Y_adj,
        'c': float(c),
        'mean_raw': float(np.mean(Y)),
        'mean_adj': float(np.mean(Y_adj)),
        'std_raw': float(np.std(Y, ddof=1)) if n > 1 else 0.0,
        'std_adj': float(np.std(Y_adj, ddof=1)) if n > 1 else 0.0,
    }
    return out


def make_antithetic(Z: np.ndarray) -> np.ndarray:
    """Given standard normal draws Z shape (n, steps), produce concatenated antithetic draws [Z; -Z]."""
    Z = np.asarray(Z)
    return np.vstack([Z, -Z])
