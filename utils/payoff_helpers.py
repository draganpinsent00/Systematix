"""Safe, curated payoff helpers exposed to user custom payoff functions.

Available helpers:
- discount_factor(r, T)
- terminal_call(ST, K)
- terminal_put(ST, K)
- asian_arithmetic(paths, K)
- running_average(paths)
- log_returns(paths)
- rebated_barrier(paths, barrier, rebate, direction='up')
- path_stats(paths)

These helpers are intentionally small, vectorized, and safe to expose to user code.
"""
import numpy as np
from typing import Dict


def discount_factor(r: float, T: float) -> float:
    """Return continuous discount factor e^{-r T}."""
    return float(np.exp(-float(r) * float(T)))


def terminal_call(ST: np.ndarray, K: float) -> np.ndarray:
    """Vectorized European call payoff on terminal prices."""
    return np.maximum(ST - K, 0.0)


def terminal_put(ST: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(K - ST, 0.0)


def asian_arithmetic(paths: np.ndarray, K: float) -> np.ndarray:
    """Arithmetic-average Asian call payoff (ignores initial spot at index 0)."""
    avg = np.mean(paths[:, 1:], axis=1)
    return np.maximum(avg - K, 0.0)


def running_average(paths: np.ndarray) -> np.ndarray:
    """Return arithmetic running average per-path (last column contains final average)."""
    return np.cumsum(paths[:, 1:], axis=1) / np.arange(1, paths.shape[1])


def log_returns(paths: np.ndarray) -> np.ndarray:
    """Return per-step log returns shape (n_paths, steps)."""
    return np.log(paths[:, 1:] / paths[:, :-1])


def rebated_barrier(paths: np.ndarray, barrier: float, rebate: float, direction: str = 'up') -> np.ndarray:
    """Return rebate payments for single barrier knock-outs/ins.

    direction: 'up' means barrier touched when path >= barrier, 'down' means path <= barrier.
    Returns an array of rebate values per path (0 or rebate).
    """
    if direction == 'up':
        breached = np.any(paths >= barrier, axis=1)
    else:
        breached = np.any(paths <= barrier, axis=1)
    return np.where(breached, rebate, 0.0)


def path_max(paths: np.ndarray) -> np.ndarray:
    return np.max(paths, axis=1)


def path_min(paths: np.ndarray) -> np.ndarray:
    return np.min(paths, axis=1)


def path_stats(paths: np.ndarray) -> Dict[str, float]:
    ST = paths[:, -1]
    return {
        'mean': float(np.mean(ST)),
        'std': float(np.std(ST, ddof=1)),
        'min': float(np.min(ST)),
        'max': float(np.max(ST))
    }
