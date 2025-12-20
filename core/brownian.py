"""
Brownian motion generation and Brownian bridge methods.
"""

import numpy as np
from typing import Tuple, Optional


def generate_brownian_paths(
    rng_engine,
    num_paths: int,
    num_steps: int,
    time_to_maturity: float,
    drift: float = 0.0,
    volatility: float = 1.0,
    antithetic: bool = True,
) -> np.ndarray:
    """
    Generate standard Brownian paths with optional antithetic variates.

    Args:
        rng_engine: RNG engine instance
        num_paths: number of simulated paths
        num_steps: number of time steps
        time_to_maturity: total time horizon
        drift: drift parameter (0 for standard BM)
        volatility: volatility multiplier (1 for standard BM)
        antithetic: if True, half paths are antithetic

    Returns:
        Array of shape (num_paths, num_steps + 1) with paths starting at 0
    """
    dt = time_to_maturity / num_steps
    sqrt_dt = np.sqrt(dt)

    if antithetic:
        # Generate half, then mirror
        half_paths = num_paths // 2
        actual_paths = half_paths * 2
        dW = rng_engine.standard_normal((half_paths, num_steps))
        dW = np.vstack([dW, -dW])  # Antithetic mirror
        num_paths = actual_paths
    else:
        dW = rng_engine.standard_normal((num_paths, num_steps))

    # Integrate increments
    paths = np.zeros((num_paths, num_steps + 1))
    for t in range(1, num_steps + 1):
        paths[:, t] = paths[:, t - 1] + (drift * dt + volatility * sqrt_dt * dW[:, t - 1])

    return paths


def brownian_bridge(
    z_start: np.ndarray,
    z_end: np.ndarray,
    intermediate_times: Optional[np.ndarray] = None,
    rng_engine=None,
) -> np.ndarray:
    """
    Fill in intermediate points using Brownian bridge.

    Args:
        z_start: Brownian value at t=0 (shape: num_paths)
        z_end: Brownian value at t=T (shape: num_paths)
        intermediate_times: array of intermediate times in [0, T]
        rng_engine: RNG engine for bridge innovations

    Returns:
        Array of bridged values at intermediate times
    """
    if intermediate_times is None:
        return np.array([z_start, z_end])

    num_paths = z_start.shape[0]
    num_intermediate = len(intermediate_times)
    T = 1.0  # Normalized time

    # Generate bridge innovations
    if rng_engine is not None:
        bridge_noise = rng_engine.standard_normal((num_paths, num_intermediate))
    else:
        bridge_noise = np.random.standard_normal((num_paths, num_intermediate))

    result = np.zeros((num_paths, num_intermediate + 2))
    result[:, 0] = z_start
    result[:, -1] = z_end

    for i, t in enumerate(intermediate_times):
        # Brownian bridge: variance reduction, interpolate linearly in mean
        variance = (t / T) * ((T - t) / T)
        std_bridge = np.sqrt(variance)
        result[:, i + 1] = (
            z_start * (1 - t / T) +
            z_end * (t / T) +
            std_bridge * bridge_noise[:, i]
        )

    return result

