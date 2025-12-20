"""
Sobol quasi-random sequence wrapper.
"""

import numpy as np
from typing import Tuple


def generate_sobol_normals(rng_engine, num_paths: int, num_steps: int) -> np.ndarray:
    """
    Generate quasi-random normals using Sobol sequence.

    For now, uses pseudo-random as placeholder. In production,
    would use scipy.stats.qmc.Sobol or ghalton.
    """
    from scipy.stats.qmc import Sobol
    from scipy.stats import norm

    sampler = Sobol(d=num_steps, seed=42)
    u = sampler.random(num_paths)  # uniform [0,1)

    # Transform to normal via inverse CDF
    Z = norm.ppf(u)

    return Z
