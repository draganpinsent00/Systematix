"""monte_carlo.py
High-level simulation orchestration functions.
Delegates to simulator.py lower-level engines where possible to preserve behavior.
"""
from typing import Any
import numpy as np

from simulator import simulate_paths


def run_batched_simulation_adapter(model_key: str, model_params: dict, S0: float, r: float, sigma: float, T: float, steps: int,
                                   total_paths: int, batch_size: int, seed: int, antithetic: bool, sampler: str, dist_key: str,
                                   dist_params: dict, moment_match: bool) -> Any:
    """Adapter that maps higher-level run parameters to simulator.simulate_paths.

    Returns combined paths array (total_paths, steps+1)
    """
    parts = []
    rng_seed = None if int(seed) == 0 else int(seed)
    n_batches = (int(total_paths) + int(batch_size) - 1) // int(batch_size)
    for i in range(n_batches):
        this_batch = int(batch_size) if (i < n_batches - 1) else (int(total_paths) - int(batch_size) * (n_batches - 1))
        batch_seed = None if rng_seed is None else (rng_seed + i)
        # reuse existing simulate_paths routing
        paths = simulate_paths(model_key, S0, r, sigma, T, int(steps), int(this_batch), seed=batch_seed, antithetic=antithetic, dist=( 'student_t' if dist_key=='student-t' else ('sobol' if sampler=='sobol' else 'normal')), dist_params=dist_params, moment_match=moment_match)
        parts.append(paths)
    if parts:
        return np.vstack(parts)
    return None


# Compatibility alias: older code expects run_batched_simulation
run_batched_simulation = run_batched_simulation_adapter
