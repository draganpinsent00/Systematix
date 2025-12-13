"""GBM model wrapper class."""
from typing import Optional, Dict, Any
from simulator import simulate_gbm_paths


class GBMModel:
    name = 'GBM'

    def __init__(self, mu: float = 0.0, sigma: float = 0.2):
        self.mu = float(mu)
        self.sigma = float(sigma)

    def simulate(self, S0: float, r: float, T: float, steps: int, n_paths: int, seed: Optional[int] = None, antithetic: bool = False, dist: str = 'normal', dist_params: Optional[dict] = None, moment_match: bool = False):
        # perform a minimal validation to keep signature checks
        if steps <= 0:
            raise ValueError('steps must be > 0')
        if n_paths <= 0:
            raise ValueError('n_paths must be > 0')
        # note: simulate_gbm_paths expects r and sigma parameters; mu currently unused (drift uses r)
        return simulate_gbm_paths(S0, r, self.sigma, T, steps, n_paths, seed=seed, antithetic=antithetic, dist=dist, dist_params=dist_params, moment_match=moment_match)

    def to_dict(self) -> Dict[str, Any]:
        return {'name': self.name, 'mu': self.mu, 'sigma': self.sigma}
