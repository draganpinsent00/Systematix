
"""
Multi-asset models.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from models.base import StochasticModel


class MultiAssetGBM(StochasticModel):
    """Multi-asset GBM with correlation structure."""

    def __init__(
        self,
        spots: List[float],
        risk_free_rate: float,
        dividend_yields: List[float],
        volatilities: List[float],
        time_to_maturity: float,
        correlation_matrix: Optional[np.ndarray] = None,
    ):
        """Initialize multi-asset GBM."""
        self.spots = np.array(spots)
        self.num_assets = len(spots)
        self.r = risk_free_rate
        self.q = np.array(dividend_yields)
        self.sigmas = np.array(volatilities)
        self.T = time_to_maturity

        if correlation_matrix is None:
            self.corr = np.eye(self.num_assets)
        else:
            self.corr = correlation_matrix

        self.drift = self.r - self.q

    def generate_paths(
        self,
        rng_engine,
        num_paths: int,
        num_steps: int,
        distribution: str = "normal",
        student_t_df: float = 3.0,
        antithetic_variates: bool = True,
        use_sobol: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Generate correlated multi-asset paths.

        Returns:
            Array of shape (num_paths, num_assets, num_steps + 1)
        """
        dt = self.T / num_steps
        sqrt_dt = np.sqrt(dt)

        # Cholesky decomposition of correlation
        L = np.linalg.cholesky(self.corr)

        # Independent Brownian increments
        Z_indep = rng_engine.standard_normal((num_paths, self.num_assets, num_steps))

        # Correlate
        Z = np.zeros((num_paths, self.num_assets, num_steps))
        for t in range(num_steps):
            Z[:, :, t] = Z_indep[:, :, t] @ L.T

        paths = np.zeros((num_paths, self.num_assets, num_steps + 1))
        paths[:, :, 0] = self.spots

        log_paths = np.log(paths[:, :, 0])

        for t in range(num_steps):
            log_paths += (
                (self.drift - 0.5 * self.sigmas ** 2)[:, np.newaxis] * dt +
                (self.sigmas[:, np.newaxis] * sqrt_dt) * Z[:, :, t].T
            )
            paths[:, :, t + 1] = np.exp(log_paths)

        return paths

    def get_required_params(self) -> Dict[str, Any]:
        """Return required parameters."""
        return {
            "spots": self.spots,
            "risk_free_rate": self.r,
            "dividend_yields": self.q,
            "volatilities": self.sigmas,
            "time_to_maturity": self.T,
            "correlation_matrix": self.corr,
        }

    def validate(self) -> tuple:
        """Validate parameters."""
        if np.any(self.spots <= 0):
            return False, "All spots must be positive"
        if self.T <= 0:
            return False, "Time to maturity must be positive"
        if np.any(self.sigmas <= 0):
            return False, "All volatilities must be positive"
        return True, None

