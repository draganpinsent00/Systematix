"""
Heston stochastic volatility model.
"""

import numpy as np
from typing import Dict, Any
from models.base import StochasticModel
from core.rng_distributions import InnovationTransform


class Heston(StochasticModel):
    """Heston stochastic volatility model."""

    def __init__(
        self,
        spot: float,
        risk_free_rate: float,
        dividend_yield: float,
        initial_volatility: float,
        time_to_maturity: float,
        kappa: float = 2.0,
        theta: float = 0.04,
        sigma: float = 0.3,
        rho: float = -0.5,
    ):
        """Initialize Heston model."""
        super().__init__(spot, risk_free_rate, dividend_yield, time_to_maturity)
        self.sigma_0 = initial_volatility
        self.kappa = kappa  # mean reversion speed
        self.theta = theta  # long-run variance
        self.sigma = sigma  # vol of vol
        self.rho = rho  # correlation between spot and vol

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
        Generate Heston paths using Euler discretization.

        Returns:
            Array of shape (num_paths, num_steps + 1)
        """
        dt = self.T / num_steps
        sqrt_dt = np.sqrt(dt)

        # Generate correlated random numbers
        dW_spot = rng_engine.standard_normal((num_paths, num_steps))
        dW_vol = rng_engine.standard_normal((num_paths, num_steps))

        # Apply correlation
        dW_vol_corr = self.rho * dW_spot + np.sqrt(1 - self.rho ** 2) * dW_vol

        # Initialize paths
        spot_paths = np.zeros((num_paths, num_steps + 1))
        vol_paths = np.zeros((num_paths, num_steps + 1))

        spot_paths[:, 0] = self.spot
        vol_paths[:, 0] = self.sigma_0 ** 2

        # Euler scheme
        for t in range(num_steps):
            # Variance process (Feller condition helps)
            vol_t = vol_paths[:, t]
            vol_t = np.maximum(vol_t, 1e-10)  # Enforce non-negativity

            # Spot dynamics: dS = r*S*dt + sqrt(v)*S*dW
            drift_spot = self.drift * spot_paths[:, t]
            diffusion_spot = np.sqrt(vol_t) * spot_paths[:, t] * sqrt_dt * dW_spot[:, t]
            spot_paths[:, t + 1] = spot_paths[:, t] + drift_spot * dt + diffusion_spot

            # Vol dynamics (CIR-like)
            drift_vol = self.kappa * (self.theta - vol_t) * dt
            diffusion_vol = self.sigma * np.sqrt(vol_t) * sqrt_dt * dW_vol_corr[:, t]
            vol_paths[:, t + 1] = vol_t + drift_vol + diffusion_vol
            vol_paths[:, t + 1] = np.maximum(vol_paths[:, t + 1], 1e-10)

        return spot_paths

    def get_required_params(self) -> Dict[str, Any]:
        """Return required parameters."""
        return {
            "spot": self.spot,
            "risk_free_rate": self.r,
            "dividend_yield": self.q,
            "initial_volatility": self.sigma_0,
            "time_to_maturity": self.T,
            "heston_kappa": self.kappa,
            "heston_theta": self.theta,
            "heston_sigma": self.sigma,
            "heston_rho": self.rho,
        }

    def validate(self) -> tuple:
        """Validate Heston parameters."""
        valid, msg = super().validate()
        if not valid:
            return valid, msg
        if self.sigma_0 <= 0:
            return False, "Initial volatility must be positive"
        if self.kappa < 0:
            return False, "Kappa must be non-negative"
        if self.theta <= 0:
            return False, "Theta must be positive"
        if self.sigma < 0:
            return False, "Sigma (vol of vol) must be non-negative"
        if abs(self.rho) > 0.99:
            return False, "Rho must be in (-1, 1)"
        return True, None

