"""
SABR stochastic volatility model (scaffolding).
"""

import numpy as np
from typing import Dict, Any
from models.base import StochasticModel


class SABR(StochasticModel):
    """SABR stochastic volatility model."""

    def __init__(
        self,
        spot: float,
        risk_free_rate: float,
        dividend_yield: float,
        initial_volatility: float,
        time_to_maturity: float,
        alpha: float = 0.4,
        beta: float = 0.5,
        nu: float = 0.5,
        rho: float = -0.5,
    ):
        """Initialize SABR model."""
        super().__init__(spot, risk_free_rate, dividend_yield, time_to_maturity)
        self.sigma_0 = initial_volatility
        self.alpha = alpha  # ATM volatility
        self.beta = beta  # Elasticity (0-1)
        self.nu = nu  # Vol of vol
        self.rho = rho  # Spot-vol correlation

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
        Generate SABR paths using Euler discretization.

        dF = sigma * F^beta * dW_F
        dsigma = nu * sigma * dW_sigma
        """
        dt = self.T / num_steps
        sqrt_dt = np.sqrt(dt)

        dW_f = rng_engine.standard_normal((num_paths, num_steps))
        dW_sigma = rng_engine.standard_normal((num_paths, num_steps))

        # Correlation
        dW_sigma_corr = self.rho * dW_f + np.sqrt(1 - self.rho ** 2) * dW_sigma

        paths = np.zeros((num_paths, num_steps + 1))
        vol_paths = np.zeros((num_paths, num_steps + 1))

        paths[:, 0] = self.spot
        vol_paths[:, 0] = self.sigma_0

        for t in range(num_steps):
            sigma_t = vol_paths[:, t]
            sigma_t = np.maximum(sigma_t, 1e-10)
            f_t = paths[:, t]
            f_t = np.maximum(f_t, 1e-10)

            # SABR dynamics
            df = sigma_t * (f_t ** self.beta) * sqrt_dt * dW_f[:, t]
            dsigma = self.nu * sigma_t * sqrt_dt * dW_sigma_corr[:, t]

            paths[:, t + 1] = f_t + df
            vol_paths[:, t + 1] = sigma_t + dsigma
            vol_paths[:, t + 1] = np.maximum(vol_paths[:, t + 1], 1e-10)

        return paths

    def get_required_params(self) -> Dict[str, Any]:
        """Return required parameters."""
        return {
            "spot": self.spot,
            "risk_free_rate": self.r,
            "dividend_yield": self.q,
            "initial_volatility": self.sigma_0,
            "time_to_maturity": self.T,
            "sabr_alpha": self.alpha,
            "sabr_beta": self.beta,
            "sabr_nu": self.nu,
            "sabr_rho": self.rho,
        }

    def validate(self) -> tuple:
        """Validate SABR parameters."""
        valid, msg = super().validate()
        if not valid:
            return valid, msg
        if self.sigma_0 <= 0:
            return False, "Initial volatility must be positive"
        if self.alpha <= 0:
            return False, "Alpha must be positive"
        if not (0 <= self.beta <= 1):
            return False, "Beta must be in [0, 1]"
        if self.nu <= 0:
            return False, "Nu (vol of vol) must be positive"
        if abs(self.rho) > 0.99:
            return False, "Rho must be in (-1, 1)"
        return True, None

