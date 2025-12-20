"""
Merton Jump Diffusion model.
"""

import numpy as np
from typing import Dict, Any
from models.base import StochasticModel


class MertonJump(StochasticModel):
    """Merton Jump Diffusion model."""

    def __init__(
        self,
        spot: float,
        risk_free_rate: float,
        dividend_yield: float,
        initial_volatility: float,
        time_to_maturity: float,
        lambda_: float = 0.5,
        mu_j: float = 0.0,
        sigma_j: float = 0.2,
    ):
        """Initialize Merton Jump model."""
        super().__init__(spot, risk_free_rate, dividend_yield, time_to_maturity)
        self.sigma = initial_volatility
        self.lambda_ = lambda_  # Jump intensity
        self.mu_j = mu_j  # Log-jump mean
        self.sigma_j = sigma_j  # Log-jump std

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
        Generate Merton Jump Diffusion paths.
        """
        dt = self.T / num_steps
        sqrt_dt = np.sqrt(dt)

        # Brownian increments - use rng_engine
        dW = rng_engine.standard_normal((num_paths, num_steps))

        # Jump counts (Poisson) - use rng_engine for CRN consistency
        dN = rng_engine.poisson(self.lambda_ * dt, (num_paths, num_steps))

        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.spot
        log_paths = np.log(paths[:, 0])

        # Compensate drift for jumps
        jump_compensation = self.lambda_ * (np.exp(self.mu_j + 0.5 * self.sigma_j ** 2) - 1)
        adjusted_drift = self.drift - jump_compensation

        for t in range(num_steps):
            # Diffusion
            log_paths += (adjusted_drift - 0.5 * self.sigma ** 2) * dt + self.sigma * sqrt_dt * dW[:, t]

            # Jumps: draw jump sizes for each path that has a jump
            for i in range(num_paths):
                num_jumps = dN[i, t]
                if num_jumps > 0:
                    # Draw num_jumps log-normal jump sizes for this path using rng_engine
                    jump_sizes = rng_engine.normal(self.mu_j, self.sigma_j, num_jumps)
                    log_paths[i] += np.sum(jump_sizes)

            paths[:, t + 1] = np.exp(log_paths)

        return paths

    def get_required_params(self) -> Dict[str, Any]:
        """Return required parameters."""
        return {
            "spot": self.spot,
            "risk_free_rate": self.r,
            "dividend_yield": self.q,
            "initial_volatility": self.sigma,
            "time_to_maturity": self.T,
            "merton_lambda": self.lambda_,
            "merton_mu_j": self.mu_j,
            "merton_sigma_j": self.sigma_j,
        }

    def validate(self) -> tuple:
        """Validate parameters."""
        valid, msg = super().validate()
        if not valid:
            return valid, msg
        if self.sigma <= 0:
            return False, "Volatility must be positive"
        if self.lambda_ < 0:
            return False, "Lambda (jump intensity) must be non-negative"
        if self.sigma_j < 0:
            return False, "Sigma_j (jump std) must be non-negative"
        return True, None

