"""
Bachelier Model - Arithmetic Brownian Motion.

S_t = S_0 + r*t + σ*W_t
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from .base import StochasticModel


class Bachelier(StochasticModel):
    """Bachelier (Arithmetic Brownian Motion) model."""

    def __init__(
        self,
        spot: float,
        risk_free_rate: float,
        dividend_yield: float,
        initial_volatility: float,
        time_to_maturity: float,
    ):
        """Initialize Bachelier model."""
        super().__init__(spot, risk_free_rate, dividend_yield, time_to_maturity)
        self.sigma = initial_volatility

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
        Generate Bachelier paths.

        dS = r*dt + σ*dW

        Returns:
            Array of shape (num_paths, num_steps + 1)
        """
        dt = self.T / num_steps
        sqrt_dt = np.sqrt(dt)

        # Generate innovations
        Z = rng_engine.standard_normal((num_paths, num_steps))

        # Apply antithetic variates
        if antithetic_variates:
            half = num_paths // 2
            Z = np.vstack([Z[:half], -Z[:half]])
            num_paths = half * 2

        # Transform to desired distribution
        from core.rng_distributions import InnovationTransform
        transform = InnovationTransform(distribution, student_t_df, use_sobol=False)
        Z_transformed = transform.transform(Z)

        # Generate paths
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.spot

        for t in range(num_steps):
            paths[:, t + 1] = paths[:, t] + self.drift * dt + self.sigma * sqrt_dt * Z_transformed[:, t]

        return paths

    def get_required_params(self) -> Dict[str, Any]:
        """Return required parameters."""
        return {
            "spot": self.spot,
            "risk_free_rate": self.r,
            "dividend_yield": self.q,
            "initial_volatility": self.sigma,
            "time_to_maturity": self.T,
        }

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate Bachelier parameters."""
        valid, msg = super().validate()
        if not valid:
            return valid, msg
        if self.sigma <= 0:
            return False, "Volatility must be positive"
        return True, None

