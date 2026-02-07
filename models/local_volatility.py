"""
Local Volatility (Flat Surface) Model.

Wrapper around GBM with time-dependent volatility.
σ(t, S) = σ₀ (constant in this simplified version)
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from .base import StochasticModel


class LocalVolatility(StochasticModel):
    """Local Volatility model with flat deterministic surface."""

    def __init__(
        self,
        spot: float,
        risk_free_rate: float,
        dividend_yield: float,
        initial_volatility: float,
        time_to_maturity: float,
    ):
        """Initialize Local Volatility model."""
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
        Generate Local Volatility paths using flat surface.

        dS/S = (r - q)*dt + σ(t, S)*dW

        In flat case, σ(t, S) = σ₀ (reduces to GBM).

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

        # Generate paths (log-space, like GBM)
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.spot

        log_paths = np.log(paths[:, 0])

        for t in range(num_steps):
            log_paths += (self.drift - 0.5 * self.sigma ** 2) * dt + self.sigma * sqrt_dt * Z_transformed[:, t]
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
        }

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate Local Volatility parameters."""
        valid, msg = super().validate()
        if not valid:
            return valid, msg
        if self.sigma <= 0:
            return False, "Volatility must be positive"
        return True, None

