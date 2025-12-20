"""
Shifted Lognormal Model.

Dynamics: (S + shift) follows GBM
Handles negative spot prices in pricing contexts.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from .base import StochasticModel


class ShiftedLognormal(StochasticModel):
    """Shifted Lognormal model."""

    def __init__(
        self,
        spot: float,
        risk_free_rate: float,
        dividend_yield: float,
        initial_volatility: float,
        time_to_maturity: float,
        shift: float = 0.0,
    ):
        """
        Initialize Shifted Lognormal model.

        Args:
            shift: Shift parameter. S_shifted = S + shift must be > 0.
        """
        super().__init__(spot, risk_free_rate, dividend_yield, time_to_maturity)
        self.sigma = initial_volatility
        self.shift = shift

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
        Generate Shifted Lognormal paths.

        d(S + shift) / (S + shift) = (r - q)*dt + σ*dW

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

        # Shifted forward price
        shifted_spot = self.spot + self.shift

        # Generate paths in log-space of shifted price
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.spot

        log_shifted_paths = np.log(shifted_spot)

        for t in range(num_steps):
            log_shifted_paths += (self.drift - 0.5 * self.sigma ** 2) * dt + self.sigma * sqrt_dt * Z_transformed[:, t]
            # Convert back: S = shifted_S - shift
            paths[:, t + 1] = np.exp(log_shifted_paths) - self.shift

        return paths

    def get_required_params(self) -> Dict[str, Any]:
        """Return required parameters."""
        return {
            "spot": self.spot,
            "risk_free_rate": self.r,
            "dividend_yield": self.q,
            "initial_volatility": self.sigma,
            "time_to_maturity": self.T,
            "shift": self.shift,
        }

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate Shifted Lognormal parameters."""
        valid, msg = super().validate()
        if not valid:
            return valid, msg
        if self.sigma <= 0:
            return False, "Volatility must be positive"
        if self.spot + self.shift <= 0:
            return False, "Spot + Shift must be positive"
        return True, None

