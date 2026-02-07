"""
Base model class and interface.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional


class StochasticModel(ABC):
    """Abstract base class for stochastic models."""

    def __init__(self, spot: float, risk_free_rate: float, dividend_yield: float, time_to_maturity: float):
        """Initialize base parameters."""
        self.spot = spot
        self.r = risk_free_rate
        self.q = dividend_yield
        self.T = time_to_maturity
        self.drift = self.r - self.q  # Standard drift for equity

    @abstractmethod
    def generate_paths(
        self,
        rng_engine,
        num_paths: int,
        num_steps: int,
        **kwargs
    ) -> np.ndarray:
        """
        Generate sample paths.

        Returns:
            Array of shape (num_paths, num_steps + 1)
        """
        pass

    @abstractmethod
    def get_required_params(self) -> Dict[str, Any]:
        """Return required parameters as dict."""
        pass

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate model parameters."""
        if self.spot <= 0:
            return False, "Spot price must be positive"
        if self.T <= 0:
            return False, "Time to maturity must be positive"
        return True, None

