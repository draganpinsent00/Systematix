
"""
Base instrument and payoff interface.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class Instrument(ABC):
    """Abstract base class for options."""

    def __init__(self, strike: float = 100.0, **params):
        """Initialize instrument."""
        self.strike = strike
        self.params = params

    @abstractmethod
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Compute payoff at expiration.

        Args:
            paths: Simulated paths, shape (num_paths, num_steps + 1) or
                   (num_paths, num_assets, num_steps + 1) for multi-asset

        Returns:
            Array of payoffs, shape (num_paths,)
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return all parameters."""
        pass

