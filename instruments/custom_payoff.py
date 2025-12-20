"""
Custom payoff definition and validation.
"""

import numpy as np
from typing import Callable, Optional


class PayoffDefinition:
    """Encapsulates a custom payoff formula."""

    def __init__(self, formula_str: str, strike: float = 100.0):
        """
        Initialize with payoff formula string.

        Args:
            formula_str: String like 'max(S-K, 0)' or 'max(S-100, 0)'
            strike: Strike price for reference
        """
        self.formula_str = formula_str
        self.strike = strike

    def compile(self) -> Callable:
        """Compile formula into executable function."""
        safe_dict = {
            'np': np,
            'max': np.maximum,
            'min': np.minimum,
            'abs': np.abs,
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
            'K': self.strike,
        }

        def payoff_func(S):
            """S can be a scalar or array."""
            return eval(self.formula_str, {"__builtins__": {}}, {**safe_dict, 'S': S})

        return payoff_func

