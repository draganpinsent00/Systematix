"""
Variance reduction techniques.
"""

import numpy as np
from typing import Tuple, Callable


class AntitheticVariates:
    """Antithetic variates variance reduction."""

    @staticmethod
    def apply(paths: np.ndarray) -> np.ndarray:
        """
        Assumes paths already include antithetic pair.
        Just return as-is.
        """
        return paths


class ControlVariates:
    """Control variates variance reduction using analytical benchmark."""

    def __init__(self, benchmark_price: float):
        """
        Initialize with known analytical price.

        Args:
            benchmark_price: Analytical price of control security
        """
        self.benchmark_price = benchmark_price
        self.control_coeff = None

    def calibrate(self, mc_prices: np.ndarray, control_prices: np.ndarray) -> None:
        """
        Calibrate optimal coefficient.

        Args:
            mc_prices: MC estimate of option price
            control_prices: Control variate values across paths
        """
        if len(control_prices) < 2:
            self.control_coeff = 0
            return
        cov = np.cov(mc_prices, control_prices)[0, 1]
        var_control = np.var(control_prices, ddof=1)
        if var_control > 1e-10:
            self.control_coeff = cov / var_control
        else:
            self.control_coeff = 0

    def apply(self, option_payoffs: np.ndarray, control_payoffs: np.ndarray) -> np.ndarray:
        """
        Apply control variate adjustment.

        Returns:
            Adjusted payoffs
        """
        if self.control_coeff is None or self.control_coeff == 0:
            return option_payoffs

        control_correction = self.control_coeff * (
            control_payoffs - np.mean(control_payoffs)
        )
        return option_payoffs - control_correction


class ImportanceSampling:
    """Importance sampling for rare events / tail risk."""

    def __init__(self, shift_parameter: float = 1.0):
        """
        Initialize importance sampling.

        Args:
            shift_parameter: Drift shift for importance sampling
        """
        self.shift_parameter = shift_parameter
        self.likelihood_ratio = None

    def compute_likelihood_ratio(
        self,
        original_drift: float,
        shifted_drift: float,
        paths: np.ndarray,
    ) -> np.ndarray:
        """
        Compute Radon-Nikodym likelihood ratio.

        For Gaussian increments, this is:
        L = exp(-(lambda^2 T)/2 + lambda * W_T)
        """
        T = 1.0  # Normalized
        W_T = paths[:, -1]  # Final value
        lambda_shift = shifted_drift - original_drift

        ratio = np.exp(
            -0.5 * (lambda_shift ** 2) * T +
            lambda_shift * W_T
        )
        return ratio

    def apply(self, payoffs: np.ndarray, likelihood_ratio: np.ndarray) -> np.ndarray:
        """
        Apply importance sampling correction.

        Args:
            payoffs: Original payoffs
            likelihood_ratio: Likelihood ratio values

        Returns:
            Corrected payoffs
        """
        return payoffs * likelihood_ratio

