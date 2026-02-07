"""
Risk metrics: VaR, CVaR, drawdown analysis.
"""

import numpy as np
from typing import Dict, Tuple


class RiskAnalyzer:
    """Compute risk metrics from payoff distributions."""

    @staticmethod
    def compute_var_cvar(
        payoffs: np.ndarray,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Compute Value-at-Risk and Conditional VaR.

        Args:
            payoffs: Array of P&L values
            confidence_level: Confidence level (e.g., 0.95 for 95%)

        Returns:
            (VaR, CVaR) as positive loss figures
        """
        sorted_payoffs = np.sort(payoffs)
        var_idx = int(len(payoffs) * (1 - confidence_level))
        var = -sorted_payoffs[var_idx]
        cvar = -np.mean(sorted_payoffs[:var_idx])

        return var, cvar

    @staticmethod
    def compute_statistics(payoffs: np.ndarray) -> Dict[str, float]:
        """Compute basic statistics."""
        return {
            "mean": np.mean(payoffs),
            "std": np.std(payoffs),
            "min": np.min(payoffs),
            "max": np.max(payoffs),
            "skewness": _skewness(payoffs),
            "kurtosis": _kurtosis(payoffs),
        }

    @staticmethod
    def compute_pnl_distribution(
        spot: float,
        paths: np.ndarray,
        payoffs: np.ndarray,
        initial_cost: float,
    ) -> Dict[str, np.ndarray]:
        """
        Compute P&L distribution.

        Args:
            spot: Initial spot price
            paths: Simulated paths
            payoffs: Option payoffs
            initial_cost: Option premium paid

        Returns:
            Dict with spot values and P&L
        """
        final_spots = paths[:, -1]
        pnl = payoffs - initial_cost

        return {
            "final_spots": final_spots,
            "pnl": pnl,
            "payoffs": payoffs,
        }

    @staticmethod
    def compute_max_drawdown(paths: np.ndarray) -> float:
        """Compute maximum drawdown from paths."""
        cum_max = np.maximum.accumulate(paths, axis=1)
        drawdown = (paths - cum_max) / cum_max
        max_dd = np.min(drawdown)
        return max_dd


def _skewness(x: np.ndarray) -> float:
    """Compute skewness."""
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return np.mean((x - mean) ** 3) / (std ** 3)


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis."""
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0.0
    return np.mean((x - mean) ** 4) / (std ** 4) - 3

