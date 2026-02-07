"""
Convergence and diagnostic analysis.
"""

import numpy as np
from typing import Dict, Optional


class DiagnosticsAnalyzer:
    """Analyze MC convergence and quality."""

    @staticmethod
    def convergence_analysis(convergence_history: np.ndarray) -> Dict[str, float]:
        """
        Analyze convergence from cumulative mean history.

        Args:
            convergence_history: Array of cumulative means

        Returns:
            Dict with convergence metrics
        """
        if convergence_history is None or len(convergence_history) == 0:
            return {}

        # Standard error decline rate
        n = len(convergence_history)
        se_initial = np.std(convergence_history[:n//10])
        se_final = np.std(convergence_history[-n//10:])
        convergence_rate = se_initial / se_final if se_final > 0 else 1.0

        return {
            "initial_std": se_initial,
            "final_std": se_final,
            "convergence_rate": convergence_rate,
            "final_estimate": convergence_history[-1],
        }

    @staticmethod
    def path_diagnostics(paths: np.ndarray) -> Dict[str, float]:
        """
        Analyze path quality.

        Args:
            paths: Simulated paths

        Returns:
            Diagnostic metrics
        """
        num_paths = paths.shape[0]
        num_steps = paths.shape[1]

        # Final spot statistics
        final_spots = paths[:, -1]

        return {
            "mean_final_spot": np.mean(final_spots),
            "std_final_spot": np.std(final_spots),
            "min_final_spot": np.min(final_spots),
            "max_final_spot": np.max(final_spots),
            "num_paths": num_paths,
            "num_steps": num_steps,
        }

    @staticmethod
    def autocorrelation_test(payoffs: np.ndarray, max_lag: int = 10) -> Dict[int, float]:
        """Compute autocorrelation of payoffs (should be ~0)."""
        mean = np.mean(payoffs)
        c0 = np.sum((payoffs - mean) ** 2) / len(payoffs)

        autocorr = {}
        for lag in range(1, min(max_lag + 1, len(payoffs) // 2)):
            c_lag = np.sum((payoffs[:-lag] - mean) * (payoffs[lag:] - mean)) / len(payoffs)
            autocorr[lag] = c_lag / c0 if c0 > 0 else 0.0

        return autocorr

