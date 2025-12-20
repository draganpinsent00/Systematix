"""
Random distribution transforms and inverse CDFs.
"""

import numpy as np
from scipy import special
from typing import Tuple


def standard_normal_to_student_t(z: np.ndarray, df: float) -> np.ndarray:
    """Transform standard normal to Student-t using variance normalization."""
    chi2_draw = np.random.chisquare(df, z.shape)
    return z * np.sqrt(df / chi2_draw)


def inverse_normal_cdf(u: np.ndarray) -> np.ndarray:
    """Inverse CDF (quantile) for standard normal."""
    return special.erfinv(2 * u - 1) * np.sqrt(2)


def inverse_student_t_cdf(u: np.ndarray, df: float) -> np.ndarray:
    """Inverse CDF for Student-t distribution."""
    return special.t.ppf(u, df)


def normalize_student_t_variance(x: np.ndarray, df: float) -> np.ndarray:
    """Normalize Student-t to unit variance."""
    # Var(t_df) = df / (df - 2)
    if df <= 2:
        return x
    var_t = df / (df - 2.0)
    return x / np.sqrt(var_t)


class InnovationTransform:
    """Transform uniform or normal samples to desired distribution."""

    def __init__(self, distribution: str = "normal", df: float = 3.0, use_sobol: bool = False):
        """
        Initialize innovation transform.

        Args:
            distribution: "normal" or "student_t"
            df: degrees of freedom (only for Student-t)
            use_sobol: if True, expect Sobol input; if False, expect normal input
        """
        self.distribution = distribution.lower()
        self.df = df
        self.use_sobol = use_sobol

        if self.distribution not in ["normal", "student_t"]:
            raise ValueError(f"Unknown distribution: {distribution}")

    def transform(self, z: np.ndarray) -> np.ndarray:
        """
        Transform input to desired distribution.

        For Sobol mode (use_sobol=True):
            Input z should be uniform(0,1), will apply inverse CDF.
        For normal mode (use_sobol=False):
            Input z should be standard normal, will transform if needed.
        """
        if self.use_sobol:
            # Sobol sequence: uniform input
            if self.distribution == "normal":
                return inverse_normal_cdf(z)
            else:  # student_t
                return inverse_student_t_cdf(z, self.df)
        else:
            # Normal input
            if self.distribution == "normal":
                return z
            else:  # student_t
                # Transform normal to Student-t and normalize variance
                chi2_draw = np.random.chisquare(self.df, z.shape)
                t_raw = z * np.sqrt(self.df / chi2_draw)
                return normalize_student_t_variance(t_raw, self.df)

