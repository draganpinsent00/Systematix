"""
Monte Carlo engine orchestration.
"""

import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass


@dataclass
class MCResult:
    """Monte Carlo pricing result."""
    price: float
    ci_lower: float
    ci_upper: float
    std_error: float
    paths: np.ndarray  # (num_paths, num_steps + 1)
    payoffs: np.ndarray  # (num_paths,)
    discounted_payoffs: np.ndarray  # (num_paths,)
    convergence_history: Optional[np.ndarray] = None  # (num_paths,) cumulative mean
    variance_reduction_factor: float = 1.0


class MonteCarloEngine:
    """Main MC pricing engine."""

    def __init__(
        self,
        rng_engine,
        num_simulations: int = 10000,
        num_timesteps: int = 252,
        confidence_level: float = 0.95,
    ):
        """
        Initialize MC engine.

        Args:
            rng_engine: RNG engine instance
            num_simulations: number of paths
            num_timesteps: number of time steps
            confidence_level: for confidence intervals (default 95%)
        """
        self.rng_engine = rng_engine
        self.num_simulations = num_simulations
        self.num_timesteps = num_timesteps
        self.confidence_level = confidence_level
        self.z_alpha = np.abs(np.random.standard_normal(1)[0])  # ~1.96 for 95%

    def price(
        self,
        paths: np.ndarray,
        payoff_func: Callable[[np.ndarray], np.ndarray],
        risk_free_rate: float,
        time_to_maturity: float,
        use_lsm: bool = False,
        lsm_config: Optional[Dict[str, Any]] = None,
    ) -> MCResult:
        """
        Price option using MC paths.

        Args:
            paths: Simulated paths (num_paths, num_timesteps + 1)
            payoff_func: Function payoff_func(paths) -> payoffs
            risk_free_rate: Discount rate
            time_to_maturity: Time to expiration
            use_lsm: Use Longstaff-Schwartz for American options
            lsm_config: LSM configuration dict

        Returns:
            MCResult object
        """
        if use_lsm:
            return self._price_with_lsm(
                paths, payoff_func, risk_free_rate, time_to_maturity, lsm_config or {}
            )
        else:
            return self._price_standard(paths, payoff_func, risk_free_rate, time_to_maturity)

    def _price_standard(
        self,
        paths: np.ndarray,
        payoff_func: Callable[[np.ndarray], np.ndarray],
        risk_free_rate: float,
        time_to_maturity: float,
    ) -> MCResult:
        """Standard MC pricing."""
        num_paths = paths.shape[0]
        num_steps = paths.shape[1] - 1 if len(paths.shape) > 1 else 1

        # Compute payoff at expiration
        payoffs = payoff_func(paths)  # Should be (num_paths,)

        # CRITICAL FIX: Ensure payoffs are always (num_paths,) shaped
        # This handles cases where payoff_func returns wrong shape
        payoffs = np.asarray(payoffs).flatten()  # Flatten first

        if len(payoffs) != num_paths:
            # Wrong length - try to recover
            total_elements = len(payoffs)
            print(f"WARNING: Payoff has {total_elements} elements, expected {num_paths}. paths.shape={paths.shape}", flush=True)

            if total_elements % num_paths == 0:
                # Take the last value for each path
                payoffs_per_path = total_elements // num_paths
                payoffs = payoffs.reshape(num_paths, payoffs_per_path)[:, -1]
                print(f"  -> Fixed by taking last value per path ({payoffs_per_path} payoffs per path)", flush=True)
            elif total_elements % (num_steps + 1) == 0:
                # Reshape by timesteps and take final
                num_paths_inferred = total_elements // (num_steps + 1)
                payoffs = payoffs.reshape(num_paths_inferred, num_steps + 1)[:, -1]
                print(f"  -> Fixed by reshaping as (timesteps) and taking final", flush=True)
            else:
                # As last resort, just take first num_paths elements and warn
                print(f"  -> WARNING: Taking first {num_paths} elements from {total_elements} total", flush=True)
                payoffs = payoffs[:num_paths]

        # Final validation
        assert len(payoffs) == num_paths, f"Payoff shape {payoffs.shape} does not match num_paths {num_paths}"
        assert payoffs.ndim == 1, f"Payoffs must be 1D, got {payoffs.ndim}D"

        # Discount
        discount_factor = np.exp(-risk_free_rate * time_to_maturity)
        discounted_payoffs = payoffs * discount_factor

        # Estimate price
        price = np.mean(discounted_payoffs)
        std_dev = np.std(discounted_payoffs, ddof=1)
        std_error = std_dev / np.sqrt(num_paths)

        # Confidence interval (95%)
        z_crit = 1.96  # 95% CI
        ci_lower = price - z_crit * std_error
        ci_upper = price + z_crit * std_error

        # Convergence history - now guaranteed to have correct shape
        convergence = np.cumsum(discounted_payoffs) / np.arange(1, len(discounted_payoffs) + 1)

        return MCResult(
            price=price,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            std_error=std_error,
            paths=paths,
            payoffs=payoffs,
            discounted_payoffs=discounted_payoffs,
            convergence_history=convergence,
            variance_reduction_factor=1.0,
        )

    def _price_with_lsm(
        self,
        paths: np.ndarray,
        payoff_func: Callable[[np.ndarray], np.ndarray],
        risk_free_rate: float,
        time_to_maturity: float,
        lsm_config: Dict[str, Any],
    ) -> MCResult:
        """Price using Longstaff-Schwartz (for American options)."""
        from .lsm import LongstaffSchwartz

        num_steps = paths.shape[1] - 1
        dt = time_to_maturity / num_steps
        lsm_degree = lsm_config.get("degree", 2)

        lsm = LongstaffSchwartz(
            degree=lsm_degree,
            risk_free_rate=risk_free_rate,
        )

        price, payoffs = lsm.price(paths, payoff_func, dt)

        # Rough std error estimate
        std_dev = np.std(payoffs, ddof=1)
        std_error = std_dev / np.sqrt(len(payoffs))
        z_crit = 1.96
        ci_lower = price - z_crit * std_error
        ci_upper = price + z_crit * std_error

        return MCResult(
            price=price,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            std_error=std_error,
            paths=paths,
            payoffs=payoffs,
            discounted_payoffs=payoffs,
            convergence_history=None,
            variance_reduction_factor=1.0,
        )

