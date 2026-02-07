"""
Regime-Switching GBM Model.

Two regimes with different volatilities and Markov transition matrix.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from .base import StochasticModel


class RegimeSwitchingGBM(StochasticModel):
    """Simple 2-regime Markov-switching GBM model."""

    def __init__(
        self,
        spot: float,
        risk_free_rate: float,
        dividend_yield: float,
        initial_volatility: float,
        time_to_maturity: float,
        sigma_low: float = 0.10,
        sigma_high: float = 0.40,
        p_ll: float = 0.95,
        p_hh: float = 0.95,
    ):
        """
        Initialize Regime-Switching GBM.

        Args:
            spot, risk_free_rate, dividend_yield, initial_volatility, time_to_maturity: Market params
            sigma_low: Volatility in low-vol regime
            sigma_high: Volatility in high-vol regime
            p_ll: Probability of staying in low regime (given in low)
            p_hh: Probability of staying in high regime (given in high)
        """
        super().__init__(spot, risk_free_rate, dividend_yield, time_to_maturity)
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.p_ll = p_ll  # P(regime_t = low | regime_t-1 = low)
        self.p_hh = p_hh  # P(regime_t = high | regime_t-1 = high)

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
        Generate regime-switching GBM paths.

        dS/S = (r - q)*dt + σ(regime_t)*dW
        regime_t ~ Markov(transition matrix)

        Returns:
            Array of shape (num_paths, num_steps + 1)
        """
        dt = self.T / num_steps
        sqrt_dt = np.sqrt(dt)

        # Generate innovations for price and regime changes
        Z_price = rng_engine.standard_normal((num_paths, num_steps))
        U_regime = rng_engine.uniform((num_paths, num_steps))

        # Apply antithetic variates (to price innovations)
        if antithetic_variates:
            half = num_paths // 2
            Z_price = np.vstack([Z_price[:half], -Z_price[:half]])
            U_regime = np.vstack([U_regime[:half], U_regime[:half]])  # Same regime for antithetic pair
            num_paths = half * 2

        # Transform innovations
        from core.rng_distributions import InnovationTransform
        transform = InnovationTransform(distribution, student_t_df, use_sobol=False)
        Z_transformed = transform.transform(Z_price)

        # Initialize paths and regimes (0 = low, 1 = high)
        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.spot
        regimes = np.zeros((num_paths, num_steps + 1), dtype=int)  # Start in low regime

        log_paths = np.log(paths[:, 0])

        for t in range(num_steps):
            # Switch regimes based on Markov transition
            for i in range(num_paths):
                if regimes[i, t] == 0:  # Currently low
                    if U_regime[i, t] >= self.p_ll:
                        regimes[i, t + 1] = 1  # Switch to high
                    else:
                        regimes[i, t + 1] = 0  # Stay in low
                else:  # Currently high
                    if U_regime[i, t] >= self.p_hh:
                        regimes[i, t + 1] = 0  # Switch to low
                    else:
                        regimes[i, t + 1] = 1  # Stay in high

            # Get volatility for each path based on regime
            sigma_t = np.where(regimes[:, t + 1] == 0, self.sigma_low, self.sigma_high)

            # Update log prices
            log_paths += (self.drift - 0.5 * sigma_t ** 2) * dt + sigma_t * sqrt_dt * Z_transformed[:, t]
            paths[:, t + 1] = np.exp(log_paths)

        return paths

    def get_required_params(self) -> Dict[str, Any]:
        """Return required parameters."""
        return {
            "spot": self.spot,
            "risk_free_rate": self.r,
            "dividend_yield": self.q,
            "initial_volatility": (self.sigma_low + self.sigma_high) / 2,
            "time_to_maturity": self.T,
            "sigma_low": self.sigma_low,
            "sigma_high": self.sigma_high,
            "p_ll": self.p_ll,
            "p_hh": self.p_hh,
        }

    def validate(self) -> Tuple[bool, Optional[str]]:
        """Validate Regime-Switching GBM parameters."""
        valid, msg = super().validate()
        if not valid:
            return valid, msg
        if self.sigma_low <= 0 or self.sigma_high <= 0:
            return False, "Volatilities must be positive"
        if not (0 <= self.p_ll <= 1) or not (0 <= self.p_hh <= 1):
            return False, "Transition probabilities must be in [0, 1]"
        return True, None

