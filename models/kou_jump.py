"""
Kou Double Exponential Jump Diffusion model.
"""

import numpy as np
from typing import Dict, Any
from models.base import StochasticModel


class KouJump(StochasticModel):
    """Kou Double Exponential Jump Diffusion model."""

    def __init__(
        self,
        spot: float,
        risk_free_rate: float,
        dividend_yield: float,
        initial_volatility: float,
        time_to_maturity: float,
        lambda_: float = 0.5,
        p_up: float = 0.5,
        eta_up: float = 20.0,
        eta_down: float = 10.0,
    ):
        """Initialize Kou Jump model."""
        super().__init__(spot, risk_free_rate, dividend_yield, time_to_maturity)
        self.sigma = initial_volatility
        self.lambda_ = lambda_  # Jump intensity
        self.p_up = p_up  # Probability of up jump
        self.eta_up = eta_up  # Decay rate of up jumps
        self.eta_down = eta_down  # Decay rate of down jumps

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
        Generate Kou Jump Diffusion paths.

        Jumps follow double exponential distribution.
        """
        dt = self.T / num_steps
        sqrt_dt = np.sqrt(dt)

        dW = rng_engine.standard_normal((num_paths, num_steps))

        # Jump counts (Poisson)
        dN = np.random.poisson(self.lambda_ * dt, (num_paths, num_steps))

        paths = np.zeros((num_paths, num_steps + 1))
        paths[:, 0] = self.spot
        log_paths = np.log(paths[:, 0])

        # Compensate drift
        p_up = self.p_up
        p_down = 1 - self.p_up
        mean_jump = p_up * (self.eta_up / (self.eta_up + 1)) + p_down * (-self.eta_down / (self.eta_down + 1))
        jump_compensation = self.lambda_ * mean_jump
        adjusted_drift = self.drift - jump_compensation

        for t in range(num_steps):
            # Diffusion
            log_paths += (adjusted_drift - 0.5 * self.sigma ** 2) * dt + self.sigma * sqrt_dt * dW[:, t]

            # Jumps
            for i in range(num_paths):
                num_jumps = dN[i, t]
                if num_jumps > 0:
                    for _ in range(num_jumps):
                        if np.random.rand() < p_up:
                            # Exponential up jump
                            J = np.random.exponential(1.0 / self.eta_up)
                        else:
                            # Exponential down jump
                            J = -np.random.exponential(1.0 / self.eta_down)
                        log_paths[i] += J

            paths[:, t + 1] = np.exp(log_paths)

        return paths

    def get_required_params(self) -> Dict[str, Any]:
        """Return required parameters."""
        return {
            "spot": self.spot,
            "risk_free_rate": self.r,
            "dividend_yield": self.q,
            "initial_volatility": self.sigma,
            "time_to_maturity": self.T,
            "kou_lambda": self.lambda_,
            "kou_p_up": self.p_up,
            "kou_eta_up": self.eta_up,
            "kou_eta_down": self.eta_down,
        }

    def validate(self) -> tuple:
        """Validate parameters."""
        valid, msg = super().validate()
        if not valid:
            return valid, msg
        if self.sigma <= 0:
            return False, "Volatility must be positive"
        if self.lambda_ < 0:
            return False, "Lambda must be non-negative"
        if not (0 <= self.p_up <= 1):
            return False, "p_up must be in [0, 1]"
        if self.eta_up <= 0:
            return False, "eta_up must be positive"
        if self.eta_down <= 0:
            return False, "eta_down must be positive"
        return True, None

