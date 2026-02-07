"""
Greeks computation: Delta, Gamma, Vega, Theta, Rho.
"""

import numpy as np
from typing import Dict, Any
from scipy.stats import norm
import inspect


class GreeksComputer:
    def __init__(self, bump_size: float = 0.01, rng_seed: int = 42):
        self.bump_size = bump_size
        self.rng_seed = rng_seed

    def _with_vol(self, model, new_vol):
        params = model.get_required_params()

        for key in ("volatility", "sigma", "vol", "initial_volatility"):
            if key in params:
                params[key] = new_vol
                return self._clone_model(model, **params)

        # Try model-specific prefixed keys
        for key in ("heston_sigma", "kou_sigma", "merton_sigma", "sabr_sigma"):
            if key in params:
                params[key] = new_vol
                return self._clone_model(model, **params)

        raise KeyError(
            "Model does not expose a recognizable volatility parameter "
            "(expected one of: volatility, sigma, vol, initial_volatility, or model-specific variants)"
        )

    def _with_time(self, model, new_T):
        params = model.get_required_params()
        for key in ("time_to_maturity", "T", "maturity", "tenor"):
            if key in params:
                params[key] = new_T
                return self._clone_model(model, **params)
        raise KeyError(
            "Model does not expose a recognizable maturity parameter "
            "(expected one of: time_to_maturity, T, maturity, tenor)"
        )

    def _with_rate(self, model, new_r):
        params = model.get_required_params()
        for key in ("risk_free_rate", "r", "rate", "interest_rate"):
            if key in params:
                params[key] = new_r
                return self._clone_model(model, **params)
        # Not all models store r; many just use it in discounting externally.
        return None

    def compute_all(
        self,
        spot: float,
        price: float,
        paths: np.ndarray,
        payoff_func,
        risk_free_rate: float,
        time_to_maturity: float,
        volatility: float,
        model=None,
        rng_engine=None,
        num_paths: int = None,
        num_steps: int = None,
    ) -> Dict[str, float]:
        """
        Compute all Greeks.

        Args:
            spot: Current spot price
            price: Option price estimate
            paths: Original MC paths
            payoff_func: Function to compute payoffs
            risk_free_rate: Risk-free rate
            time_to_maturity: Time to expiration
            volatility: Volatility
            model: Stochastic model instance (for regenerating paths)
            rng_engine: RNG engine (for regenerating paths)
            num_paths: Number of paths (if regenerating)
            num_steps: Number of steps per path (if regenerating)

        Returns:
            Dict of Greeks
        """
        delta = self._compute_delta_fd(spot, price, paths, payoff_func, risk_free_rate, time_to_maturity, model, rng_engine, num_paths, num_steps)
        gamma = self._compute_gamma_fd(spot, price, paths, payoff_func, risk_free_rate, time_to_maturity, model, rng_engine, num_paths, num_steps)
        vega = self._compute_vega_fd(spot, price, paths, payoff_func, risk_free_rate, time_to_maturity, volatility, model, rng_engine, num_paths, num_steps)
        theta = self._compute_theta_fd(spot, price, paths, payoff_func, risk_free_rate, time_to_maturity, model, rng_engine, num_paths, num_steps)
        rho = self._compute_rho_fd(spot, price, paths, payoff_func, risk_free_rate, time_to_maturity, model, rng_engine, num_paths, num_steps)

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }

    def _clone_model(self, model, **kwargs):
        """
        Clone a model with potentially different parameters.

        Safely filters parameters to match the model's __init__ signature,
        handling model-specific prefixes (e.g., 'heston_kappa' -> 'kappa').
        """
        model_class = type(model)

        # Get current model params from get_required_params()
        params = model.get_required_params()

        # Override with provided kwargs
        params.update(kwargs)

        # Get the __init__ signature to determine valid parameter names
        sig = inspect.signature(model_class.__init__)
        valid_params = {
            param_name for param_name in sig.parameters.keys()
            if param_name != 'self'
        }

        # Filter and map params: try exact match first, then strip prefixes
        filtered_params = {}
        model_name = model_class.__name__.lower()

        for key, value in params.items():
            # Exact match with constructor parameter
            if key in valid_params:
                filtered_params[key] = value
            # Try stripping model-specific prefix (e.g., 'heston_kappa' -> 'kappa')
            else:
                stripped_key = key
                for prefix in [f"{model_name}_", "model_"]:
                    if key.startswith(prefix):
                        stripped_key = key[len(prefix):]
                        break

                if stripped_key in valid_params:
                    filtered_params[stripped_key] = value
                # else: skip keys that don't match any constructor param

        # Create new instance with filtered params
        return model_class(**filtered_params)

    def _compute_delta_fd(self, spot, price, paths, payoff_func, r, T, model=None, rng_engine=None, num_paths=None, num_steps=None):
        """Delta via finite difference on spot."""
        bump = spot * self.bump_size

        if model and rng_engine and num_paths and num_steps:
            # Regenerate paths with bumped spot
            model_up = self._clone_model(model, spot=spot + bump)
            paths_up = model_up.generate_paths(rng_engine, num_paths, num_steps)

            model_down = self._clone_model(model, spot=spot - bump)
            paths_down = model_down.generate_paths(rng_engine, num_paths, num_steps)
        else:
            # Fall back to scaling paths
            paths_up = paths.copy()
            paths_up[:, :] *= (spot + bump) / spot

            paths_down = paths.copy()
            paths_down[:, :] *= (spot - bump) / spot

        payoffs_up = payoff_func(paths_up)
        payoffs_down = payoff_func(paths_down)

        price_up = np.mean(payoffs_up) * np.exp(-r * T)
        price_down = np.mean(payoffs_down) * np.exp(-r * T)

        # Central difference for better accuracy
        delta = (price_up - price_down) / (2 * bump)
        return delta

    def _compute_gamma_fd(
            self, spot, price, paths, payoff_func, r, T,
            model=None, rng_engine=None, num_paths=None, num_steps=None
    ):
        """
        Gamma via central finite difference.
        Uses identical Brownian shocks for up/down/base (CRN is mandatory).
        """
        bump = spot * self.bump_size

        if not (model and num_paths and num_steps):
            raise ValueError("Gamma requires model-based path regeneration")

        # --- ONE RNG, ONE set of shocks ---
        rng = np.random.default_rng(self.rng_seed)

        # Base
        model_0 = self._clone_model(model, spot=spot)
        paths_0 = model_0.generate_paths(rng, num_paths, num_steps)
        price_0 = np.mean(payoff_func(paths_0)) * np.exp(-r * T)

        # Reset RNG → identical shocks
        rng = np.random.default_rng(self.rng_seed)

        # Up
        model_up = self._clone_model(model, spot=spot + bump)
        paths_up = model_up.generate_paths(rng, num_paths, num_steps)
        price_up = np.mean(payoff_func(paths_up)) * np.exp(-r * T)

        # Reset RNG → identical shocks
        rng = np.random.default_rng(self.rng_seed)

        # Down
        model_down = self._clone_model(model, spot=spot - bump)
        paths_down = model_down.generate_paths(rng, num_paths, num_steps)
        price_down = np.mean(payoff_func(paths_down)) * np.exp(-r * T)

        gamma = (price_up - 2.0 * price_0 + price_down) / (bump ** 2)
        return gamma

    def _compute_vega_fd(
            self, spot, price, paths, payoff_func, r, T, vol,
            model=None, rng_engine=None, num_paths=None, num_steps=None
    ):
        bump = 1e-4

        if not (model and num_paths and num_steps):
            raise ValueError("Vega requires model-based path regeneration")

        # --- CRN via seed ---
        rng_up = np.random.default_rng(self.rng_seed)
        model_up = self._with_vol(model, vol + bump)
        paths_up = model_up.generate_paths(rng_up, num_paths, num_steps)

        rng_down = np.random.default_rng(self.rng_seed)
        model_down = self._with_vol(model, vol - bump)
        paths_down = model_down.generate_paths(rng_down, num_paths, num_steps)

        price_up = np.mean(payoff_func(paths_up)) * np.exp(-r * T)
        price_down = np.mean(payoff_func(paths_down)) * np.exp(-r * T)

        # Central difference: (price_up - price_down) / (2 * bump)
        # Fixed: was (200 * bump) causing 100x underestimation
        return (price_up - price_down) / (2 * bump)

    def _compute_theta_fd(
            self, spot, price, paths, payoff_func, r, T,
            model=None, rng_engine=None, num_paths=None, num_steps=None
    ):
        dt = 1 / 365  # 1 day

        if T <= dt:
            return 0.0

        if not (model and num_paths and num_steps):
            raise ValueError("Theta requires model-based path regeneration")

        # Price at T + dt
        rng_up = np.random.default_rng(self.rng_seed)
        model_up = self._with_time(model, T + dt)
        paths_up = model_up.generate_paths(rng_up, num_paths, num_steps)
        price_up = np.mean(payoff_func(paths_up)) * np.exp(-r * (T + dt))

        # Price at T - dt
        rng_down = np.random.default_rng(self.rng_seed)
        model_down = self._with_time(model, T - dt)
        paths_down = model_down.generate_paths(rng_down, num_paths, num_steps)
        price_down = np.mean(payoff_func(paths_down)) * np.exp(-r * (T - dt))

        # Theta = - dV/dT
        theta = -(price_up - price_down) / (2 * T)
        return theta

    def _compute_rho_fd(
            self, spot, price, paths, payoff_func, r, T,
            model=None, rng_engine=None, num_paths=None, num_steps=None
    ):
        """
        Rho = ∂V/∂r
        For equity calls under risk-neutral GBM, r affects BOTH:
          - drift of S_t (hence payoff distribution)
          - discounting
        So we must regenerate paths with bumped r.
        """
        bump = 1e-4

        if not (model and num_paths and num_steps):
            raise ValueError("Rho requires model-based path regeneration (r affects drift).")

        # r + bump
        rng_up = np.random.default_rng(self.rng_seed)
        model_up = self._with_rate(model, r + bump)
        paths_up = model_up.generate_paths(rng_up, num_paths, num_steps)
        price_up = np.mean(payoff_func(paths_up)) * np.exp(-(r + bump) * T)

        # r - bump
        rng_down = np.random.default_rng(self.rng_seed)
        model_down = self._with_rate(model, r - bump)
        paths_down = model_down.generate_paths(rng_down, num_paths, num_steps)
        price_down = np.mean(payoff_func(paths_down)) * np.exp(-(r - bump) * T)

        # Central difference: (price_up - price_down) / (2 * bump)
        # Fixed: was (200 * bump) causing 100x underestimation
        return (price_up - price_down) / (2 * bump)

