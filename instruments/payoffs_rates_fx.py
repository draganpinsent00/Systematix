"""
Multi-asset and rates/FX payoffs.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from .base import Instrument


# MULTI-ASSET PAYOFFS
class BasketCall(Instrument):
    """Basket Call: max(weighted_sum - K, 0)."""

    def __init__(self, strike: float = 100.0, weights: Optional[np.ndarray] = None, **params):
        super().__init__(strike, **params)
        self.weights = weights if weights is not None else np.array([0.5, 0.5])
        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """paths shape: (num_paths, num_assets, num_steps + 1)"""
        if paths.ndim == 2:
            paths = paths[:, np.newaxis, :]  # Single asset case

        num_paths = paths.shape[0]
        num_assets = paths.shape[1]

        basket_value = np.zeros(num_paths)
        for i in range(num_assets):
            basket_value += self.weights[i] * paths[:, i, -1]

        return np.maximum(basket_value - self.strike, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "weights": self.weights.tolist()}


class BasketPut(Instrument):
    """Basket Put: max(K - weighted_sum, 0)."""

    def __init__(self, strike: float = 100.0, weights: Optional[np.ndarray] = None, **params):
        super().__init__(strike, **params)
        self.weights = weights if weights is not None else np.array([0.5, 0.5])
        self.weights = self.weights / np.sum(self.weights)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        if paths.ndim == 2:
            paths = paths[:, np.newaxis, :]

        num_paths = paths.shape[0]
        num_assets = paths.shape[1]

        basket_value = np.zeros(num_paths)
        for i in range(num_assets):
            basket_value += self.weights[i] * paths[:, i, -1]

        return np.maximum(self.strike - basket_value, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "weights": self.weights.tolist()}


class BestOfCall(Instrument):
    """Best-of Call: max(max(S_1, S_2, ...) - K, 0)."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        if paths.ndim == 2:
            return np.maximum(paths[:, -1] - self.strike, 0)

        # Multi-asset: take max across assets at expiration
        max_asset = np.max(paths[:, :, -1], axis=1)
        return np.maximum(max_asset - self.strike, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


class WorstOfPut(Instrument):
    """Worst-of Put: max(K - min(S_1, S_2, ...), 0)."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        if paths.ndim == 2:
            return np.maximum(self.strike - paths[:, -1], 0)

        # Multi-asset: take min across assets
        min_asset = np.min(paths[:, :, -1], axis=1)
        return np.maximum(self.strike - min_asset, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


class SpreadOption(Instrument):
    """Spread Option: max(S_1 - S_2 - K, 0)."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        if paths.ndim == 2:
            # Degenerate case: not applicable for single asset
            return np.zeros(paths.shape[0])

        # Assume two assets
        spread = paths[:, 0, -1] - paths[:, 1, -1]
        return np.maximum(spread - self.strike, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


class RainbowMaxCall(Instrument):
    """Rainbow Max Call."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        if paths.ndim == 2:
            return np.maximum(paths[:, -1] - self.strike, 0)

        max_asset = np.max(paths[:, :, -1], axis=1)
        return np.maximum(max_asset - self.strike, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


class RainbowMinPut(Instrument):
    """Rainbow Min Put."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        if paths.ndim == 2:
            return np.maximum(self.strike - paths[:, -1], 0)

        min_asset = np.min(paths[:, :, -1], axis=1)
        return np.maximum(self.strike - min_asset, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


# FORWARD-START OPTIONS
class ForwardStartCall(Instrument):
    """Forward-start call: strike set at future date."""

    def __init__(self, strike_ratio: float = 1.0, **params):
        super().__init__(100.0, **params)
        self.strike_ratio = strike_ratio

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        # Simplified: strike set at mid-point
        mid_idx = paths.shape[1] // 2
        strike_at_forward = self.strike_ratio * paths[:, mid_idx]
        S_T = paths[:, -1]
        return np.maximum(S_T - strike_at_forward, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike_ratio": self.strike_ratio}


class ForwardStartPut(Instrument):
    """Forward-start put."""

    def __init__(self, strike_ratio: float = 1.0, **params):
        super().__init__(100.0, **params)
        self.strike_ratio = strike_ratio

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        mid_idx = paths.shape[1] // 2
        strike_at_forward = self.strike_ratio * paths[:, mid_idx]
        S_T = paths[:, -1]
        return np.maximum(strike_at_forward - S_T, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike_ratio": self.strike_ratio}


# CLIQUET
class Cliquet(Instrument):
    """Cliquet: sum of periodic returns."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        num_paths = paths.shape[0]
        cliquet_payoff = np.zeros(num_paths)

        # Assume equal time spacing
        num_periods = paths.shape[1] - 1
        period_len = max(1, num_periods // 4)  # 4 periods

        for i in range(4):
            start_idx = i * period_len
            end_idx = min((i + 1) * period_len, paths.shape[1] - 1)
            if end_idx > start_idx:
                returns = (paths[:, end_idx] - paths[:, start_idx]) / paths[:, start_idx]
                cliquet_payoff += np.maximum(returns, 0)

        return paths[:, 0] * (1 + cliquet_payoff)

    def get_params(self) -> Dict[str, Any]:
        return {}


# VARIANCE SWAP
class VarianceSwap(Instrument):
    """Variance Swap: pays realized variance."""

    def __init__(self, strike_variance: float = 0.04, **params):
        super().__init__(100.0, **params)
        self.strike_variance = strike_variance

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        num_paths = paths.shape[0]
        num_steps = paths.shape[1]

        # Compute realized variance
        log_returns = np.diff(np.log(paths), axis=1)
        realized_variance = np.sum(log_returns ** 2, axis=1) * 252  # Annualize

        # Payoff: (realized_variance - strike_variance) * notional
        notional = 100  # Arbitrary
        payoff = (realized_variance - self.strike_variance) * notional

        return payoff

    def get_params(self) -> Dict[str, Any]:
        return {"strike_variance": self.strike_variance}

