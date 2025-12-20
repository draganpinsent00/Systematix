
"""
Vanilla European and simple exotic payoffs.
"""

import numpy as np
from typing import Dict, Any, Optional
from .base import Instrument


class EuropeanCall(Instrument):
    """European Call option."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """max(S_T - K, 0)"""
        S_T = paths[:, -1]
        return np.maximum(S_T - self.strike, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


class EuropeanPut(Instrument):
    """European Put option."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """max(K - S_T, 0)"""
        S_T = paths[:, -1]
        return np.maximum(self.strike - S_T, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


class DigitalCashCall(Instrument):
    """Cash-or-Nothing Digital Call: pays 1 if S_T > K, 0 otherwise."""

    def __init__(self, strike: float = 100.0, payout: float = 1.0, **params):
        super().__init__(strike, **params)
        self.payout = payout

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        S_T = paths[:, -1]
        return np.where(S_T > self.strike, self.payout, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "payout": self.payout}


class DigitalCashPut(Instrument):
    """Cash-or-Nothing Digital Put: pays 1 if S_T < K, 0 otherwise."""

    def __init__(self, strike: float = 100.0, payout: float = 1.0, **params):
        super().__init__(strike, **params)
        self.payout = payout

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        S_T = paths[:, -1]
        return np.where(S_T < self.strike, self.payout, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "payout": self.payout}


class DigitalAssetCall(Instrument):
    """Asset-or-Nothing Digital Call: pays S_T if S_T > K, 0 otherwise."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        S_T = paths[:, -1]
        return np.where(S_T > self.strike, S_T, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


class DigitalAssetPut(Instrument):
    """Asset-or-Nothing Digital Put: pays S_T if S_T < K, 0 otherwise."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        S_T = paths[:, -1]
        return np.where(S_T < self.strike, S_T, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


class GapCall(Instrument):
    """Gap Call: if S_T > K1, pays S_T - K2 (K2 != K1)."""

    def __init__(self, strike: float = 100.0, trigger: Optional[float] = None, **params):
        super().__init__(strike, **params)
        self.trigger = trigger if trigger is not None else strike

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        S_T = paths[:, -1]
        return np.where(S_T > self.trigger, S_T - self.strike, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "trigger": self.trigger}


class GapPut(Instrument):
    """Gap Put: if S_T < K1, pays K2 - S_T (K2 != K1)."""

    def __init__(self, strike: float = 100.0, trigger: Optional[float] = None, **params):
        super().__init__(strike, **params)
        self.trigger = trigger if trigger is not None else strike

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        S_T = paths[:, -1]
        return np.where(S_T < self.trigger, self.strike - S_T, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "trigger": self.trigger}


class AmericanCall(Instrument):
    """
    American Call option priced with Least Squares Monte Carlo.

    Early exercise is allowed at any time before maturity.
    The payoff method computes intrinsic value max(S - K, 0).
    """

    def payoff(self, spot_prices: np.ndarray) -> np.ndarray:
        """
        Compute intrinsic payoff for American call.

        Args:
            spot_prices: 1D array of spot prices (at any time t)

        Returns:
            Intrinsic value max(S - K, 0) for each spot price
        """
        return np.maximum(spot_prices - self.strike, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


class AmericanPut(Instrument):
    """
    American Put option priced with Least Squares Monte Carlo.

    Early exercise is allowed at any time before maturity.
    The payoff method computes intrinsic value max(K - S, 0).
    """

    def payoff(self, spot_prices: np.ndarray) -> np.ndarray:
        """
        Compute intrinsic payoff for American put.

        Args:
            spot_prices: 1D array of spot prices (at any time t)

        Returns:
            Intrinsic value max(K - S, 0) for each spot price
        """
        return np.maximum(self.strike - spot_prices, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


class BermudanCall(Instrument):
    """Bermudan Call: early exercise at discrete dates."""

    def __init__(self, strike: float = 100.0, exercise_dates: Optional[list] = None, **params):
        super().__init__(strike, **params)
        self.exercise_dates = exercise_dates or []

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Intrinsic."""
        return np.maximum(paths[:, -1] - self.strike, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "exercise_dates": self.exercise_dates}


class BermudanPut(Instrument):
    """Bermudan Put: early exercise at discrete dates."""

    def __init__(self, strike: float = 100.0, exercise_dates: Optional[list] = None, **params):
        super().__init__(strike, **params)
        self.exercise_dates = exercise_dates or []

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Intrinsic."""
        return np.maximum(self.strike - paths[:, -1], 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "exercise_dates": self.exercise_dates}

