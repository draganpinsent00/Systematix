"""
Exotic payoffs: Asian, Barrier, Lookback, etc.
"""

import numpy as np
from typing import Dict, Any, Optional
from .base import Instrument


# ASIAN OPTIONS
class AsianArithmeticCall(Instrument):
    """Asian Call with arithmetic average."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """max(A_arith - K, 0) where A_arith = avg(S_t) over all times"""
        avg_price = np.mean(paths, axis=1)
        return np.maximum(avg_price - self.strike, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


class AsianArithmeticPut(Instrument):
    """Asian Put with arithmetic average."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        avg_price = np.mean(paths, axis=1)
        return np.maximum(self.strike - avg_price, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


class AsianGeometricCall(Instrument):
    """Asian Call with geometric average."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        # Geometric mean: exp(mean(log(S_t)))
        log_paths = np.log(np.maximum(paths, 1e-10))
        avg_log = np.mean(log_paths, axis=1)
        geo_avg = np.exp(avg_log)
        return np.maximum(geo_avg - self.strike, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


class AsianGeometricPut(Instrument):
    """Asian Put with geometric average."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        log_paths = np.log(np.maximum(paths, 1e-10))
        avg_log = np.mean(log_paths, axis=1)
        geo_avg = np.exp(avg_log)
        return np.maximum(self.strike - geo_avg, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


# BARRIER OPTIONS
class BarrierUpOutCall(Instrument):
    """Up-and-Out Call: knocked out if S ever hits barrier."""

    def __init__(self, strike: float = 100.0, barrier: float = 120.0, **params):
        super().__init__(strike, **params)
        self.barrier = barrier

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        # Knocked out if max(S) >= barrier
        max_price = np.max(paths, axis=1)
        intrinsic = np.maximum(paths[:, -1] - self.strike, 0)
        return np.where(max_price < self.barrier, intrinsic, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "barrier": self.barrier, "barrier_type": "knock_out", "barrier_direction": "up"}


class BarrierUpOutPut(Instrument):
    """Up-and-Out Put: knocked out if S ever hits barrier."""

    def __init__(self, strike: float = 100.0, barrier: float = 120.0, **params):
        super().__init__(strike, **params)
        self.barrier = barrier

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        max_price = np.max(paths, axis=1)
        intrinsic = np.maximum(self.strike - paths[:, -1], 0)
        return np.where(max_price < self.barrier, intrinsic, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "barrier": self.barrier, "barrier_type": "knock_out", "barrier_direction": "up"}


class BarrierDownOutCall(Instrument):
    """Down-and-Out Call: knocked out if S ever hits barrier."""

    def __init__(self, strike: float = 100.0, barrier: float = 80.0, **params):
        super().__init__(strike, **params)
        self.barrier = barrier

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        min_price = np.min(paths, axis=1)
        intrinsic = np.maximum(paths[:, -1] - self.strike, 0)
        return np.where(min_price > self.barrier, intrinsic, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "barrier": self.barrier, "barrier_type": "knock_out", "barrier_direction": "down"}


class BarrierDownOutPut(Instrument):
    """Down-and-Out Put: knocked out if S ever hits barrier."""

    def __init__(self, strike: float = 100.0, barrier: float = 80.0, **params):
        super().__init__(strike, **params)
        self.barrier = barrier

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        min_price = np.min(paths, axis=1)
        intrinsic = np.maximum(self.strike - paths[:, -1], 0)
        return np.where(min_price > self.barrier, intrinsic, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "barrier": self.barrier, "barrier_type": "knock_out", "barrier_direction": "down"}


class BarrierUpInCall(Instrument):
    """Up-and-In Call: activated only if S hits barrier."""

    def __init__(self, strike: float = 100.0, barrier: float = 120.0, **params):
        super().__init__(strike, **params)
        self.barrier = barrier

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        max_price = np.max(paths, axis=1)
        intrinsic = np.maximum(paths[:, -1] - self.strike, 0)
        return np.where(max_price >= self.barrier, intrinsic, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "barrier": self.barrier, "barrier_type": "knock_in", "barrier_direction": "up"}


class BarrierUpInPut(Instrument):
    """Up-and-In Put: activated only if S hits barrier."""

    def __init__(self, strike: float = 100.0, barrier: float = 120.0, **params):
        super().__init__(strike, **params)
        self.barrier = barrier

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        max_price = np.max(paths, axis=1)
        intrinsic = np.maximum(self.strike - paths[:, -1], 0)
        return np.where(max_price >= self.barrier, intrinsic, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "barrier": self.barrier, "barrier_type": "knock_in", "barrier_direction": "up"}


class BarrierDownInCall(Instrument):
    """Down-and-In Call: activated only if S hits barrier."""

    def __init__(self, strike: float = 100.0, barrier: float = 80.0, **params):
        super().__init__(strike, **params)
        self.barrier = barrier

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        min_price = np.min(paths, axis=1)
        intrinsic = np.maximum(paths[:, -1] - self.strike, 0)
        return np.where(min_price <= self.barrier, intrinsic, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "barrier": self.barrier, "barrier_type": "knock_in", "barrier_direction": "down"}


class BarrierDownInPut(Instrument):
    """Down-and-In Put: activated only if S hits barrier."""

    def __init__(self, strike: float = 100.0, barrier: float = 80.0, **params):
        super().__init__(strike, **params)
        self.barrier = barrier

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        min_price = np.min(paths, axis=1)
        intrinsic = np.maximum(self.strike - paths[:, -1], 0)
        return np.where(min_price <= self.barrier, intrinsic, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "barrier": self.barrier, "barrier_type": "knock_in", "barrier_direction": "down"}


# DOUBLE KNOCK OPTIONS
class DoubleKnockOutCall(Instrument):
    """Knocked out if S hits either barrier."""

    def __init__(self, strike: float = 100.0, upper_barrier: float = 120.0, lower_barrier: float = 80.0, **params):
        super().__init__(strike, **params)
        self.upper_barrier = upper_barrier
        self.lower_barrier = lower_barrier

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        max_price = np.max(paths, axis=1)
        min_price = np.min(paths, axis=1)
        intrinsic = np.maximum(paths[:, -1] - self.strike, 0)
        return np.where((max_price < self.upper_barrier) & (min_price > self.lower_barrier), intrinsic, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "upper_barrier": self.upper_barrier, "lower_barrier": self.lower_barrier}


class DoubleKnockOutPut(Instrument):
    """Put knocked out if S hits either barrier."""

    def __init__(self, strike: float = 100.0, upper_barrier: float = 120.0, lower_barrier: float = 80.0, **params):
        super().__init__(strike, **params)
        self.upper_barrier = upper_barrier
        self.lower_barrier = lower_barrier

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        max_price = np.max(paths, axis=1)
        min_price = np.min(paths, axis=1)
        intrinsic = np.maximum(self.strike - paths[:, -1], 0)
        return np.where((max_price < self.upper_barrier) & (min_price > self.lower_barrier), intrinsic, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "upper_barrier": self.upper_barrier, "lower_barrier": self.lower_barrier}


class DoubleKnockInCall(Instrument):
    """Activated only if S hits either barrier."""

    def __init__(self, strike: float = 100.0, upper_barrier: float = 120.0, lower_barrier: float = 80.0, **params):
        super().__init__(strike, **params)
        self.upper_barrier = upper_barrier
        self.lower_barrier = lower_barrier

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        max_price = np.max(paths, axis=1)
        min_price = np.min(paths, axis=1)
        intrinsic = np.maximum(paths[:, -1] - self.strike, 0)
        return np.where((max_price >= self.upper_barrier) | (min_price <= self.lower_barrier), intrinsic, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "upper_barrier": self.upper_barrier, "lower_barrier": self.lower_barrier}


class DoubleKnockInPut(Instrument):
    """Put activated only if S hits either barrier."""

    def __init__(self, strike: float = 100.0, upper_barrier: float = 120.0, lower_barrier: float = 80.0, **params):
        super().__init__(strike, **params)
        self.upper_barrier = upper_barrier
        self.lower_barrier = lower_barrier

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        max_price = np.max(paths, axis=1)
        min_price = np.min(paths, axis=1)
        intrinsic = np.maximum(self.strike - paths[:, -1], 0)
        return np.where((max_price >= self.upper_barrier) | (min_price <= self.lower_barrier), intrinsic, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "upper_barrier": self.upper_barrier, "lower_barrier": self.lower_barrier}


# LOOKBACK OPTIONS
class LookbackFloatStrikeCall(Instrument):
    """Call on max(S) - min(S)."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        max_price = np.max(paths, axis=1)
        min_price = np.min(paths, axis=1)
        return max_price - min_price

    def get_params(self) -> Dict[str, Any]:
        return {}


class LookbackFloatStrikePut(Instrument):
    """Put on max(S) - min(S)."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        max_price = np.max(paths, axis=1)
        min_price = np.min(paths, axis=1)
        return max_price - min_price

    def get_params(self) -> Dict[str, Any]:
        return {}


class LookbackFixedStrikeCall(Instrument):
    """max(S) - K."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        max_price = np.max(paths, axis=1)
        return np.maximum(max_price - self.strike, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


class LookbackFixedStrikePut(Instrument):
    """K - min(S)."""

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        min_price = np.min(paths, axis=1)
        return np.maximum(self.strike - min_price, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike}


# PARISIAN BARRIERS
class ParisianUpOutCall(Instrument):
    """Up-and-Out with time window constraint."""

    def __init__(self, strike: float = 100.0, barrier: float = 120.0, window: int = 5, **params):
        super().__init__(strike, **params)
        self.barrier = barrier
        self.window = window  # steps above barrier to knock out

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        # Simplified: check if ever stayed above barrier for window steps
        above_barrier = paths >= self.barrier
        num_paths = paths.shape[0]
        knocked_out = np.zeros(num_paths, dtype=bool)

        for i in range(num_paths):
            consecutive_count = 0
            for j in range(paths.shape[1]):
                if above_barrier[i, j]:
                    consecutive_count += 1
                    if consecutive_count >= self.window:
                        knocked_out[i] = True
                        break
                else:
                    consecutive_count = 0

        intrinsic = np.maximum(paths[:, -1] - self.strike, 0)
        return np.where(~knocked_out, intrinsic, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "barrier": self.barrier, "window": self.window}


class ParisianDownOutPut(Instrument):
    """Down-and-Out with time window constraint."""

    def __init__(self, strike: float = 100.0, barrier: float = 80.0, window: int = 5, **params):
        super().__init__(strike, **params)
        self.barrier = barrier
        self.window = window

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        below_barrier = paths <= self.barrier
        num_paths = paths.shape[0]
        knocked_out = np.zeros(num_paths, dtype=bool)

        for i in range(num_paths):
            consecutive_count = 0
            for j in range(paths.shape[1]):
                if below_barrier[i, j]:
                    consecutive_count += 1
                    if consecutive_count >= self.window:
                        knocked_out[i] = True
                        break
                else:
                    consecutive_count = 0

        intrinsic = np.maximum(self.strike - paths[:, -1], 0)
        return np.where(~knocked_out, intrinsic, 0.0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "barrier": self.barrier, "window": self.window}


# CHOOSER
class ChooserOption(Instrument):
    """Chooser: at maturity of choice, select call or put."""

    def __init__(self, strike: float = 100.0, choice_time: float = 0.5, **params):
        super().__init__(strike, **params)
        self.choice_time = choice_time

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        S_T = paths[:, -1]
        call_payoff = np.maximum(S_T - self.strike, 0)
        put_payoff = np.maximum(self.strike - S_T, 0)
        return np.maximum(call_payoff, put_payoff)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "choice_time": self.choice_time}


# COMPOUND OPTIONS
class CompoundCallOnCall(Instrument):
    """Option to buy a call on the underlying."""

    def __init__(self, strike: float = 100.0, option_strike: float = 5.0, **params):
        super().__init__(strike, **params)
        self.option_strike = option_strike

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        S_T = paths[:, -1]
        underlying_call = np.maximum(S_T - self.strike, 0)
        return np.maximum(underlying_call - self.option_strike, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "option_strike": self.option_strike}


class CompoundPutOnPut(Instrument):
    """Option to buy a put on the underlying."""

    def __init__(self, strike: float = 100.0, option_strike: float = 5.0, **params):
        super().__init__(strike, **params)
        self.option_strike = option_strike

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        S_T = paths[:, -1]
        underlying_put = np.maximum(self.strike - S_T, 0)
        return np.maximum(underlying_put - self.option_strike, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "option_strike": self.option_strike}


class CompoundCallOnPut(Instrument):
    """Call option on a put."""

    def __init__(self, strike: float = 100.0, option_strike: float = 5.0, **params):
        super().__init__(strike, **params)
        self.option_strike = option_strike

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        S_T = paths[:, -1]
        underlying_put = np.maximum(self.strike - S_T, 0)
        return np.maximum(underlying_put - self.option_strike, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "option_strike": self.option_strike}


class CompoundPutOnCall(Instrument):
    """Put option on a call."""

    def __init__(self, strike: float = 100.0, option_strike: float = 5.0, **params):
        super().__init__(strike, **params)
        self.option_strike = option_strike

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        S_T = paths[:, -1]
        underlying_call = np.maximum(S_T - self.strike, 0)
        return np.maximum(self.option_strike - underlying_call, 0)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "option_strike": self.option_strike}

