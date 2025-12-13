from abc import ABC, abstractmethod
import numpy as np


class Payoff(ABC):
    @abstractmethod
    def payoff(self, paths: np.ndarray) -> np.ndarray:
        """Compute path-dependent payoff. paths shape: (n_paths, steps+1)"""

    def payoff_at_maturity(self, ST: np.ndarray) -> np.ndarray:
        """Optional convenience for payoffs that depend only on terminal price."""
        raise NotImplementedError


class EuropeanCall(Payoff):
    def __init__(self, K: float):
        self.K = float(K)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        ST = paths[:, -1]
        return np.maximum(ST - self.K, 0.0)

    def payoff_at_maturity(self, ST: np.ndarray) -> np.ndarray:
        return np.maximum(ST - self.K, 0.0)


class EuropeanPut(Payoff):
    def __init__(self, K: float):
        self.K = float(K)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        ST = paths[:, -1]
        return np.maximum(self.K - ST, 0.0)

    def payoff_at_maturity(self, ST: np.ndarray) -> np.ndarray:
        return np.maximum(self.K - ST, 0.0)


class AsianArithmeticCall(Payoff):
    def __init__(self, K: float):
        self.K = float(K)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        # arithmetic average excluding S0
        avg = paths[:, 1:].mean(axis=1)
        return np.maximum(avg - self.K, 0.0)


class AsianArithmeticPut(Payoff):
    def __init__(self, K: float):
        self.K = float(K)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        # arithmetic average excluding S0
        avg = paths[:, 1:].mean(axis=1)
        return np.maximum(self.K - avg, 0.0)


class AsianGeometricCall(Payoff):
    def __init__(self, K: float):
        self.K = float(K)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        # geometric average excluding S0
        geo = np.exp(np.mean(np.log(np.maximum(paths[:, 1:], 1e-12)), axis=1))
        return np.maximum(geo - self.K, 0.0)


class AsianFloatingStrikeCall(Payoff):
    def __init__(self):
        pass

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        # payoff = max(S_T - average, 0)
        avg = paths[:, 1:].mean(axis=1)
        ST = paths[:, -1]
        return np.maximum(ST - avg, 0.0)


class AsianFloatingStrikePut(Payoff):
    def __init__(self):
        pass

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        avg = paths[:, 1:].mean(axis=1)
        ST = paths[:, -1]
        return np.maximum(avg - ST, 0.0)


class LookbackFloatingCall(Payoff):
    def __init__(self):
        pass

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        # payoff: S_T - min_{t<=T} S_t
        minS = paths.min(axis=1)
        ST = paths[:, -1]
        return np.maximum(ST - minS, 0.0)


class LookbackFixedStrikeCall(Payoff):
    def __init__(self, K: float):
        self.K = float(K)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        maxS = paths.max(axis=1)
        return np.maximum(maxS - self.K, 0.0)


class LookbackFixedStrikePut(Payoff):
    def __init__(self, K: float):
        self.K = float(K)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        # payoff: max(K - min S, 0)
        minS = paths.min(axis=1)
        return np.maximum(self.K - minS, 0.0)


class LookbackFloatingPut(Payoff):
    def __init__(self):
        pass

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        maxS = paths.max(axis=1)
        ST = paths[:, -1]
        return np.maximum(maxS - ST, 0.0)


class DigitalCash(Payoff):
    def __init__(self, K: float, cash: float = 1.0, at_maturity: bool = True):
        self.K = float(K)
        self.cash = float(cash)
        self.at_maturity = bool(at_maturity)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        if self.at_maturity:
            ST = paths[:, -1]
            return np.where(ST >= self.K, self.cash, 0.0)
        else:
            breached = np.any(paths >= self.K, axis=1)
            return np.where(breached, self.cash, 0.0)


class DigitalAsset(Payoff):
    def __init__(self, K: float, at_maturity: bool = True):
        self.K = float(K)
        self.at_maturity = bool(at_maturity)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        if self.at_maturity:
            ST = paths[:, -1]
            return np.where(ST >= self.K, ST, 0.0)
        else:
            # asset-or-nothing if barrier breached
            breached = np.any(paths >= self.K, axis=1)
            ST = paths[:, -1]
            return np.where(breached, ST, 0.0)


class BarrierPayoff(Payoff):
    def __init__(self, K: float, barrier_low: float = None, barrier_high: float = None, barrier_type: str = 'up-and-out', rebate: float = 0.0):
        """Barrier option supporting single and double barriers.

        barrier_type examples: 'up-and-out', 'down-and-out', 'up-and-in', 'down-and-in', 'double-knock-out', 'double-knock-in'
        For double barriers provide barrier_low and barrier_high.
        """
        self.K = float(K)
        self.barrier_low = None if barrier_low is None else float(barrier_low)
        self.barrier_high = None if barrier_high is None else float(barrier_high)
        self.barrier_type = barrier_type
        self.rebate = float(rebate)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        ST = paths[:, -1]
        if self.barrier_type in ('up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'):
            if 'up' in self.barrier_type:
                breached = np.any(paths >= (self.barrier_high if self.barrier_high is not None else self.barrier_low), axis=1)
            else:
                breached = np.any(paths <= (self.barrier_low if self.barrier_low is not None else self.barrier_high), axis=1)
            if self.barrier_type.endswith('out'):
                return np.where(breached, self.rebate, np.maximum(ST - self.K, 0.0))
            else:
                return np.where(breached, np.maximum(ST - self.K, 0.0), self.rebate)
        elif self.barrier_type in ('double-knock-out', 'double-knock-in'):
            if self.barrier_low is None or self.barrier_high is None:
                raise ValueError('Double barrier requires barrier_low and barrier_high')
            breached_low = np.any(paths <= self.barrier_low, axis=1)
            breached_high = np.any(paths >= self.barrier_high, axis=1)
            breached_any = breached_low | breached_high
            if self.barrier_type == 'double-knock-out':
                return np.where(breached_any, self.rebate, np.maximum(ST - self.K, 0.0))
            else:
                return np.where(breached_any, np.maximum(ST - self.K, 0.0), self.rebate)
        else:
            raise ValueError(f'Unknown barrier_type: {self.barrier_type}')


class BarrierDigitalCash(Payoff):
    def __init__(self, K: float, barrier_low: float = None, barrier_high: float = None, barrier_type: str = 'up-and-out', cash: float = 1.0, at_maturity: bool = True):
        self.K = float(K)
        self.barrier_low = None if barrier_low is None else float(barrier_low)
        self.barrier_high = None if barrier_high is None else float(barrier_high)
        self.barrier_type = barrier_type
        self.cash = float(cash)
        self.at_maturity = bool(at_maturity)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        # Determine breach
        if self.barrier_type in ('up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'):
            if 'up' in self.barrier_type:
                breached = np.any(paths >= (self.barrier_high if self.barrier_high is not None else self.barrier_low), axis=1)
            else:
                breached = np.any(paths <= (self.barrier_low if self.barrier_low is not None else self.barrier_high), axis=1)
            if self.barrier_type.endswith('out'):
                # pays if NOT breached
                return np.where(~breached, self.cash, 0.0)
            else:
                # pays if breached
                return np.where(breached, self.cash, 0.0)
        elif self.barrier_type in ('double-knock-out', 'double-knock-in'):
            if self.barrier_low is None or self.barrier_high is None:
                raise ValueError('Double barrier requires barrier_low and barrier_high')
            breached_low = np.any(paths <= self.barrier_low, axis=1)
            breached_high = np.any(paths >= self.barrier_high, axis=1)
            breached_any = breached_low | breached_high
            if self.barrier_type == 'double-knock-out':
                return np.where(~breached_any, self.cash, 0.0)
            else:
                return np.where(breached_any, self.cash, 0.0)
        else:
            raise ValueError(f'Unknown barrier_type: {self.barrier_type}')


class BarrierDigitalAsset(Payoff):
    def __init__(self, K: float, barrier_low: float = None, barrier_high: float = None, barrier_type: str = 'up-and-out', at_maturity: bool = True):
        self.K = float(K)
        self.barrier_low = None if barrier_low is None else float(barrier_low)
        self.barrier_high = None if barrier_high is None else float(barrier_high)
        self.barrier_type = barrier_type
        self.at_maturity = bool(at_maturity)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        ST = paths[:, -1]
        if self.barrier_type in ('up-and-out', 'down-and-out', 'up-and-in', 'down-and-in'):
            if 'up' in self.barrier_type:
                breached = np.any(paths >= (self.barrier_high if self.barrier_high is not None else self.barrier_low), axis=1)
            else:
                breached = np.any(paths <= (self.barrier_low if self.barrier_low is not None else self.barrier_high), axis=1)
            if self.barrier_type.endswith('out'):
                return np.where(~breached, ST, 0.0)
            else:
                return np.where(breached, ST, 0.0)
        elif self.barrier_type in ('double-knock-out', 'double-knock-in'):
            if self.barrier_low is None or self.barrier_high is None:
                raise ValueError('Double barrier requires barrier_low and barrier_high')
            breached_low = np.any(paths <= self.barrier_low, axis=1)
            breached_high = np.any(paths >= self.barrier_high, axis=1)
            breached_any = breached_low | breached_high
            if self.barrier_type == 'double-knock-out':
                return np.where(~breached_any, ST, 0.0)
            else:
                return np.where(breached_any, ST, 0.0)
        else:
            raise ValueError(f'Unknown barrier_type: {self.barrier_type}')


class BasketCall(Payoff):
    def __init__(self, weights, K: float):
        self.weights = np.asarray(weights, dtype=float)
        self.K = float(K)

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        # paths shape: (n_paths, steps+1, d) for multivariate
        if paths.ndim != 3:
            raise ValueError('paths must be 3D (n_paths, steps+1, d) for BasketCall')
        ST = paths[:, -1, :]
        basket = ST.dot(self.weights)
        return np.maximum(basket - self.K, 0.0)


__all__ = [
    'Payoff', 'EuropeanCall', 'EuropeanPut',
    'AsianArithmeticCall', 'AsianArithmeticPut', 'AsianGeometricCall', 'AsianFloatingStrikeCall', 'AsianFloatingStrikePut',
    'LookbackFloatingCall', 'LookbackFixedStrikeCall', 'LookbackFixedStrikePut', 'LookbackFloatingPut',
    'DigitalCash', 'DigitalAsset', 'BarrierPayoff', 'BarrierDigitalCash', 'BarrierDigitalAsset', 'BasketCall'
]
