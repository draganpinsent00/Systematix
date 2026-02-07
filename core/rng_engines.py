"""
Random Number Generator engines supporting multiple algorithms.

RNG Interface Contract:
- All engines must conform to the same interface
- uniform(size) → iid U(0,1)
- standard_normal(size) → iid N(0,1)
- Guarantees: Uniforms in (0,1), Normals have unit variance
- No shared state across independent calls
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class RNGEngineBase(ABC):
    """Abstract base class for RNG engines."""

    def __init__(self, seed: int = 42):
        """Initialize RNG with seed."""
        self.seed = seed
        self._setup_state(seed)

    @abstractmethod
    def _setup_state(self, seed: int) -> None:
        """Initialize internal state."""
        pass

    @abstractmethod
    def standard_normal(self, size: Tuple[int, ...]) -> np.ndarray:
        """
        Generate standard normal random variables N(0,1).

        Returns:
            Array of shape `size` with iid N(0,1) samples.
            Mean ≈ 0, Variance ≈ 1.
        """
        pass

    @abstractmethod
    def uniform(self, size: Tuple[int, ...]) -> np.ndarray:
        """
        Generate uniform random variables U(0,1).

        Returns:
            Array of shape `size` with iid U(0,1) samples.
            All values strictly in (0,1).
        """
        pass

    def normal(self, mean: float = 0.0, std: float = 1.0, size: Tuple[int, ...] = (1,)) -> np.ndarray:
        """Generate normal(mean, std) random variables."""
        return mean + std * self.standard_normal(size)

    def poisson(self, lam: float, size: Tuple[int, ...] = (1,)) -> np.ndarray:
        """Generate Poisson(lam) random variables using seeded generator."""
        # Use numpy's poisson generator from the underlying Generator object
        # All concrete RNG engines have self.generator attribute
        if hasattr(self, 'generator'):
            return self.generator.poisson(lam, size)
        else:
            # Defensive programming: should not reach here for standard engines
            # Create a temporary generator from seed to maintain reproducibility
            temp_gen = np.random.Generator(np.random.PCG64(self.seed))
            return temp_gen.poisson(lam, size)

    def correlated_normals(self, mean: np.ndarray, cov: np.ndarray, num_samples: int) -> np.ndarray:
        """Generate correlated normal random variables using Cholesky decomposition."""
        dim = cov.shape[0]
        L = np.linalg.cholesky(cov)
        Z = self.standard_normal((num_samples, dim))
        return mean + Z @ L.T


class MersenneTwister(RNGEngineBase):
    """NumPy's Mersenne Twister (MT19937) via modern Generator interface."""

    def _setup_state(self, seed: int) -> None:
        self.generator = np.random.Generator(np.random.MT19937(seed))

    def standard_normal(self, size: Tuple[int, ...]) -> np.ndarray:
        """Generate N(0,1) samples."""
        return self.generator.standard_normal(size)

    def uniform(self, size: Tuple[int, ...]) -> np.ndarray:
        """Generate U(0,1) samples."""
        return self.generator.uniform(0, 1, size)


class PCG64RNG(RNGEngineBase):
    """PCG64 PRNG via NumPy's Generator interface."""

    def _setup_state(self, seed: int) -> None:
        self.generator = np.random.Generator(np.random.PCG64(seed))

    def standard_normal(self, size: Tuple[int, ...]) -> np.ndarray:
        """Generate N(0,1) samples."""
        return self.generator.standard_normal(size)

    def uniform(self, size: Tuple[int, ...]) -> np.ndarray:
        """Generate U(0,1) samples."""
        return self.generator.uniform(0, 1, size)


class XorShift128Plus(RNGEngineBase):
    """XorShift via NumPy's SFC64 (similar performance/quality)."""

    def _setup_state(self, seed: int) -> None:
        self.generator = np.random.Generator(np.random.SFC64(seed))

    def standard_normal(self, size: Tuple[int, ...]) -> np.ndarray:
        """Generate N(0,1) samples."""
        return self.generator.standard_normal(size)

    def uniform(self, size: Tuple[int, ...]) -> np.ndarray:
        """Generate U(0,1) samples."""
        return self.generator.uniform(0, 1, size)


class Philox(RNGEngineBase):
    """Philox counter-based PRNG via NumPy's Generator interface."""

    def _setup_state(self, seed: int) -> None:
        self.generator = np.random.Generator(np.random.Philox(seed))

    def standard_normal(self, size: Tuple[int, ...]) -> np.ndarray:
        """Generate N(0,1) samples."""
        return self.generator.standard_normal(size)

    def uniform(self, size: Tuple[int, ...]) -> np.ndarray:
        """Generate U(0,1) samples."""
        return self.generator.uniform(0, 1, size)


class MiddleSquare(RNGEngineBase):
    """
    Middle-Square PRNG (Neumann).

    WARNING: This is a historical reference implementation.
    Quality is poor - use only for educational purposes.
    For production use, select another engine.

    Falls back to Mersenne Twister for normal generation
    to ensure statistical correctness.
    """

    def _setup_state(self, seed: int) -> None:
        # Initialize state to 8-digit range
        self.state = abs(seed) % (10**8)
        if self.state == 0:
            self.state = 1234567
        # Fallback for normal generation
        self.fallback = np.random.Generator(np.random.MT19937(seed))

    def _next_uniform(self) -> float:
        """Generate next uniform(0,1) value using middle-square."""
        # Square the state and extract middle digits
        squared = self.state * self.state
        # Extract middle 8 digits
        self.state = (squared // 100) % (10**8)
        if self.state == 0:
            self.state = 1
        # Normalize to [0,1)
        return self.state / (10**8)

    def standard_normal(self, size: Tuple[int, ...]) -> np.ndarray:
        """
        Generate N(0,1) using fallback (Mersenne Twister).

        Middle-Square quality is too poor for accurate normal distribution.
        """
        return self.fallback.standard_normal(size)

    def uniform(self, size: Tuple[int, ...]) -> np.ndarray:
        """Generate uniform [0,1) variates."""
        n_total = int(np.prod(size))
        result = np.zeros(n_total)
        for i in range(n_total):
            result[i] = self._next_uniform()
        return result.reshape(size)


ENGINE_MAPPING = {
    "mersenne": MersenneTwister,
    "pcg64": PCG64RNG,
    "xorshift": XorShift128Plus,
    "philox": Philox,
    "middle_square": MiddleSquare,
}


def create_rng(engine: str = "mersenne", seed: int = 42) -> RNGEngineBase:
    """
    Factory function to create RNG engine.

    Args:
        engine: Name of RNG engine ("mersenne", "pcg64", "xorshift", "philox", "middle_square")
        seed: Random seed for reproducibility

    Returns:
        RNGEngineBase instance with specified engine

    Raises:
        ValueError: If engine name is not recognized
    """
    engine_lower = engine.lower()
    if engine_lower not in ENGINE_MAPPING:
        raise ValueError(f"Unknown RNG engine: {engine}. Available: {list(ENGINE_MAPPING.keys())}")
    return ENGINE_MAPPING[engine_lower](seed)

