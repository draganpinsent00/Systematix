"""RNG and distribution adapters.

Provides simple factories for numpy PRNG, Sobol quasi-Monte Carlo mapped to normals,
and a simple stratified RNG. SobolRNG supports an optional brownian_bridge ordering
that reassigns Sobol dimensions to time steps to improve QMC performance for path-dependent payoffs.
"""
from typing import Optional, Tuple
import numpy as np

try:
    from scipy.stats import qmc, norm
    _HAS_SCIPY = True
except Exception:
    qmc = None
    norm = None
    _HAS_SCIPY = False


class NumpyRNG:
    """Wrapper around numpy.random.Generator providing a small, stable API."""
    def __init__(self, seed: Optional[int] = None):
        self._gen = np.random.default_rng(seed)
        self.name = 'numpy_pcg64'

    def seed(self, seed: Optional[int]):
        self._gen = np.random.default_rng(seed)

    def standard_normal(self, size: Tuple[int, ...]):
        return self._gen.standard_normal(size=size)


class SobolRNG:
    """Sobol quasi-Monte Carlo mapped to standard normals via inverse CDF.

    Parameters
    - dim: number of dimensions (time steps) expected
    - scramble: whether to scramble the Sobol sequence
    - seed: seed for scrambling
    - brownian_bridge: whether to reorder QMC dimensions using a Brownian-bridge order
      (this assigns low-discrepancy coordinates to more important time indices)
    """
    def __init__(self, dim: int, scramble: bool = True, seed: Optional[int] = None, brownian_bridge: bool = False):
        if not _HAS_SCIPY:
            raise RuntimeError('SciPy not available for Sobol generator')
        self.dim = int(dim)
        self.scramble = bool(scramble)
        self.seed_val = seed
        self._sampler = qmc.Sobol(self.dim, scramble=self.scramble, seed=seed)
        self.name = 'sobol'
        self.brownian_bridge = bool(brownian_bridge)
        # precompute bridge order
        self._bridge_order = self._compute_bridge_order(self.dim) if self.brownian_bridge else None

    def seed(self, seed: Optional[int]):
        self.seed_val = seed
        self._sampler = qmc.Sobol(self.dim, scramble=self.scramble, seed=seed)
        if self.brownian_bridge:
            self._bridge_order = self._compute_bridge_order(self.dim)

    @staticmethod
    def _compute_bridge_order(n: int):
        """Compute a Brownian-bridge ordering of indices 0..n-1.

        We will ensure the last index (n) is produced first (endpoint), followed by midpoints
        recursively; this ordering is suitable for bridge construction where endpoint is drawn
        first then midpoints conditioned on known endpoints.
        """
        order = []
        # we will work with node indices 1..n (positions at times i/n). We'll store indices 1..n
        def rec(l, r):
            if l > r:
                return
            mid = (l + r) // 2
            order.append(mid)
            rec(l, mid - 1)
            rec(mid + 1, r)
        # include final time n first
        order.append(n)
        rec(0, n - 1)
        # convert node indices in 1..n to zero-based increments positions 0..n-1
        # remove any duplicates and clamp
        seen = set()
        final_order = []
        for idx in order:
            if idx < 1 or idx > n:
                continue
            pos = idx - 1
            if pos not in seen:
                final_order.append(pos)
                seen.add(pos)
        # if somehow we are missing positions, append them
        for p in range(n):
            if p not in seen:
                final_order.append(p)
        return final_order

    def _apply_brownian_bridge(self, Z: np.ndarray) -> np.ndarray:
        """Given Z shape (n, dim) where columns correspond to bridge normals in order self._bridge_order,
        construct Brownian increments for each row.

        Returns array of shape (n, dim) where columns are increments dW_i for i=1..dim normalized to unit-time.
        """
        n, dim = Z.shape
        # times t_i = i/dim for i=0..dim
        t = np.linspace(0.0, 1.0, dim + 1)
        increments = np.zeros((n, dim), dtype=float)
        # For each sample row, build W values
        for i in range(n):
            W = np.zeros(dim + 1, dtype=float)
            known = {0: 0.0}
            # Use Z columns in index order as provided by bridge_order
            z_row = Z[i, :]
            # track which Sobol column to use
            k = 0
            # set endpoint W[dim] using first available z
            W[dim] = np.sqrt(1.0) * z_row[k]
            known[dim] = W[dim]
            k += 1
            # Now fill in according to bridge order (which excludes some duplicates)
            # We'll iterate until all interior points assigned
            # use a queue of intervals
            intervals = [(0, dim)]
            while intervals and k < dim:
                a, b = intervals.pop(0)
                mid = (a + b) // 2
                if mid == a or mid == b:
                    continue
                if mid in known:
                    # subdivide
                    intervals.append((a, mid))
                    intervals.append((mid, b))
                    continue
                # compute conditional mean and variance for W[mid] given W[a], W[b]
                ta = t[a]
                tb = t[b]
                tm = t[mid]
                Wa = known.get(a, None)
                Wb = known.get(b, None)
                if Wa is None or Wb is None:
                    # ensure endpoints are known by pushing back
                    intervals.append((a, b))
                    continue
                mean = ((tm - ta) / (tb - ta)) * Wb + ((tb - tm) / (tb - ta)) * Wa
                var = (tm - ta) * (tb - tm) / (tb - ta)
                # draw using next normal
                Wm = mean + np.sqrt(max(0.0, var)) * z_row[k]
                W[mid] = Wm
                known[mid] = Wm
                k += 1
                # subdivide
                intervals.append((a, mid))
                intervals.append((mid, b))
            # if any remaining unknown points, fill by linear interpolation
            for j in range(1, dim):
                if j not in known:
                    # find nearest known neighbors
                    left = max([idx for idx in known.keys() if idx < j])
                    right = min([idx for idx in known.keys() if idx > j])
                    Wa = known[left]
                    Wb = known[right]
                    tm = t[j]
                    ta = t[left]
                    tb = t[right]
                    W[j] = ((tm - ta) / (tb - ta)) * Wb + ((tb - tm) / (tb - ta)) * Wa
            # compute increments
            dW = np.diff(W)
            increments[i, :] = dW
        return increments

    def standard_normal(self, size: Tuple[int, int]):
        # size expected as (n, dim)
        n, d = size
        if d != self.dim:
            # re-init sampler and bridge order for the new dimension
            self.dim = d
            self._sampler = qmc.Sobol(self.dim, scramble=self.scramble, seed=self.seed_val)
            if self.brownian_bridge:
                self._bridge_order = self._compute_bridge_order(self.dim)
        u = self._sampler.random(n)
        # numeric-safe mapping: clamp to (eps, 1-eps) to avoid +/-inf from ppf
        eps = 1e-16
        u = np.clip(u, eps, 1.0 - eps)
        Z = norm.ppf(u)
        if self.brownian_bridge:
            # map Sobol columns into bridge normals order
            if self._bridge_order is None:
                self._bridge_order = self._compute_bridge_order(d)
            # create array where column k in bridge_cols is Z[:, k]
            # we need to permute columns so that col k corresponds to bridge_order[k]
            permuted = np.empty_like(Z)
            # If bridge order length == d, assign accordingly
            for k, time_idx in enumerate(self._bridge_order):
                if k < d and time_idx < d:
                    permuted[:, k] = Z[:, k]
            # If there are remaining columns, fill them
            for col in range(d):
                if not np.any(permuted[:, col] != 0):
                    permuted[:, col] = Z[:, col]
            # apply brownian bridge to permuted normals
            increments = self._apply_brownian_bridge(permuted)
            return increments
        return Z


class StratifiedRNG:
    """Simple stratified sampling mapped to normals.

    This implements basic stratification by dividing [0,1) in n strata along sample index
    and jittering inside each stratum for each coordinate. It's a lightweight Latin-like
    stratification (not a full LHS implementation) suitable as a variance-reduction baseline.
    """
    def __init__(self, dim: int, seed: Optional[int] = None):
        self.dim = int(dim)
        self._gen = np.random.default_rng(seed)
        self.name = 'stratified'

    def seed(self, seed: Optional[int]):
        self._gen = np.random.default_rng(seed)

    def standard_normal(self, size: Tuple[int, int]):
        n, d = size
        if d != self.dim:
            self.dim = d
        # stratify along sample index: u_ij = (i + U_ij) / n
        jitter = self._gen.random((n, d))
        strata = (np.arange(n).reshape(n, 1) + jitter) / float(n)
        return norm.ppf(strata)


def get_rng(kind: str = 'numpy', seed: Optional[int] = None, dim: Optional[int] = None, **kwargs):
    kind = kind.lower() if kind is not None else 'numpy'
    if kind in ('numpy', 'pcg64', 'default'):
        return NumpyRNG(seed)
    if kind in ('sobol', 'qmc', 'quasi'):
        if dim is None:
            raise ValueError('dim must be provided for Sobol RNG')
        return SobolRNG(dim=dim, scramble=kwargs.get('scramble', True), seed=seed, brownian_bridge=kwargs.get('brownian_bridge', False))
    if kind in ('stratified', 'strat'):
        if dim is None:
            raise ValueError('dim must be provided for Stratified RNG')
        return StratifiedRNG(dim=dim, seed=seed)
    raise ValueError(f'Unknown RNG kind: {kind}')
