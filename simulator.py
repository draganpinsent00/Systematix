"""
simulator.py

Vectorized Monte Carlo path generator for basic models (GBM + Heston).
Supports antithetic variates and quasi-random Sobol draws for GBM.
"""
from typing import Optional, Tuple
import numpy as np

# optional import for quasi-random Sobol and Latin Hypercube
try:
    from scipy.stats import qmc
    from scipy.stats import norm
    _HAS_QMC = True
except Exception:
    qmc = None
    norm = None
    _HAS_QMC = False

# available bitgenerators mapping
_BITGENS = {
    'pcg64': np.random.PCG64,
    'mt19937': np.random.MT19937,
    'sfc64': np.random.SFC64,
}


class MiddleSquareRNG:
    """Simple middle-square RNG for demonstration (not cryptographically strong).
    Produces uniform floats in (0,1).
    """
    def __init__(self, seed: int = 12345):
        # seed as integer, keep 8-digit state
        self.state = int(seed) % 10**8

    def random(self, size):
        flat = int(np.prod(size))
        out = np.empty(flat, dtype=float)
        s = self.state
        for i in range(flat):
            s2 = (s * s) % 10**8
            # middle 4 digits
            s = (s2 // 100) % 10000
            out[i] = s / 10000.0
            if s == 0:
                s = (i + 1) * 1234567 % 10**8
        self.state = s
        return out.reshape(size)


def _build_rng(rng_name: str, seed: Optional[int] = None) -> np.random.Generator:
    if rng_name is None:
        return np.random.default_rng(seed)
    name = str(rng_name).lower()
    if name in _BITGENS:
        bit = _BITGENS[name](seed)
        return np.random.Generator(bit)
    if name == 'middle_square':
        # we will return None and handle separately
        return None
    # fallback
    return np.random.default_rng(seed)


def _middle_square_normals(shape, seed: Optional[int] = None):
    ms = MiddleSquareRNG(seed if seed is not None else 12345)
    u = ms.random(shape)
    # inverse normal via numpy's erfinv: use scipy if available, else approximate via numpy
    # use norm.ppf if scipy is available
    if _HAS_QMC and norm is not None:
        return norm.ppf(u)
    else:
        # approximate inverse CDF via erfinv
        return np.sqrt(2) * np.erfinv(2 * u - 1)


def _draw_normals(rng_name: Optional[str], seed: Optional[int], shape, dist: str = 'normal', sampler: str = 'pseudo', moment_match: bool = False, stratified: bool = False, use_bridge: bool = False, dt: Optional[float] = None):
    """Return standard normal draws of given shape: (n_paths, steps).
    sampler: 'pseudo', 'sobol', 'stratified'
    dist: 'normal', 't', 'lognormal'
    rng_name: 'pcg64','mt19937','sfc64','middle_square' or None
    """
    n, d = shape
    # Sobol / Latin Hypercube
    if sampler in ('sobol', 'quasi') and _HAS_QMC:
        sampler_q = qmc.Sobol(d, scramble=True, seed=seed)
        u = sampler_q.random(n)
        Z = norm.ppf(u)
        # If requested, apply Brownian bridge transform to map independent normals to Brownian increments
        if use_bridge:
            # dt must be provided by caller
            if dt is None:
                raise ValueError('dt must be provided for Brownian bridge transform')
            # Build covariance matrix Sigma for Brownian motion at times t_i = i*dt
            times = np.arange(1, d + 1) * float(dt)
            # Sigma_{ij} = min(t_i, t_j)
            Sigma = np.minimum(times[:, None], times[None, :]).astype(float)
            # numerical jitter for stability
            jitter = 1e-12 * np.eye(d)
            try:
                L = np.linalg.cholesky(Sigma + jitter)
            except np.linalg.LinAlgError:
                # fallback to toeplitz-based approach or small diag increase
                L = np.linalg.cholesky(Sigma + 1e-8 * np.eye(d))
            # Map independent normals Z (n x d) to Brownian values W = Z @ L.T
            W = Z.dot(L.T)
            # increments dW: W_i - W_{i-1} with W_0 = 0
            dW = np.empty_like(W)
            dW[:, 0] = W[:, 0]
            dW[:, 1:] = W[:, 1:] - W[:, :-1]
            # standardize increments to unit normals: divide by sqrt(dt_i)
            # For equal dt, sqrt_dt = sqrt(dt)
            sqrt_dt = np.sqrt(float(dt))
            Z_inc = dW / (sqrt_dt + 1e-16)
            Z = Z_inc
    elif stratified and _HAS_QMC:
        sampler_q = qmc.LatinHypercube(d=d, seed=seed)
        u = sampler_q.random(n)
        Z = norm.ppf(u)
    else:
        # pseudo RNG via chosen bitgenerator or middle-square
        gen = _build_rng(rng_name, seed)
        if gen is None:
            Z = _middle_square_normals((n, d), seed)
        else:
            Z = gen.standard_normal(size=(n, d))

    # handle t-distribution
    if dist == 't':
        # default df=4
        df = 4
        if _HAS_QMC:
            # convert standard normals to t via chi2
            # simplest: use scipy.stats.t.ppf on uniforms but fallback
            pass
        # fallback: use normal scaled by sqrt(df/(chi2/df)) approx via gen if available
    # moment matching: adjust columns to have zero mean and unit std
    if moment_match:
        Z_mean = Z.mean(axis=0, keepdims=True)
        Z_std = Z.std(axis=0, ddof=1, keepdims=True)
        Z = (Z - Z_mean) / (Z_std + 1e-12)
    return Z


# update simulate_gbm_paths signature to accept rng_name, sampler, moment_match, stratified, use_bridge
def simulate_gbm_paths(S0: float, r: float, sigma: float, T: float, steps: int, n_paths: int, seed: Optional[int] = None,
                       antithetic: bool = False, dist: str = 'normal', dist_params: Optional[dict] = None,
                       rng_name: Optional[str] = None, sampler: str = 'pseudo', moment_match: bool = False, stratified: bool = False, use_bridge: bool = False,
                       normal_shift: float = 0.0, return_sum_z: bool = False) -> np.ndarray:
    """Simulate Geometric Brownian Motion price paths (vectorized).

    Returns S (n_paths, steps+1).
    """
    if steps <= 0:
        raise ValueError('steps must be > 0')
    if n_paths <= 0:
        raise ValueError('n_paths must be > 0')

    dt = T / steps

    use_antithetic = bool(antithetic)
    half = n_paths
    if use_antithetic:
        half = (n_paths + 1) // 2

    # pass dt to _draw_normals so Brownian bridge can construct covariance
    Z = _draw_normals(rng_name, seed, (half, steps), dist=dist, sampler=sampler, moment_match=moment_match, stratified=stratified, use_bridge=use_bridge, dt=dt)
    # apply normal shift for importance sampling (shift each increment by normal_shift)
    if normal_shift and abs(float(normal_shift)) > 0.0:
        Z = Z + float(normal_shift)

    drift = (r - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    log_increments = drift + diffusion * Z
    log_paths = np.cumsum(log_increments, axis=1)

    S = np.empty((half, steps + 1), dtype=float)
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(log_paths)

    if use_antithetic:
        # antithetic by negating Z (works for normal-based draws)
        Z2 = -Z
        log_increments2 = drift + diffusion * Z2
        log_paths2 = np.cumsum(log_increments2, axis=1)
        S2 = np.empty_like(S)
        S2[:, 0] = S0
        S2[:, 1:] = S0 * np.exp(log_paths2)
        S_full = np.vstack([S, S2])
        if return_sum_z:
            sum_z = np.sum(np.vstack([Z, Z2])[:n_paths, :], axis=1)
            return S_full[:n_paths, :], sum_z
        return S_full[:n_paths, :]
    else:
        if return_sum_z:
            sum_z = np.sum(Z, axis=1)
            return S, sum_z
        return S


# update simulate_heston_paths accordingly
def simulate_heston_paths(S0: float, r: float, v0: float, kappa: float, theta: float, xi: float, rho: float,
                           T: float, steps: int, n_paths: int, seed: Optional[int] = None,
                           antithetic: bool = False, dist: str = 'normal', dist_params: Optional[dict] = None,
                           rng_name: Optional[str] = None, sampler: str = 'pseudo', moment_match: bool = False, stratified: bool = False, use_bridge: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate Heston stochastic volatility model (Euler discretization with full truncation for variance).

    Returns S (n_paths, steps+1), V (n_paths, steps+1)
    """
    if steps <= 0:
        raise ValueError('steps must be > 0')
    if n_paths <= 0:
        raise ValueError('n_paths must be > 0')

    dt = T / steps

    use_antithetic = bool(antithetic)
    half = n_paths
    if use_antithetic:
        half = (n_paths + 1) // 2

    Z1 = _draw_normals(rng_name, seed, (half, steps), dist=dist, sampler=sampler, moment_match=moment_match, stratified=stratified, use_bridge=use_bridge)
    # draw independent second normals using same RNG seed offset
    Z2 = _draw_normals(rng_name, (None if seed is None else seed+1), (half, steps), dist=dist, sampler=sampler, moment_match=moment_match, stratified=stratified, use_bridge=use_bridge)

    S = np.zeros((half, steps + 1), dtype=float)
    V = np.zeros((half, steps + 1), dtype=float)
    S[:, 0] = S0
    V[:, 0] = v0

    for t in range(1, steps + 1):
        z1 = Z1[:, t - 1]
        z2 = Z2[:, t - 1]
        dWv = np.sqrt(dt) * z1
        dWs = np.sqrt(dt) * (rho * z1 + np.sqrt(max(0.0, 1 - rho ** 2)) * z2)
        V_prev = V[:, t - 1]
        V_t = V_prev + kappa * (theta - np.maximum(V_prev, 0.0)) * dt + xi * np.sqrt(np.maximum(V_prev, 0.0)) * dWv
        V_t = np.maximum(V_t, 0.0)
        V[:, t] = V_t
        S[:, t] = S[:, t - 1] * np.exp((r - 0.5 * V_t) * dt + np.sqrt(np.maximum(V_t, 0.0)) * dWs)

    if use_antithetic:
        Z1b = -Z1
        Z2b = -Z2
        S2 = np.zeros_like(S)
        V2 = np.zeros_like(V)
        S2[:, 0] = S0
        V2[:, 0] = v0
        for t in range(1, steps + 1):
            z1 = Z1b[:, t - 1]
            z2 = Z2b[:, t - 1]
            dWv = np.sqrt(dt) * z1
            dWs = np.sqrt(dt) * (rho * z1 + np.sqrt(max(0.0, 1 - rho ** 2)) * z2)
            V_prev = V2[:, t - 1]
            V_t = V_prev + kappa * (theta - np.maximum(V_prev, 0.0)) * dt + xi * np.sqrt(np.maximum(V_prev, 0.0)) * dWv
            V_t = np.maximum(V_t, 0.0)
            V2[:, t] = V_t
            S2[:, t] = S2[:, t - 1] * np.exp((r - 0.5 * V_t) * dt + np.sqrt(np.maximum(V_t, 0.0)) * dWs)
        S_full = np.vstack([S, S2])
        V_full = np.vstack([V, V2])
        return S_full[:n_paths, :], V_full[:n_paths, :]
    else:
        return S, V


def simulate_paths(model: str, *args, **kwargs):
    model = model.lower()
    if model in ('gbm', 'geometric_brownian_motion'):
        return simulate_gbm_paths(*args, **kwargs)
    if model in ('heston', 'heston_stochastic_volatility'):
        return simulate_heston_paths(*args, **kwargs)
    if model in ('merton', 'merton_jump_diffusion'):
        return simulate_merton_paths(*args, **kwargs)
    if model in ('kou', 'kou_double_exponential'):
        return simulate_kou_paths(*args, **kwargs)
    if model in ('g2pp', 'g2++'):
        return simulate_g2pp_paths(*args, **kwargs)
    raise NotImplementedError(f'Model {model} not implemented')


def simulate_merton_paths(S0: float, r: float, sigma: float, T: float, steps: int, n_paths: int,
                          lambda_jump: float = 0.1, mu_jump: float = 0.0, sigma_jump: float = 0.3,
                          seed: Optional[int] = None, antithetic: bool = False) -> np.ndarray:
    """Simulate Merton Jump-Diffusion paths.

    dS/S = (r - lambda*k) dt + sigma dW + (J-1) dN
    where J = exp(mu_jump + sigma_jump * Z), N is Poisson(lambda*dt)
    """
    if steps <= 0 or n_paths <= 0:
        raise ValueError('steps and n_paths must be > 0')

    rng = np.random.default_rng(seed)
    dt = T / steps
    # adjust r for jump
    k = np.exp(mu_jump + 0.5 * sigma_jump ** 2) - 1.0
    drift = (r - lambda_jump * k) * dt
    diffusion = sigma * np.sqrt(dt)

    S = np.zeros((n_paths, steps + 1), dtype=float)
    S[:, 0] = S0

    for t in range(1, steps + 1):
        # diffusion part
        Z = rng.standard_normal(n_paths)
        dS_diff = drift + diffusion * Z
        # jump part
        N = rng.poisson(lambda_jump * dt, n_paths)
        J = np.ones(n_paths)
        for i in range(n_paths):
            for _ in range(N[i]):
                Z_j = rng.standard_normal()
                J[i] *= np.exp(mu_jump + sigma_jump * Z_j)
        dS_jump = np.log(J)
        S[:, t] = S[:, t-1] * np.exp(dS_diff + dS_jump)

    return S


def simulate_kou_paths(S0: float, r: float, sigma: float, T: float, steps: int, n_paths: int,
                       lambda_jump: float = 0.1, p_up: float = 0.5, eta_up: float = 1.0, eta_down: float = 2.0,
                       seed: Optional[int] = None, antithetic: bool = False) -> np.ndarray:
    """Simulate Kou Double Exponential Jump-Diffusion paths.

    Jump size: Y = exp(U) where U ~ Exponential(eta_up) w.p. p_up, else -Exponential(eta_down)
    """
    if steps <= 0 or n_paths <= 0:
        raise ValueError('steps and n_paths must be > 0')

    rng = np.random.default_rng(seed)
    dt = T / steps

    # expected jump size
    k = p_up / eta_up - (1 - p_up) / eta_down
    drift = (r - lambda_jump * k) * dt
    diffusion = sigma * np.sqrt(dt)

    S = np.zeros((n_paths, steps + 1), dtype=float)
    S[:, 0] = S0

    for t in range(1, steps + 1):
        Z = rng.standard_normal(n_paths)
        dS_diff = drift + diffusion * Z

        # jump part
        N = rng.poisson(lambda_jump * dt, n_paths)
        J = np.ones(n_paths)
        for i in range(n_paths):
            for _ in range(N[i]):
                u = rng.uniform(0, 1)
                if u < p_up:
                    U = rng.exponential(1.0 / eta_up)
                else:
                    U = -rng.exponential(1.0 / eta_down)
                J[i] *= np.exp(U)
        dS_jump = np.log(J)
        S[:, t] = S[:, t-1] * np.exp(dS_diff + dS_jump)

    return S


def simulate_g2pp_paths(S0: float, r0: float, a: float, b: float, sigma: float, eta: float, rho: float,
                        T: float, steps: int, n_paths: int, seed: Optional[int] = None) -> np.ndarray:
    """Simulate G2++ (Two-Factor Gaussian) interest rate model paths.

    dr = a * (phi(t) - r) dt + sigma dW1 + eta dW2 (corr = rho)
    Returns bond prices or simulated rates depending on application.
    For simplicity, returns short rates: r(t)
    """
    if steps <= 0 or n_paths <= 0:
        raise ValueError('steps and n_paths must be > 0')

    rng = np.random.default_rng(seed)
    dt = T / steps

    r = np.zeros((n_paths, steps + 1), dtype=float)
    r[:, 0] = r0

    for t in range(1, steps + 1):
        Z1 = rng.standard_normal(n_paths)
        Z2 = rng.standard_normal(n_paths)
        # correlated increments
        dW1 = np.sqrt(dt) * Z1
        dW2 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

        # drift adjustment (simplified: phi(t) assumed 0 for MVP)
        dr = a * (-r[:, t-1]) * dt + sigma * dW1 + eta * dW2
        r[:, t] = r[:, t-1] + dr
        # floor at 0
        r[:, t] = np.maximum(r[:, t], 0.0)

    return r


if __name__ == '__main__':
    # quick smoke tests
    S = simulate_gbm_paths(100.0, 0.01, 0.2, 1.0, 12, 1000, seed=42, antithetic=True)
    print('GBM', S.shape)
    S_h, V_h = simulate_heston_paths(100.0, 0.01, 0.04, 1.5, 0.04, 0.3, -0.7, 1.0, 12, 1000, seed=42)
    print('Heston', S_h.shape, V_h.shape)
    S_m = simulate_merton_paths(100.0, 0.01, 0.2, 1.0, 12, 1000, lambda_jump=0.1, mu_jump=0.0, sigma_jump=0.3, seed=42)
    print('Merton', S_m.shape)
    S_k = simulate_kou_paths(100.0, 0.01, 0.2, 1.0, 12, 1000, lambda_jump=0.1, p_up=0.5, eta_up=1.0, eta_down=2.0, seed=42)
    print('Kou', S_k.shape)
    r_g2pp = simulate_g2pp_paths(0.03, 0.03, 0.1, 0.1, 0.015, 0.025, 0.8, 5.0, 12, 1000, seed=42)
    print('G2++', r_g2pp.shape)
