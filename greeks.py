from typing import Callable, Dict, Any, Optional
import numpy as np
try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except Exception:
    _HAS_JAX = False

from simulator import simulate_paths
from pricing import price_mc_from_paths


def compute_greeks_mc(payoff_obj, model_key: str, S0: float, r: float, sigma: float, T: float, steps: int, n_paths: int,
                      seed: Optional[int] = None, antithetic: bool = False, h_S: Optional[float] = None, h_sigma: float = 1e-4,
                      h_r: float = 1e-4, h_T: float = 1e-4) -> Dict[str, Any]:
    """Compute Greeks by Monte Carlo with Common Random Numbers (CRN) where possible.

    Methods:
    - Delta/Gamma: use multiplicative scaling of S_paths (works for GBM) to reuse same paths -> low variance
    - Vega/Rho/Theta: resimulate with same RNG seed for +/- bumps to use CRN

    Returns dict with keys: price, delta, gamma, vega, rho, theta, and stderr estimates.
    """
    # sensible default bumps
    h_S = h_S if h_S is not None else max(1e-3 * S0, 1e-6)

    # simulate base paths
    base_seed = None if seed is None or int(seed) == 0 else int(seed)
    S_base = simulate_paths(model_key, S0, r, sigma, T, int(steps), int(n_paths), seed=base_seed, antithetic=antithetic)
    # price base
    res_base = price_mc_from_paths(payoff_obj, S_base, r, T, discount=False, return_payoffs=True)
    price = res_base['price']
    payoffs_base = res_base.get('payoffs')

    out = {'price': float(price)}

    # Delta & Gamma via scaling (works for GBM-like multiplicative processes)
    scale_up = (S0 + h_S) / S0
    scale_dn = (S0 - h_S) / S0
    S_up = S_base * scale_up
    S_dn = S_base * scale_dn
    res_up = price_mc_from_paths(payoff_obj, S_up, r, T, discount=False, return_payoffs=True)
    res_dn = price_mc_from_paths(payoff_obj, S_dn, r, T, discount=False, return_payoffs=True)
    C_up = res_up['price']; C_dn = res_dn['price']
    delta = (C_up - C_dn) / (2 * h_S)
    gamma = (C_up - 2 * price + C_dn) / (h_S ** 2)
    # estimate stderr via paired differences
    pay_up = res_up['payoffs']; pay_dn = res_dn['payoffs']
    # delta per-path estimator: (pay_up - pay_dn)/(2*h_S)
    delta_per_path = (np.asarray(pay_up) - np.asarray(pay_dn)) / (2 * h_S)
    stderr_delta = float(np.std(delta_per_path, ddof=1) / np.sqrt(len(delta_per_path)))
    out.update({'delta': float(delta), 'gamma': float(gamma), 'stderr_delta': stderr_delta})

    # Vega (bump sigma) - use CRN by resimulating with same seeds
    sigma_up = sigma + h_sigma
    sigma_dn = max(1e-12, sigma - h_sigma)
    S_sig_up = simulate_paths(model_key, S0, r, sigma_up, T, int(steps), int(n_paths), seed=base_seed, antithetic=antithetic)
    S_sig_dn = simulate_paths(model_key, S0, r, sigma_dn, T, int(steps), int(n_paths), seed=base_seed, antithetic=antithetic)
    res_sig_up = price_mc_from_paths(payoff_obj, S_sig_up, r, T, discount=False, return_payoffs=True)
    res_sig_dn = price_mc_from_paths(payoff_obj, S_sig_dn, r, T, discount=False, return_payoffs=True)
    vega = (res_sig_up['price'] - res_sig_dn['price']) / (2 * h_sigma)
    vega_per = (np.asarray(res_sig_up['payoffs']) - np.asarray(res_sig_dn['payoffs'])) / (2 * h_sigma)
    stderr_vega = float(np.std(vega_per, ddof=1) / np.sqrt(len(vega_per)))
    out.update({'vega': float(vega), 'stderr_vega': stderr_vega})

    # Rho (bump r)
    r_up = r + h_r
    r_dn = r - h_r
    S_r_up = simulate_paths(model_key, S0, r_up, sigma, T, int(steps), int(n_paths), seed=base_seed, antithetic=antithetic)
    S_r_dn = simulate_paths(model_key, S0, r_dn, sigma, T, int(steps), int(n_paths), seed=base_seed, antithetic=antithetic)
    res_r_up = price_mc_from_paths(payoff_obj, S_r_up, r_up, T, discount=False, return_payoffs=True)
    res_r_dn = price_mc_from_paths(payoff_obj, S_r_dn, r_dn, T, discount=False, return_payoffs=True)
    rho = (res_r_up['price'] - res_r_dn['price']) / (2 * h_r)
    rho_per = (np.asarray(res_r_up['payoffs']) - np.asarray(res_r_dn['payoffs'])) / (2 * h_r)
    stderr_rho = float(np.std(rho_per, ddof=1) / np.sqrt(len(rho_per)))
    out.update({'rho': float(rho), 'stderr_rho': stderr_rho})

    # Theta (bump T) - forward or central difference depending on T
    T_up = T + h_T
    T_dn = max(1e-8, T - h_T)
    S_T_up = simulate_paths(model_key, S0, r, sigma, T_up, int(steps), int(n_paths), seed=base_seed, antithetic=antithetic)
    S_T_dn = simulate_paths(model_key, S0, r, sigma, T_dn, int(steps), int(n_paths), seed=base_seed, antithetic=antithetic)
    res_T_up = price_mc_from_paths(payoff_obj, S_T_up, r, T_up, discount=False, return_payoffs=True)
    res_T_dn = price_mc_from_paths(payoff_obj, S_T_dn, r, T_dn, discount=False, return_payoffs=True)
    theta = (res_T_up['price'] - res_T_dn['price']) / (T_up - T_dn)
    theta_per = (np.asarray(res_T_up['payoffs']) - np.asarray(res_T_dn['payoffs'])) / (T_up - T_dn)
    stderr_theta = float(np.std(theta_per, ddof=1) / np.sqrt(len(theta_per)))
    out.update({'theta': float(theta), 'stderr_theta': stderr_theta})

    return out


def compute_greeks_fd(price_fn: Callable[[float], float], S0: float, eps: float = 1e-4) -> Dict[str, float]:
    """Compute delta and gamma using centered finite differences on price_fn(S).

    price_fn: function taking spot S and returning option price (scalar)
    S0: spot
    eps: relative bump fraction
    Returns: dict with 'delta' and 'gamma'
    """
    # use relative bump
    h = max(eps * S0, 1e-8)
    p_up = price_fn(S0 + h)
    p_dn = price_fn(S0 - h)
    p0 = price_fn(S0)
    delta = (p_up - p_dn) / (2 * h)
    gamma = (p_up - 2 * p0 + p_dn) / (h ** 2)
    return {'delta': float(delta), 'gamma': float(gamma)}


if __name__ == '__main__':
    print('Greeks module loaded; JAX available:', _HAS_JAX)
