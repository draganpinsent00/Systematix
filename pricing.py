# pricing.py

from typing import Dict, Any, Optional
import numpy as np
import math

from simulator import simulate_paths
from payoffs import Payoff
from var_red import apply_control_variate


def discount_factor(r: float, T: float) -> float:
    return math.exp(-r * T)


def price_mc_from_paths(payoff: Payoff, paths: np.ndarray, r: float, T: float, discount: bool = True, conf_level: float = 0.95,
                        return_payoffs: bool = False) -> Dict[str, Any]:
    """Price a payoff given simulated paths.

    Returns dict containing price (undiscounted), std, stderr, ci, n
    """
    # support importance sampling: paths can be (S, weights)
    weights = None
    if isinstance(paths, tuple) or isinstance(paths, list):
        S = paths[0]
        weights = np.asarray(paths[1])
    else:
        S = paths
    n_paths = S.shape[0]
    payoffs = payoff.payoff(S)
    if weights is None:
        mean = float(np.mean(payoffs))
        std = float(np.std(payoffs, ddof=1)) if n_paths > 1 else 0.0
        stderr = std / math.sqrt(n_paths) if n_paths > 0 else float('nan')
    else:
        # weighted statistics
        weights = np.asarray(weights)
        # normalize weights to sum to 1 for mean; but keep raw weights for variance scaling
        wsum = np.sum(weights)
        if wsum == 0:
            mean = float('nan')
            std = float('nan')
            stderr = float('nan')
        else:
            mean = float(np.sum(weights * payoffs) / wsum)
            # weighted variance with unbiased-ish correction
            avg = np.sum(weights * payoffs) / wsum
            var = np.sum(weights * (payoffs - avg) ** 2) / wsum
            # effective sample size
            ess = (wsum ** 2) / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else n_paths
            std = float(np.sqrt(var))
            stderr = float(std / np.sqrt(ess))
    z = 1.96  # approx for 95%
    ci = (mean - z * stderr, mean + z * stderr)
    out = {
        'price': mean,
        'std': std,
        'stderr': stderr,
        'ci': ci,
        'n': n_paths
    }
    if return_payoffs:
        out['payoffs'] = payoffs
    return out


def price_mc(payoff: Payoff, S0: float, r: float, sigma: float, T: float, steps: int, n_paths: int, seed: Optional[int] = None,
             antithetic: bool = False) -> Dict[str, Any]:
    """Simulate GBM paths and price given payoff object.

    Returns discounted price and diagnostics.
    """
    # simulate paths (GBM model)
    paths = simulate_paths('gbm', S0, r, sigma, T, int(steps), int(n_paths), seed=seed, antithetic=antithetic)
    result = price_mc_from_paths(payoff, paths, r, T, discount=False, conf_level=0.95)
    # discount
    dfactor = discount_factor(r, T)
    price_disc = result['price'] * dfactor
    stderr_disc = result['stderr'] * dfactor
    ci_disc = (result['ci'][0] * dfactor, result['ci'][1] * dfactor)
    out = {
        'model': 'GBM',
        'price': price_disc,
        'stderr': stderr_disc,
        'ci': ci_disc,
        'n': result['n'],
        'raw': result,
        'S0': S0,
        'r': r,
        'sigma': sigma,
        'T': T,
        'steps': steps,
        'n_paths': n_paths,
    }
    return out


def bs_price(S, K, r, sigma, T, option='call'):
    """Black-Scholes price for European call/put (continuous compounding)."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option == 'call' else max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option == 'call':
        return S * 0.5 * (1 + math.erf(d1 / math.sqrt(2))) - K * math.exp(-r * T) * 0.5 * (1 + math.erf(d2 / math.sqrt(2)))
    else:
        return K * math.exp(-r * T) * 0.5 * (1 + math.erf(-d2 / math.sqrt(2))) - S * 0.5 * (1 + math.erf(-d1 / math.sqrt(2)))


def price_mc_with_control_variate(payoff: Payoff, S0: float, r: float, sigma: float, T: float, steps: int, n_paths: int,
                                  seed: Optional[int] = None, antithetic: bool = False) -> Dict[str, Any]:
    """Monte Carlo with simple control variate using S_T (final asset price) whose expectation under RN is S0*exp(rT).

    Returns discounted price and diagnostics and statistics on variance reduction.
    """
    paths = simulate_paths('gbm', S0, r, sigma, T, int(steps), int(n_paths), seed=seed, antithetic=antithetic)
    payoffs = payoff.payoff(paths)
    S_T = paths[:, -1]
    # control X = S_T
    EX = S0 * math.exp(r * T)
    cv = apply_control_variate(payoffs, S_T, EX)
    Y_adj = cv['adjusted']
    mean_adj = float(np.mean(Y_adj))
    std_adj = float(cv.get('std_adj', float(np.std(Y_adj, ddof=1)) if len(Y_adj) > 1 else 0.0))
    stderr_adj = std_adj / math.sqrt(len(Y_adj)) if len(Y_adj) > 0 else float('nan')
    z = 1.96
    ci_adj = (mean_adj - z * stderr_adj, mean_adj + z * stderr_adj)
    # discount
    dfactor = discount_factor(r, T)
    out = {
        'model': 'GBM',
        'price': mean_adj * dfactor,
        'stderr': stderr_adj * dfactor,
        'ci': (ci_adj[0] * dfactor, ci_adj[1] * dfactor),
        'n': len(Y_adj),
        'control_coef': cv['c'],
        'raw': {
            'payoffs_mean': float(np.mean(payoffs)),
            'payoffs_std': float(np.std(payoffs, ddof=1)),
            'adjusted_mean': mean_adj,
            'adjusted_std': float(np.std(Y_adj, ddof=1)) if len(Y_adj) > 1 else 0.0,
        }
    }
    return out


def price_mc_importance_sampling(payoff: Payoff, S0: float, r: float, sigma: float, T: float, steps: int, n_paths: int,
                                 theta: float = 0.0, seed: Optional[int] = None, antithetic: bool = False) -> Dict[str, Any]:
    """Importance sampling by exponential tilting (shift of normal increments) for GBM.

    - theta: shift applied to each standard-normal increment (same for each time step).
    Uses simulate_gbm_paths(..., normal_shift=theta, return_sum_z=True) to obtain sum of shifted normals.
    The importance weight per path is w = exp(-theta * sum_z + 0.5 * theta^2 * steps).
    Returns discounted price estimate and diagnostics similar to price_mc.
    """
    # simulate under tilted measure
    from simulator import simulate_gbm_paths
    res = simulate_gbm_paths(S0, r, sigma, T, int(steps), int(n_paths), seed=seed, antithetic=antithetic, dist='normal', dist_params=None, moment_match=False, normal_shift=theta, return_sum_z=True)
    if isinstance(res, tuple):
        paths, sum_z = res
    else:
        paths = res
        # cannot compute weights without sum_z
        raise RuntimeError('simulate_gbm_paths did not return sum_z; ensure return_sum_z=True is supported')

    payoffs = payoff.payoff(paths)
    # weights
    sum_z = np.asarray(sum_z, dtype=float)
    weights = np.exp(-theta * sum_z + 0.5 * (theta ** 2) * int(steps))
    # apply weights
    weighted = payoffs * weights
    mean_w = float(np.sum(weighted) / np.sum(weights)) if np.sum(weights) != 0 else float('nan')
    # compute weighted variance using normalized weights
    w_norm = weights / np.sum(weights) if np.sum(weights) != 0 else np.ones_like(weights) / len(weights)
    mean_payoff = float(np.sum(w_norm * payoffs))
    # effective sample size
    ess = 1.0 / np.sum(w_norm ** 2) if np.sum(w_norm ** 2) > 0 else float('nan')
    # estimate variance of weighted estimator via weighted second moment (approx)
    var_est = float(np.sum(w_norm * (payoffs - mean_payoff) ** 2))
    stderr = math.sqrt(var_est / len(payoffs)) if len(payoffs) > 0 else float('nan')
    dfactor = discount_factor(r, T)
    out = {
        'model': 'GBM-IS',
        'price': mean_payoff * dfactor,
        'stderr': stderr * dfactor,
        'ci': ( (mean_payoff - 1.96 * stderr) * dfactor, (mean_payoff + 1.96 * stderr) * dfactor ),
        'n': len(payoffs),
        'theta': theta,
        'ess': ess,
    }
    return out


def price_american_lsm_from_paths(payoff_func, paths: np.ndarray, K: Optional[float] = None, r: float = 0.0, T: float = 1.0, degree: int = 2) -> Dict[str, Any]:
    """Price an American option using Longstaff-Schwartz on provided paths.

    - payoff_func: function S -> payoff (vectorized)
    - paths: (n_paths, steps+1)
    Returns dict with 'price' (discounted to t=0), 'stderr', 'n'
    """
    n_paths, steps_plus = paths.shape
    steps = steps_plus - 1
    dt = T / steps
    discount = math.exp(-r * dt)

    # initialize: cashflows and exercise times (in time steps)
    cashflows = payoff_func(paths[:, -1])
    exercise_time = np.full(n_paths, steps, dtype=int)

    # Work backwards
    for t in range(steps - 1, 0, -1):
        S_t = paths[:, t]
        payoff_t = payoff_func(S_t)
        itm = payoff_t > 0
        if not np.any(itm):
            # nothing to do
            continue
        # Discount future cashflows to time t
        tau = exercise_time[itm]  # times (in steps) when cashflows occur
        future_CF = cashflows[itm] * np.exp(-r * dt * (tau - t))
        # Basis functions: [1, S, S^2, ...]
        X = np.vstack([S_t[itm] ** p for p in range(degree + 1)]).T
        # Solve regression: X beta = future_CF
        try:
            beta, *_ = np.linalg.lstsq(X, future_CF, rcond=None)
        except Exception:
            continue
        cont = X.dot(beta)
        # Exercise decision
        exercise_now = payoff_t[itm] > cont
        idx_itm = np.where(itm)[0]
        exercise_idx = idx_itm[exercise_now]
        # Update cashflows and exercise times for exercised paths
        cashflows[exercise_idx] = payoff_t[exercise_idx]
        exercise_time[exercise_idx] = t

    # Discount cashflows to time 0
    pv = cashflows * np.exp(-r * dt * exercise_time)
    price = float(np.mean(pv))
    std = float(np.std(pv, ddof=1)) if n_paths > 1 else 0.0
    stderr = std / math.sqrt(n_paths) if n_paths > 0 else float('nan')
    z = 1.96
    ci = (price - z * stderr, price + z * stderr)
    return {'price': price, 'stderr': stderr, 'ci': ci, 'n': n_paths}


def price_american_lsm(payoff_func, S0: float, r: float, sigma: float, T: float, steps: int, n_paths: int, seed: Optional[int] = None,
                       antithetic: bool = False, degree: int = 2) -> Dict[str, Any]:
    paths = simulate_paths('gbm', S0, r, sigma, T, int(steps), int(n_paths), seed=seed, antithetic=antithetic)
    return price_american_lsm_from_paths(payoff_func, paths, K=None, r=r, T=T, degree=degree)


def price_heston(payoff: Payoff, S0: float, r: float, v0: float, kappa: float, theta: float, xi: float, rho: float,
                 T: float, steps: int, n_paths: int, seed: Optional[int] = None, antithetic: bool = False) -> Dict[str, Any]:
    """Price option under Heston model."""
    S, V = simulate_paths('heston', S0, r, v0, kappa, theta, xi, rho, T, int(steps), int(n_paths), seed=seed, antithetic=antithetic)
    result = price_mc_from_paths(payoff, S, r, T, discount=False, return_payoffs=True)
    dfactor = discount_factor(r, T)
    return {
        'model': 'Heston',
        'price': result['price'] * dfactor,
        'stderr': result['stderr'] * dfactor,
        'ci': (result['ci'][0] * dfactor, result['ci'][1] * dfactor),
        'n': result['n'],
        'payoffs': result.get('payoffs'),
    }


def price_merton(payoff: Payoff, S0: float, r: float, sigma: float, T: float, steps: int, n_paths: int,
                 lambda_jump: float = 0.1, mu_jump: float = 0.0, sigma_jump: float = 0.3,
                 seed: Optional[int] = None, antithetic: bool = False) -> Dict[str, Any]:
    """Price option under Merton Jump-Diffusion model."""
    S = simulate_paths('merton', S0, r, sigma, T, int(steps), int(n_paths), lambda_jump=lambda_jump, mu_jump=mu_jump, sigma_jump=sigma_jump, seed=seed, antithetic=antithetic)
    result = price_mc_from_paths(payoff, S, r, T, discount=False, return_payoffs=True)
    dfactor = discount_factor(r, T)
    return {
        'model': 'Merton',
        'price': result['price'] * dfactor,
        'stderr': result['stderr'] * dfactor,
        'ci': (result['ci'][0] * dfactor, result['ci'][1] * dfactor),
        'n': result['n'],
        'payoffs': result.get('payoffs'),
    }


def price_kou(payoff: Payoff, S0: float, r: float, sigma: float, T: float, steps: int, n_paths: int,
              lambda_jump: float = 0.1, p_up: float = 0.5, eta_up: float = 1.0, eta_down: float = 2.0,
              seed: Optional[int] = None, antithetic: bool = False) -> Dict[str, Any]:
    """Price option under Kou Double Exponential Jump model."""
    S = simulate_paths('kou', S0, r, sigma, T, int(steps), int(n_paths), lambda_jump=lambda_jump, p_up=p_up, eta_up=eta_up, eta_down=eta_down, seed=seed, antithetic=antithetic)
    result = price_mc_from_paths(payoff, S, r, T, discount=False, return_payoffs=True)
    dfactor = discount_factor(r, T)
    return {
        'model': 'Kou',
        'price': result['price'] * dfactor,
        'stderr': result['stderr'] * dfactor,
        'ci': (result['ci'][0] * dfactor, result['ci'][1] * dfactor),
        'n': result['n'],
        'payoffs': result.get('payoffs'),
    }


if __name__ == '__main__':
    # quick smoke
    from payoffs import EuropeanCall
    res = price_mc(EuropeanCall(100.0), 100.0, 0.01, 0.2, 1.0, 12, 2000, seed=42, antithetic=True)
    print(res)
