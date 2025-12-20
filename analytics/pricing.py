"""
Option pricing functions.
"""

import numpy as np
from typing import Optional
from scipy.stats import norm


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def digital_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes digital (cash-or-nothing) call."""
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-r * T) * norm.cdf(d2)


def digital_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes digital (cash-or-nothing) put."""
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-r * T) * norm.cdf(-d2)

