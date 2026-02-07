"""
Instrument registry and custom payoff builder.
"""

from typing import Dict, Type, Any, Optional, Callable
import numpy as np
from .base import Instrument
from .payoffs_vanilla import *
from .payoffs_exotic import *
from .payoffs_rates_fx import *


INSTRUMENT_REGISTRY: Dict[str, Type[Instrument]] = {
    # European
    "european_call": EuropeanCall,
    "european_put": EuropeanPut,

    # Digital
    "digital_cash_call": DigitalCashCall,
    "digital_cash_put": DigitalCashPut,
    "digital_asset_call": DigitalAssetCall,
    "digital_asset_put": DigitalAssetPut,

    # Gap
    "gap_call": GapCall,
    "gap_put": GapPut,

    # American
    "american_call": AmericanCall,
    "american_put": AmericanPut,

    # Bermudan
    "bermudan_call": BermudanCall,
    "bermudan_put": BermudanPut,

    # Asian
    "asian_arithmetic_call": AsianArithmeticCall,
    "asian_arithmetic_put": AsianArithmeticPut,
    "asian_geometric_call": AsianGeometricCall,
    "asian_geometric_put": AsianGeometricPut,

    # Barrier
    "barrier_up_out_call": BarrierUpOutCall,
    "barrier_up_out_put": BarrierUpOutPut,
    "barrier_down_out_call": BarrierDownOutCall,
    "barrier_down_out_put": BarrierDownOutPut,
    "barrier_up_in_call": BarrierUpInCall,
    "barrier_up_in_put": BarrierUpInPut,
    "barrier_down_in_call": BarrierDownInCall,
    "barrier_down_in_put": BarrierDownInPut,

    # Double Knock
    "double_knock_out_call": DoubleKnockOutCall,
    "double_knock_out_put": DoubleKnockOutPut,
    "double_knock_in_call": DoubleKnockInCall,
    "double_knock_in_put": DoubleKnockInPut,

    # Parisian
    "parisian_up_out_call": ParisianUpOutCall,
    "parisian_down_out_put": ParisianDownOutPut,

    # Lookback
    "lookback_float_strike_call": LookbackFloatStrikeCall,
    "lookback_float_strike_put": LookbackFloatStrikePut,
    "lookback_fixed_strike_call": LookbackFixedStrikeCall,
    "lookback_fixed_strike_put": LookbackFixedStrikePut,

    # Chooser
    "chooser": ChooserOption,

    # Compound
    "compound_call_on_call": CompoundCallOnCall,
    "compound_put_on_put": CompoundPutOnPut,
    "compound_call_on_put": CompoundCallOnPut,
    "compound_put_on_call": CompoundPutOnCall,

    # Multi-asset
    "basket_call": BasketCall,
    "basket_put": BasketPut,
    "best_of_call": BestOfCall,
    "worst_of_put": WorstOfPut,
    "spread_option": SpreadOption,

    # Rainbow
    "rainbow_max_call": RainbowMaxCall,
    "rainbow_min_put": RainbowMinPut,

    # Forward-start
    "forward_start_call": ForwardStartCall,
    "forward_start_put": ForwardStartPut,

    # Path-dependent
    "cliquet": Cliquet,
    "variance_swap": VarianceSwap,
}


class CustomPayoff(Instrument):
    """User-defined custom payoff."""

    def __init__(self, payoff_func: Callable, strike: float = 100.0, **params):
        """
        Initialize custom payoff.

        Args:
            payoff_func: Function that takes paths and returns payoffs
            strike: Strike price (optional context)
        """
        super().__init__(strike, **params)
        self.payoff_func = payoff_func

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        return self.payoff_func(paths)

    def get_params(self) -> Dict[str, Any]:
        return {"strike": self.strike, "custom": True}


def create_instrument(
    option_type: str,
    strike: float = 100.0,
    **kwargs
) -> Instrument:
    """
    Factory function to create instrument.

    Args:
        option_type: Key from INSTRUMENT_REGISTRY
        strike: Strike price
        **kwargs: Option-specific parameters

    Returns:
        Instrument instance
    """
    option_type_lower = option_type.lower()

    if option_type_lower not in INSTRUMENT_REGISTRY:
        raise ValueError(f"Unknown option type: {option_type}. Available: {list(INSTRUMENT_REGISTRY.keys())}")

    # Map UI parameter names to instrument parameter names
    param_map = {
        # Digital options
        "cash_amount": "payout",
        # Gap options: UI uses trigger_strike + payoff_strike
        # Instrument uses trigger (condition) + strike (payoff)
        "trigger_strike": "trigger",
        # Barrier options
        "barrier_level": "barrier",
        "lower_barrier": "lower_barrier",
        "upper_barrier": "upper_barrier",
        # Lookback
        "maturity_strike": "maturity_strike",
        # Forward-start
        "forward_start_time": "t_start",
    }

    # Rename parameters according to mapping
    mapped_kwargs = {}

    for key, value in kwargs.items():
        if key == "payoff_strike":
            # For gap options, payoff_strike becomes the strike parameter
            strike = value
        else:
            mapped_key = param_map.get(key, key)
            mapped_kwargs[mapped_key] = value

    instrument_class = INSTRUMENT_REGISTRY[option_type_lower]
    return instrument_class(strike=strike, **mapped_kwargs)

