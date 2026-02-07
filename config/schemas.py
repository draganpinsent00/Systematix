"""
Input and output schemas with UI metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class RNGEngine(str, Enum):
    MERSENNE = "mersenne"
    MIDDLE_SQUARE = "middle_square"
    PCG64 = "pcg64"
    XORSHIFT = "xorshift"
    PHILOX = "philox"


class Distribution(str, Enum):
    NORMAL = "normal"
    STUDENT_T = "student_t"


@dataclass
class RNGSettings:
    """RNG configuration."""
    engine: RNGEngine = RNGEngine.MERSENNE
    seed: int = 42
    use_sobol: bool = False
    distribution: Distribution = Distribution.NORMAL
    student_t_df: float = 3.0
    antithetic_variates: bool = True

    @property
    def ui_groups(self) -> Dict[str, List[str]]:
        """Return UI grouping for dynamic form generation."""
        return {
            "RNG Engine & Seed": ["engine", "seed"],
            "Quasi-Random (Sobol)": ["use_sobol"],
            "Distribution": ["distribution", "student_t_df"] if not self.use_sobol else [],
            "Variance Reduction": ["antithetic_variates"],
        }


@dataclass
class MarketParams:
    """Market parameters."""
    spot: float = 100.0
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    initial_volatility: float = 0.2
    time_to_maturity: float = 1.0

    @property
    def ui_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {
            "spot": {"min": 0.01, "max": 10000.0, "step": 1.0, "unit": "$"},
            "risk_free_rate": {"min": -0.05, "max": 0.5, "step": 0.01, "unit": "%"},
            "dividend_yield": {"min": 0.0, "max": 0.5, "step": 0.01, "unit": "%"},
            "initial_volatility": {"min": 0.01, "max": 2.0, "step": 0.01, "unit": "σ"},
            "time_to_maturity": {"min": 0.01, "max": 10.0, "step": 0.25, "unit": "years"},
        }


@dataclass
class MonteCarloSettings:
    """MC execution parameters."""
    num_simulations: int = 10000
    num_timesteps: int = 252
    use_lsm_american: bool = False
    lsm_degree: int = 2


@dataclass
class OptionParams:
    """Base option parameters. Extended by specific types."""
    strike: float = 100.0
    is_call: bool = True
    # Barrier params (for barrier options)
    barrier_level: Optional[float] = None
    barrier_type: str = "knock_out"  # knock_out, knock_in
    barrier_direction: str = "up"  # up, down
    # Averaging params
    averaging_start_date: int = 1
    averaging_frequency: str = "daily"  # daily, monthly, etc.
    # Jump params (populated if model uses jumps)
    # Lookup params
    fix_strike_lookback: Optional[float] = None  # for fixed-strike

    @property
    def ui_metadata(self) -> Dict[str, Dict[str, Any]]:
        return {
            "strike": {"min": 0.01, "max": 10000.0, "step": 1.0, "unit": "$"},
        }


@dataclass
class ModelConfig:
    """Model configuration with registry metadata."""
    name: str
    required_params: List[str]
    param_metadata: Dict[str, Dict[str, Any]]
    supports_jumps: bool = False
    supports_correlation: bool = False
    supports_stoch_vol: bool = False


MODEL_REGISTRY = {
    "gbm": ModelConfig(
        name="Geometric Brownian Motion",
        required_params=["spot", "risk_free_rate", "dividend_yield", "initial_volatility", "time_to_maturity"],
        param_metadata={},
        supports_correlation=True,
    ),
    "heston": ModelConfig(
        name="Heston",
        required_params=["spot", "risk_free_rate", "dividend_yield", "initial_volatility", "time_to_maturity",
                        "heston_kappa", "heston_theta", "heston_sigma", "heston_rho"],
        param_metadata={
            "heston_kappa": {"min": 0.01, "max": 10.0, "step": 0.1, "desc": "mean reversion speed"},
            "heston_theta": {"min": 0.01, "max": 2.0, "step": 0.01, "desc": "long-run vol"},
            "heston_sigma": {"min": 0.01, "max": 2.0, "step": 0.01, "desc": "vol of vol"},
            "heston_rho": {"min": -0.99, "max": 0.99, "step": 0.01, "desc": "spot-vol correlation"},
        },
        supports_stoch_vol=True,
        supports_correlation=True,
    ),
    "heston_3_2": ModelConfig(
        name="3/2 Heston",
        required_params=["spot", "risk_free_rate", "dividend_yield", "initial_volatility", "time_to_maturity",
                        "heston_32_kappa", "heston_32_theta", "heston_32_sigma", "heston_32_rho"],
        param_metadata={
            "heston_32_kappa": {"min": 0.01, "max": 10.0, "step": 0.1, "desc": "mean reversion speed"},
            "heston_32_theta": {"min": 0.01, "max": 2.0, "step": 0.01, "desc": "long-run vol"},
            "heston_32_sigma": {"min": 0.01, "max": 2.0, "step": 0.01, "desc": "vol of vol"},
            "heston_32_rho": {"min": -0.99, "max": 0.99, "step": 0.01, "desc": "spot-vol correlation"},
        },
        supports_stoch_vol=True,
        supports_correlation=True,
    ),
    "merton_jump": ModelConfig(
        name="Merton Jump Diffusion",
        required_params=["spot", "risk_free_rate", "dividend_yield", "initial_volatility", "time_to_maturity",
                        "merton_lambda", "merton_mu_j", "merton_sigma_j"],
        param_metadata={
            "merton_lambda": {"min": 0.0, "max": 5.0, "step": 0.1, "desc": "jump intensity"},
            "merton_mu_j": {"min": -0.5, "max": 0.5, "step": 0.01, "desc": "log-jump mean"},
            "merton_sigma_j": {"min": 0.01, "max": 1.0, "step": 0.01, "desc": "log-jump std"},
        },
        supports_jumps=True,
        supports_correlation=True,
    ),
    "kou_jump": ModelConfig(
        name="Kou Double Exponential Jump",
        required_params=["spot", "risk_free_rate", "dividend_yield", "initial_volatility", "time_to_maturity",
                        "kou_lambda", "kou_p_up", "kou_eta_up", "kou_eta_down"],
        param_metadata={
            "kou_lambda": {"min": 0.0, "max": 5.0, "step": 0.1, "desc": "jump intensity"},
            "kou_p_up": {"min": 0.0, "max": 1.0, "step": 0.01, "desc": "prob of up jump"},
            "kou_eta_up": {"min": 0.1, "max": 50.0, "step": 0.1, "desc": "up-jump decay rate"},
            "kou_eta_down": {"min": 0.1, "max": 50.0, "step": 0.1, "desc": "down-jump decay rate"},
        },
        supports_jumps=True,
        supports_correlation=True,
    ),
    "sabr": ModelConfig(
        name="SABR",
        required_params=["spot", "risk_free_rate", "dividend_yield", "initial_volatility", "time_to_maturity",
                        "sabr_alpha", "sabr_beta", "sabr_nu", "sabr_rho"],
        param_metadata={
            "sabr_alpha": {"min": 0.01, "max": 1.0, "step": 0.01, "desc": "ATM volatility"},
            "sabr_beta": {"min": 0.0, "max": 1.0, "step": 0.1, "desc": "elasticity"},
            "sabr_nu": {"min": 0.0, "max": 2.0, "step": 0.1, "desc": "vol of vol"},
            "sabr_rho": {"min": -0.99, "max": 0.99, "step": 0.01, "desc": "spot-vol correlation"},
        },
        supports_stoch_vol=True,
        supports_correlation=True,
    ),
    "bachelier": ModelConfig(
        name="Bachelier (ABM)",
        required_params=["spot", "risk_free_rate", "dividend_yield", "initial_volatility", "time_to_maturity"],
        param_metadata={},
        supports_correlation=True,
    ),
    "local_volatility": ModelConfig(
        name="Local Volatility (Flat)",
        required_params=["spot", "risk_free_rate", "dividend_yield", "initial_volatility", "time_to_maturity"],
        param_metadata={},
        supports_correlation=True,
    ),
    "regime_switching": ModelConfig(
        name="Regime-Switching GBM",
        required_params=["spot", "risk_free_rate", "dividend_yield", "initial_volatility", "time_to_maturity",
                        "rs_sigma_low", "rs_sigma_high", "rs_p_ll", "rs_p_hh"],
        param_metadata={
            "rs_sigma_low": {"min": 0.01, "max": 1.0, "step": 0.01, "desc": "low-vol regime volatility"},
            "rs_sigma_high": {"min": 0.01, "max": 2.0, "step": 0.01, "desc": "high-vol regime volatility"},
            "rs_p_ll": {"min": 0.0, "max": 1.0, "step": 0.01, "desc": "stay in low regime prob"},
            "rs_p_hh": {"min": 0.0, "max": 1.0, "step": 0.01, "desc": "stay in high regime prob"},
        },
        supports_correlation=True,
    ),
    "shifted_lognormal": ModelConfig(
        name="Shifted Lognormal",
        required_params=["spot", "risk_free_rate", "dividend_yield", "initial_volatility", "time_to_maturity",
                        "sl_shift"],
        param_metadata={
            "sl_shift": {"min": -10000.0, "max": 10000.0, "step": 1.0, "desc": "shift parameter"},
        },
        supports_correlation=True,
    ),
}

OPTION_TYPES = {
    # European
    "european_call": {"name": "European Call", "category": "European"},
    "european_put": {"name": "European Put", "category": "European"},

    # Digital
    "digital_cash_call": {"name": "Cash-or-Nothing Digital Call", "category": "Digital"},
    "digital_cash_put": {"name": "Cash-or-Nothing Digital Put", "category": "Digital"},
    "digital_asset_call": {"name": "Asset-or-Nothing Digital Call", "category": "Digital"},
    "digital_asset_put": {"name": "Asset-or-Nothing Digital Put", "category": "Digital"},

    # Gap
    "gap_call": {"name": "Gap Call", "category": "Gap"},
    "gap_put": {"name": "Gap Put", "category": "Gap"},

    # American
    "american_call": {"name": "American Call", "category": "American"},
    "american_put": {"name": "American Put", "category": "American"},

    # Bermudan
    "bermudan_call": {"name": "Bermudan Call", "category": "Bermudan"},
    "bermudan_put": {"name": "Bermudan Put", "category": "Bermudan"},

    # Asian
    "asian_arithmetic_call": {"name": "Asian Call (Arithmetic)", "category": "Asian"},
    "asian_arithmetic_put": {"name": "Asian Put (Arithmetic)", "category": "Asian"},
    "asian_geometric_call": {"name": "Asian Call (Geometric)", "category": "Asian"},
    "asian_geometric_put": {"name": "Asian Put (Geometric)", "category": "Asian"},

    # Barrier - Knock-Out
    "barrier_up_out_call": {"name": "Up-and-Out Call", "category": "Barrier KO"},
    "barrier_up_out_put": {"name": "Up-and-Out Put", "category": "Barrier KO"},
    "barrier_down_out_call": {"name": "Down-and-Out Call", "category": "Barrier KO"},
    "barrier_down_out_put": {"name": "Down-and-Out Put", "category": "Barrier KO"},

    # Barrier - Knock-In
    "barrier_up_in_call": {"name": "Up-and-In Call", "category": "Barrier KI"},
    "barrier_up_in_put": {"name": "Up-and-In Put", "category": "Barrier KI"},
    "barrier_down_in_call": {"name": "Down-and-In Call", "category": "Barrier KI"},
    "barrier_down_in_put": {"name": "Down-and-In Put", "category": "Barrier KI"},

    # Double Knock
    "double_knock_out_call": {"name": "Double Knock-Out Call", "category": "Double Knock"},
    "double_knock_out_put": {"name": "Double Knock-Out Put", "category": "Double Knock"},
    "double_knock_in_call": {"name": "Double Knock-In Call", "category": "Double Knock"},
    "double_knock_in_put": {"name": "Double Knock-In Put", "category": "Double Knock"},

    # Parisian
    "parisian_up_out_call": {"name": "Parisian Up-and-Out Call", "category": "Parisian"},
    "parisian_down_out_put": {"name": "Parisian Down-and-Out Put", "category": "Parisian"},

    # Lookback
    "lookback_float_strike_call": {"name": "Floating-Strike Lookback Call", "category": "Lookback"},
    "lookback_float_strike_put": {"name": "Floating-Strike Lookback Put", "category": "Lookback"},
    "lookback_fixed_strike_call": {"name": "Fixed-Strike Lookback Call", "category": "Lookback"},
    "lookback_fixed_strike_put": {"name": "Fixed-Strike Lookback Put", "category": "Lookback"},

    # Chooser
    "chooser": {"name": "Chooser Option", "category": "Chooser"},

    # Compound
    "compound_call_on_call": {"name": "Compound Call-on-Call", "category": "Compound"},
    "compound_put_on_put": {"name": "Compound Put-on-Put", "category": "Compound"},
    "compound_call_on_put": {"name": "Compound Call-on-Put", "category": "Compound"},
    "compound_put_on_call": {"name": "Compound Put-on-Call", "category": "Compound"},

    # Multi-Asset
    "basket_call": {"name": "Basket Call (Weighted)", "category": "Multi-Asset"},
    "basket_put": {"name": "Basket Put (Weighted)", "category": "Multi-Asset"},
    "best_of_call": {"name": "Best-of Call", "category": "Multi-Asset"},
    "worst_of_put": {"name": "Worst-of Put", "category": "Multi-Asset"},
    "spread_option": {"name": "Spread Option", "category": "Multi-Asset"},

    # Rainbow
    "rainbow_max_call": {"name": "Rainbow Max Call", "category": "Rainbow"},
    "rainbow_min_put": {"name": "Rainbow Min Put", "category": "Rainbow"},

    # Forward-Start
    "forward_start_call": {"name": "Forward-Start Call", "category": "Forward-Start"},
    "forward_start_put": {"name": "Forward-Start Put", "category": "Forward-Start"},

    # Path-Dependent
    "cliquet": {"name": "Cliquet Option", "category": "Path-Dependent"},
    "variance_swap": {"name": "Variance Swap", "category": "Variance"},
}


# Payoff Parameter Registry
PAYOFF_PARAMS = {
    # Basic params for all options
    "strike": {"type": "float", "min": 0.01, "max": 10000.0, "step": 1.0, "desc": "Strike price"},

    # Barrier options
    "barrier_level": {"type": "float", "min": 0.01, "max": 10000.0, "step": 1.0, "desc": "Barrier level"},
    "lower_barrier": {"type": "float", "min": 0.01, "max": 10000.0, "step": 1.0, "desc": "Lower barrier"},
    "upper_barrier": {"type": "float", "min": 0.01, "max": 10000.0, "step": 1.0, "desc": "Upper barrier"},

    # Digital options
    "cash_amount": {"type": "float", "min": 0.01, "max": 10000.0, "step": 1.0, "desc": "Cash payout amount"},

    # Gap options
    "trigger_strike": {"type": "float", "min": 0.01, "max": 10000.0, "step": 1.0, "desc": "Trigger strike price"},
    "payoff_strike": {"type": "float", "min": 0.01, "max": 10000.0, "step": 1.0, "desc": "Payoff strike price"},

    # Forward-start options
    "forward_start_time": {"type": "float", "min": 0.01, "max": 10.0, "step": 0.01, "desc": "Forward start time (years)"},

    # Asian options
    "averaging_start": {"type": "int", "min": 1, "max": 252, "step": 1, "desc": "Averaging start step"},
    "averaging_end": {"type": "int", "min": 1, "max": 252, "step": 1, "desc": "Averaging end step"},

    # Cliquet options
    "cap": {"type": "float", "min": 0.01, "max": 1.0, "step": 0.01, "desc": "Cap on return"},
    "floor": {"type": "float", "min": -1.0, "max": 0.5, "step": 0.01, "desc": "Floor on return"},
    "reset_frequency": {"type": "int", "min": 1, "max": 252, "step": 1, "desc": "Reset frequency (steps)"},

    # Lookback options
    "maturity_strike": {"type": "float", "min": 0.01, "max": 10000.0, "step": 1.0, "desc": "Strike at maturity"},
}


# Option type to required parameters mapping
OPTION_PAYOFF_PARAMS = {
    # Basic european
    "european_call": ["strike"],
    "european_put": ["strike"],

    # Digital
    "digital_cash_call": ["strike", "cash_amount"],
    "digital_cash_put": ["strike", "cash_amount"],
    "digital_asset_call": ["strike"],
    "digital_asset_put": ["strike"],

    # Gap
    "gap_call": ["trigger_strike", "payoff_strike"],
    "gap_put": ["trigger_strike", "payoff_strike"],

    # American
    "american_call": ["strike"],
    "american_put": ["strike"],

    # Bermudan
    "bermudan_call": ["strike"],
    "bermudan_put": ["strike"],

    # Asian
    "asian_arithmetic_call": ["strike", "averaging_start", "averaging_end"],
    "asian_arithmetic_put": ["strike", "averaging_start", "averaging_end"],
    "asian_geometric_call": ["strike", "averaging_start", "averaging_end"],
    "asian_geometric_put": ["strike", "averaging_start", "averaging_end"],

    # Barrier
    "barrier_up_out_call": ["strike", "barrier_level"],
    "barrier_up_out_put": ["strike", "barrier_level"],
    "barrier_down_out_call": ["strike", "barrier_level"],
    "barrier_down_out_put": ["strike", "barrier_level"],
    "barrier_up_in_call": ["strike", "barrier_level"],
    "barrier_up_in_put": ["strike", "barrier_level"],
    "barrier_down_in_call": ["strike", "barrier_level"],
    "barrier_down_in_put": ["strike", "barrier_level"],

    # Double Knock
    "double_knock_out_call": ["strike", "lower_barrier", "upper_barrier"],
    "double_knock_out_put": ["strike", "lower_barrier", "upper_barrier"],
    "double_knock_in_call": ["strike", "lower_barrier", "upper_barrier"],
    "double_knock_in_put": ["strike", "lower_barrier", "upper_barrier"],

    # Parisian
    "parisian_up_out_call": ["strike", "barrier_level"],
    "parisian_down_out_put": ["strike", "barrier_level"],

    # Lookback
    "lookback_float_strike_call": ["strike"],
    "lookback_float_strike_put": ["strike"],
    "lookback_fixed_strike_call": ["strike", "maturity_strike"],
    "lookback_fixed_strike_put": ["strike", "maturity_strike"],

    # Chooser
    "chooser": ["strike"],

    # Compound
    "compound_call_on_call": ["strike"],
    "compound_put_on_put": ["strike"],
    "compound_call_on_put": ["strike"],
    "compound_put_on_call": ["strike"],

    # Multi-asset
    "basket_call": ["strike"],
    "basket_put": ["strike"],
    "best_of_call": ["strike"],
    "worst_of_put": ["strike"],
    "spread_option": ["strike"],

    # Rainbow
    "rainbow_max_call": ["strike"],
    "rainbow_min_put": ["strike"],

    # Forward-start
    "forward_start_call": ["strike", "forward_start_time"],
    "forward_start_put": ["strike", "forward_start_time"],

    # Path-dependent
    "cliquet": ["strike", "cap", "floor", "reset_frequency"],
    "variance_swap": ["strike"],
}
