"""
Streamlit-specific payoff parameter schema for visible UI rendering.
"""

# Map option types to their specific UI parameters
# This controls EXACTLY what appears on screen
PAYOFF_UI_SCHEMA = {
    # European (base case - just strike)
    "european_call": {
        "category": "European",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "european_put": {
        "category": "European",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },

    # Digital options with cash payout
    "digital_cash_call": {
        "category": "Digital",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "cash_amount", "label": "Cash Payout Amount", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "digital_cash_put": {
        "category": "Digital",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "cash_amount", "label": "Cash Payout Amount", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "digital_asset_call": {
        "category": "Digital",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "digital_asset_put": {
        "category": "Digital",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },

    # Gap options with trigger and payoff strikes
    "gap_call": {
        "category": "Gap",
        "params": [
            {"key": "trigger_strike", "label": "Trigger Strike", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "payoff_strike", "label": "Payoff Strike", "type": "float", "default": 105.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "gap_put": {
        "category": "Gap",
        "params": [
            {"key": "trigger_strike", "label": "Trigger Strike", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "payoff_strike", "label": "Payoff Strike", "type": "float", "default": 95.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },

    # American options (exercise anytime)
    "american_call": {
        "category": "American",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "american_put": {
        "category": "American",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },

    # Bermudan options
    "bermudan_call": {
        "category": "Bermudan",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "bermudan_put": {
        "category": "Bermudan",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },

    # Asian options
    "asian_arithmetic_call": {
        "category": "Asian",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "averaging_start", "label": "Averaging Start Step", "type": "int", "default": 1, "min": 1, "max": 251, "step": 1},
            {"key": "averaging_end", "label": "Averaging End Step", "type": "int", "default": 252, "min": 1, "max": 252, "step": 1}
        ]
    },
    "asian_arithmetic_put": {
        "category": "Asian",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "averaging_start", "label": "Averaging Start Step", "type": "int", "default": 1, "min": 1, "max": 251, "step": 1},
            {"key": "averaging_end", "label": "Averaging End Step", "type": "int", "default": 252, "min": 1, "max": 252, "step": 1}
        ]
    },
    "asian_geometric_call": {
        "category": "Asian",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "averaging_start", "label": "Averaging Start Step", "type": "int", "default": 1, "min": 1, "max": 251, "step": 1},
            {"key": "averaging_end", "label": "Averaging End Step", "type": "int", "default": 252, "min": 1, "max": 252, "step": 1}
        ]
    },
    "asian_geometric_put": {
        "category": "Asian",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "averaging_start", "label": "Averaging Start Step", "type": "int", "default": 1, "min": 1, "max": 251, "step": 1},
            {"key": "averaging_end", "label": "Averaging End Step", "type": "int", "default": 252, "min": 1, "max": 252, "step": 1}
        ]
    },

    # Barrier options
    "barrier_up_out_call": {
        "category": "Barrier",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "barrier_level", "label": "Barrier Level", "type": "float", "default": 120.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "barrier_up_out_put": {
        "category": "Barrier",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "barrier_level", "label": "Barrier Level", "type": "float", "default": 120.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "barrier_down_out_call": {
        "category": "Barrier",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "barrier_level", "label": "Barrier Level", "type": "float", "default": 80.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "barrier_down_out_put": {
        "category": "Barrier",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "barrier_level", "label": "Barrier Level", "type": "float", "default": 80.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "barrier_up_in_call": {
        "category": "Barrier",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "barrier_level", "label": "Barrier Level", "type": "float", "default": 120.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "barrier_up_in_put": {
        "category": "Barrier",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "barrier_level", "label": "Barrier Level", "type": "float", "default": 120.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "barrier_down_in_call": {
        "category": "Barrier",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "barrier_level", "label": "Barrier Level", "type": "float", "default": 80.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "barrier_down_in_put": {
        "category": "Barrier",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "barrier_level", "label": "Barrier Level", "type": "float", "default": 80.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },

    # Double Knock options
    "double_knock_out_call": {
        "category": "Double Knock",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "lower_barrier", "label": "Lower Barrier", "type": "float", "default": 80.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "upper_barrier", "label": "Upper Barrier", "type": "float", "default": 120.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "double_knock_out_put": {
        "category": "Double Knock",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "lower_barrier", "label": "Lower Barrier", "type": "float", "default": 80.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "upper_barrier", "label": "Upper Barrier", "type": "float", "default": 120.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "double_knock_in_call": {
        "category": "Double Knock",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "lower_barrier", "label": "Lower Barrier", "type": "float", "default": 80.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "upper_barrier", "label": "Upper Barrier", "type": "float", "default": 120.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "double_knock_in_put": {
        "category": "Double Knock",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "lower_barrier", "label": "Lower Barrier", "type": "float", "default": 80.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "upper_barrier", "label": "Upper Barrier", "type": "float", "default": 120.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },

    # Parisian
    "parisian_up_out_call": {
        "category": "Parisian",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "barrier_level", "label": "Barrier Level", "type": "float", "default": 120.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "parisian_down_out_put": {
        "category": "Parisian",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "barrier_level", "label": "Barrier Level", "type": "float", "default": 80.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },

    # Lookback
    "lookback_float_strike_call": {
        "category": "Lookback",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "lookback_float_strike_put": {
        "category": "Lookback",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "lookback_fixed_strike_call": {
        "category": "Lookback",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "maturity_strike", "label": "Strike at Maturity", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "lookback_fixed_strike_put": {
        "category": "Lookback",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "maturity_strike", "label": "Strike at Maturity", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },

    # Chooser
    "chooser": {
        "category": "Chooser",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },

    # Compound
    "compound_call_on_call": {
        "category": "Compound",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "compound_put_on_put": {
        "category": "Compound",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "compound_call_on_put": {
        "category": "Compound",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "compound_put_on_call": {
        "category": "Compound",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },

    # Multi-asset
    "basket_call": {
        "category": "Multi-Asset",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "basket_put": {
        "category": "Multi-Asset",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "best_of_call": {
        "category": "Multi-Asset",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "worst_of_put": {
        "category": "Multi-Asset",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "spread_option": {
        "category": "Multi-Asset",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },

    # Rainbow
    "rainbow_max_call": {
        "category": "Rainbow",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },
    "rainbow_min_put": {
        "category": "Rainbow",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0}
        ]
    },

    # Forward-Start (CRITICAL TEST CASE)
    "forward_start_call": {
        "category": "Forward-Start",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "forward_start_time", "label": "Forward Start Time (years)", "type": "float", "default": 0.25, "min": 0.01, "max": 5.0, "step": 0.01}
        ]
    },
    "forward_start_put": {
        "category": "Forward-Start",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "forward_start_time", "label": "Forward Start Time (years)", "type": "float", "default": 0.25, "min": 0.01, "max": 5.0, "step": 0.01}
        ]
    },

    # Cliquet
    "cliquet": {
        "category": "Path-Dependent",
        "params": [
            {"key": "strike", "label": "Strike Price", "type": "float", "default": 100.0, "min": 0.01, "max": 10000, "step": 1.0},
            {"key": "cap", "label": "Cap on Return", "type": "float", "default": 0.20, "min": 0.01, "max": 1.0, "step": 0.01},
            {"key": "floor", "label": "Floor on Return", "type": "float", "default": -0.10, "min": -1.0, "max": 0.5, "step": 0.01},
            {"key": "reset_frequency", "label": "Reset Frequency (steps)", "type": "int", "default": 52, "min": 1, "max": 252, "step": 1}
        ]
    },

    # Variance Swap
    "variance_swap": {
        "category": "Variance",
        "params": [
            {"key": "strike", "label": "Variance Strike", "type": "float", "default": 0.04, "min": 0.001, "max": 1.0, "step": 0.001}
        ]
    },
}

