"""
Reusable Streamlit components.
"""

import streamlit as st
import numpy as np
from typing import Any, Dict, Tuple, Optional


def input_market_params() -> Dict[str, float]:
    """Render market parameters inputs."""
    col1, col2 = st.columns(2)

    with col1:
        spot = st.number_input(
            "Spot Price ($)",
            value=100.0,
            min_value=0.01,
            step=1.0,
        )
        rate = st.number_input(
            "Risk-Free Rate (%)",
            value=5.0,
            min_value=-5.0,
            max_value=50.0,
            step=0.1,
        ) / 100.0

    with col2:
        div_yield = st.number_input(
            "Dividend Yield (%)",
            value=0.0,
            min_value=0.0,
            max_value=50.0,
            step=0.1,
        ) / 100.0
        volatility = st.number_input(
            "Initial Volatility (%)",
            value=20.0,
            min_value=0.1,
            max_value=200.0,
            step=0.1,
        ) / 100.0

    time_to_maturity = st.number_input(
        "Time to Maturity (years)",
        value=1.0,
        min_value=0.01,
        max_value=10.0,
        step=0.25,
    )

    return {
        "spot": spot,
        "risk_free_rate": rate,
        "dividend_yield": div_yield,
        "initial_volatility": volatility,
        "time_to_maturity": time_to_maturity,
    }


def input_mc_settings() -> Dict[str, Any]:
    """Render MC simulation settings."""
    col1, col2 = st.columns(2)

    with col1:
        num_sims = st.number_input(
            "Number of Simulations",
            value=10000,
            min_value=100,
            max_value=1000000,
            step=1000,
        )

    with col2:
        num_steps = st.number_input(
            "Number of Time Steps",
            value=252,
            min_value=10,
            max_value=1000,
            step=10,
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        antithetic = st.checkbox("Antithetic Variates", value=True)
    with col2:
        control_var = st.checkbox("Control Variates", value=False)
    with col3:
        importance_sampling = st.checkbox("Importance Sampling", value=False)

    return {
        "num_simulations": num_sims,
        "num_timesteps": num_steps,
        "antithetic_variates": antithetic,
        "control_variates": control_var,
        "importance_sampling": importance_sampling,
    }


def input_rng_settings() -> Dict[str, Any]:
    """Render RNG engine selection."""
    col1, col2 = st.columns(2)

    with col1:
        engine = st.selectbox(
            "RNG Engine",
            ["mersenne", "pcg64", "xorshift", "philox", "middle_square"],
            index=0,
        )
        seed = st.number_input(
            "Seed",
            value=42,
            min_value=0,
            max_value=2**31 - 1,
        )

    with col2:
        use_sobol = st.checkbox("Use Sobol Sequence", value=False)
        sobol_bb = st.checkbox("Brownian Bridge (Sobol)", value=True, disabled=not use_sobol)

    if not use_sobol:
        distribution = st.selectbox(
            "Innovation Distribution",
            ["normal", "student_t"],
            index=0,
        )
        if distribution == "student_t":
            df = st.number_input(
                "Student-t Degrees of Freedom",
                value=5.0,
                min_value=1.0,
                max_value=30.0,
                step=0.5,
            )
        else:
            df = None
    else:
        distribution = "sobol"
        df = None

    return {
        "engine": engine,
        "seed": seed,
        "use_sobol": use_sobol,
        "distribution": distribution,
        "student_t_df": df,
    }


def input_option_params(option_type: str) -> Dict[str, Any]:
    """
    Registry-driven dynamic payoff parameter input.

    Renders only parameters required for the selected option type.
    All parameters are passed through to the pricing engine.
    """
    from config.schemas import OPTION_PAYOFF_PARAMS, PAYOFF_PARAMS

    params = {}

    # Get required parameters for this option type
    required_params = OPTION_PAYOFF_PARAMS.get(option_type, [])

    if not required_params:
        st.warning(f"No parameters defined for option type: {option_type}")
        return params

    # Organize params into columns (2 columns max for UI)
    num_params = len(required_params)
    cols_needed = min(2, max(1, (num_params + 1) // 2))

    if num_params > 0:
        if cols_needed == 1:
            cols = [st.container()]
        else:
            cols = st.columns(cols_needed)

        for idx, param_name in enumerate(required_params):
            col_idx = idx % cols_needed

            if param_name not in PAYOFF_PARAMS:
                st.warning(f"Parameter {param_name} not defined in registry")
                continue

            param_meta = PAYOFF_PARAMS[param_name]
            param_type = param_meta.get("type", "float")
            min_val = param_meta.get("min", 0.01)
            max_val = param_meta.get("max", 10000.0)
            step = param_meta.get("step", 1.0)
            desc = param_meta.get("desc", param_name)

            # Determine default value based on parameter type
            if param_name == "strike":
                default = 100.0
            elif param_name == "barrier_level":
                default = 120.0
            elif param_name in ["lower_barrier"]:
                default = 80.0
            elif param_name in ["upper_barrier"]:
                default = 120.0
            elif param_name == "cash_amount":
                default = 100.0
            elif param_name == "trigger_strike":
                default = 100.0
            elif param_name == "payoff_strike":
                default = 100.0
            elif param_name == "forward_start_time":
                default = 0.25
            elif param_name == "averaging_start":
                default = 1
            elif param_name == "averaging_end":
                default = 252
            elif param_name == "cap":
                default = 0.20
            elif param_name == "floor":
                default = -0.10
            elif param_name == "reset_frequency":
                default = 52
            elif param_name == "maturity_strike":
                default = 100.0
            else:
                default = min_val + (max_val - min_val) * 0.5

            with cols[col_idx]:
                if param_type == "float":
                    params[param_name] = st.number_input(
                        f"{desc}",
                        value=float(default),
                        min_value=float(min_val),
                        max_value=float(max_val),
                        step=float(step),
                        key=f"payoff_{param_name}",
                    )
                elif param_type == "int":
                    params[param_name] = st.number_input(
                        f"{desc}",
                        value=int(default),
                        min_value=int(min_val),
                        max_value=int(max_val),
                        step=int(step),
                        key=f"payoff_{param_name}",
                    )

    return params


def show_error(message: str):
    """Display error message."""
    st.error(f"❌ {message}")


def show_warning(message: str):
    """Display warning message."""
    st.warning(f"⚠️ {message}")


def show_success(message: str):
    """Display success message."""
    st.success(f"✅ {message}")

