"""
Dynamic form generation from schema.
"""

import streamlit as st
from typing import Dict, Any, Type
from config.schemas import MODEL_REGISTRY, OPTION_TYPES, RNGEngine, Distribution


def render_model_params_dynamic(model_name: str) -> Dict[str, Any]:
    """
    Render model parameters dynamically based on registry.

    Args:
        model_name: Model key

    Returns:
        Dict of parameter values
    """
    if model_name not in MODEL_REGISTRY:
        st.error(f"Unknown model: {model_name}")
        return {}

    config = MODEL_REGISTRY[model_name]
    params = {}

    st.subheader(f"⚙️ {config.name} Parameters")

    for param_name in config.required_params:
        if param_name in ["spot", "risk_free_rate", "dividend_yield", "initial_volatility", "time_to_maturity"]:
            continue  # Skip market params

        metadata = config.param_metadata.get(param_name, {})
        min_val = metadata.get("min", 0.0)
        max_val = metadata.get("max", 100.0)
        step = metadata.get("step", 0.1)
        desc = metadata.get("desc", param_name)

        # Guess default based on name, but ensure it respects min/max bounds
        if "kappa" in param_name:
            default = min(2.0, max_val)
        elif "theta" in param_name:
            default = min(0.04, max_val)
        elif "sigma" in param_name and param_name != "initial_volatility":
            default = min(0.3, max_val)
        elif "rho" in param_name:
            default = max(min(-0.5, max_val), min_val)  # Clamp to [min_val, max_val]
        elif "lambda" in param_name:
            default = min(0.5, max_val)
        elif "mu" in param_name:
            default = max(min(0.0, max_val), min_val)  # Clamp to [min_val, max_val]
        elif "eta" in param_name:
            default = min(10.0, max_val)  # Use 10.0 but respect max_val
        elif "p_" in param_name or "p" in param_name and max_val <= 1.0:
            default = min(0.5, max_val)
        else:
            default = (min_val + max_val) / 2

        # Final safety: clamp default to [min_val, max_val]
        default = max(min_val, min(default, max_val))

        params[param_name] = st.number_input(
            f"{desc} ({param_name})",
            value=float(default),
            min_value=float(min_val),
            max_value=float(max_val),
            step=float(step),
        )

    return params


def render_option_type_selector() -> str:
    """Render option type selector with grouping."""
    categories = {}
    for key, info in OPTION_TYPES.items():
        category = info.get("category", "Other")
        if category not in categories:
            categories[category] = []
        categories[category].append((key, info["name"]))

    selected_category = st.selectbox("Option Category", list(categories.keys()))
    options_in_category = categories[selected_category]
    option_display = [name for _, name in options_in_category]

    selected_name = st.selectbox("Option Type", option_display)
    selected_key = [key for key, (_, name) in [(k, (k, n)) for k, n in options_in_category] if name == selected_name][0]

    return selected_key


def render_model_selector() -> str:
    """Render model selector."""
    model_names = {k: v.name for k, v in MODEL_REGISTRY.items()}
    selected_name = st.selectbox("Stochastic Model", list(model_names.values()))
    selected_key = [k for k, v in model_names.items() if v == selected_name][0]
    return selected_key

