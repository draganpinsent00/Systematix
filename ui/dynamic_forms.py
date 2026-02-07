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

    # Build display labels for categories; append WIP marker for experimental categories
    WIP_CATEGORIES = {"Parisian", "Bermudan", "Multi-Asset", "Rainbow"}
    display_label_map = {}
    display_categories = []
    for cat in sorted(categories.keys()):
        if cat in WIP_CATEGORIES:
            label = f"{cat} (Work in progress)"
        else:
            label = cat
        display_categories.append(label)
        display_label_map[label] = cat

    # Show select box with user-friendly labels
    selected_display = st.selectbox("Option Category", display_categories, key="option_category_selector")
    # Map back to original category key
    selected_category = display_label_map[selected_display]

    options_in_category = categories[selected_category]

    # For WIP categories append the marker to each option display name, and
    # create a mapping from displayed name back to the option key so selection
    # still resolves correctly.
    display_to_key = {}
    option_display = []
    if selected_category in WIP_CATEGORIES:
        for key, name in options_in_category:
            disp = f"{name} (Work in progress)"
            option_display.append(disp)
            display_to_key[disp] = key
    else:
        for key, name in options_in_category:
            option_display.append(name)
            display_to_key[name] = key

    selected_name = st.selectbox("Option Type", option_display, key="option_type_selector")
    # Map displayed selection back to the option key
    selected_key = display_to_key[selected_name]

    return selected_key


def render_model_selector() -> str:
    """Render model selector."""
    model_names = {k: v.name for k, v in MODEL_REGISTRY.items()}
    selected_name = st.selectbox("Stochastic Model", list(model_names.values()))
    selected_key = [k for k, v in model_names.items() if v == selected_name][0]
    return selected_key
