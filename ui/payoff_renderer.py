"""
Explicit Streamlit rendering functions for payoff-specific parameters.
This DIRECTLY renders widgets and VISIBLY changes the UI.
"""

import streamlit as st
from typing import Dict, Any
from ui.payoff_ui_schema import PAYOFF_UI_SCHEMA


def render_payoff_parameters(option_type: str) -> Dict[str, Any]:
    """
    EXPLICITLY render payoff-specific parameters for the selected option type.

    This function:
    1. Reads the UI schema for the option type
    2. Renders exactly the widgets needed (VISIBLE CHANGE)
    3. Returns the collected parameter values

    The dashboard visibly changes when option type changes because different
    parameters are rendered here.
    """
    params = {}

    # Safety check
    if not option_type or option_type not in PAYOFF_UI_SCHEMA:
        st.info("Select an option type to configure payoff parameters.")
        return params

    # Get schema for this option type
    schema = PAYOFF_UI_SCHEMA[option_type]
    param_list = schema.get("params", [])

    if not param_list:
        st.warning(f"No parameters defined for {option_type}")
        return params

    # Display section header with option category
    category = schema.get("category", "Option")
    st.markdown(f"### 📋 Payoff Parameters ({category})")

    # Disclaimer for experimental / WIP payoff implementations
    if category in ("Parisian", "Bermudan", "Multi-Asset"):
        # Small, unobtrusive caption next to the parameters
        st.caption("(Work in progress)")

    # Determine layout: use columns for better UI organization
    num_params = len(param_list)

    if num_params <= 2:
        # Single or double parameter: use 2 columns
        cols = st.columns(2)
        for idx, param_def in enumerate(param_list):
            col_idx = idx % 2
            with cols[col_idx]:
                _render_single_parameter(param_def, params)
    elif num_params <= 4:
        # 3-4 parameters: use 2x2 grid
        cols = st.columns(2)
        for idx, param_def in enumerate(param_list):
            col_idx = idx % 2
            with cols[col_idx]:
                _render_single_parameter(param_def, params)
    else:
        # 5+ parameters: use single column
        for param_def in param_list:
            _render_single_parameter(param_def, params)

    return params


def _render_single_parameter(param_def: Dict[str, Any], params_dict: Dict[str, Any]) -> None:
    """
    Helper: Render a single parameter input widget and store the value.

    This ensures each parameter gets exactly one input widget, properly typed.
    """
    key = param_def.get("key")
    label = param_def.get("label", key)
    param_type = param_def.get("type", "float")
    default = param_def.get("default", 100.0)
    min_val = param_def.get("min", 0.01)
    max_val = param_def.get("max", 10000)
    step = param_def.get("step", 1.0)

    # Create Streamlit key for widget (ensures proper state management)
    widget_key = f"payoff_param_{key}"

    if param_type == "float":
        value = st.number_input(
            label=label,
            value=float(default),
            min_value=float(min_val),
            max_value=float(max_val),
            step=float(step),
            key=widget_key,
        )
        params_dict[key] = float(value)

    elif param_type == "int":
        value = st.number_input(
            label=label,
            value=int(default),
            min_value=int(min_val),
            max_value=int(max_val),
            step=int(step),
            key=widget_key,
        )
        params_dict[key] = int(value)


def get_payoff_params_schema(option_type: str) -> Dict[str, Any]:
    """
    Get the UI schema for an option type.
    Useful for validation and debugging.
    """
    return PAYOFF_UI_SCHEMA.get(option_type, {})


def list_payoff_parameters_for_type(option_type: str) -> list:
    """
    Return the list of parameter keys required for an option type.
    """
    schema = PAYOFF_UI_SCHEMA.get(option_type, {})
    params = schema.get("params", [])
    return [p.get("key") for p in params]
