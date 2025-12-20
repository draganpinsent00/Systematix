"""
Session state management.
"""

import streamlit as st
from typing import Any, Dict, Optional


def initialize_session_state():
    """Initialize all session state variables."""
    if "market_params" not in st.session_state:
        st.session_state.market_params = {}

    if "mc_settings" not in st.session_state:
        st.session_state.mc_settings = {}

    if "rng_settings" not in st.session_state:
        st.session_state.rng_settings = {}

    if "model_name" not in st.session_state:
        st.session_state.model_name = "gbm"

    if "model_params" not in st.session_state:
        st.session_state.model_params = {}

    if "option_type" not in st.session_state:
        st.session_state.option_type = "european_call"

    if "option_params" not in st.session_state:
        st.session_state.option_params = {}

    if "mc_result" not in st.session_state:
        st.session_state.mc_result = None

    if "pricing_config" not in st.session_state:
        st.session_state.pricing_config = {}

    if "run_history" not in st.session_state:
        st.session_state.run_history = []


def save_config(key: str, value: Any):
    """Save configuration value to session state."""
    st.session_state[key] = value


def get_config(key: str, default: Any = None) -> Any:
    """Retrieve configuration value."""
    return st.session_state.get(key, default)


def build_pricing_config() -> Dict[str, Any]:
    """Build complete pricing configuration from state."""
    return {
        "market_params": st.session_state.get("market_params", {}),
        "mc_settings": st.session_state.get("mc_settings", {}),
        "rng_settings": st.session_state.get("rng_settings", {}),
        "model_name": st.session_state.get("model_name", "gbm"),
        "model_params": st.session_state.get("model_params", {}),
        "option_type": st.session_state.get("option_type", "european_call"),
        "option_params": st.session_state.get("option_params", {}),
    }


def display_config_summary(config: Dict[str, Any]):
    """Display resolved configuration as a summary."""
    with st.expander("📋 Configuration Summary", expanded=False):
        st.json(config)


def clear_app_state():
    """Clear only application-owned session state keys, preserve page navigation."""
    # List of application-owned keys to reset
    APP_KEYS = [
        "market_params",
        "mc_settings",
        "rng_settings",
        "model_name",
        "model_params",
        "option_type",
        "option_params",
        "mc_result",
        "pricing_config",
        "model",
        "rng",
    ]

    for key in APP_KEYS:
        if key in st.session_state:
            del st.session_state[key]

    # Re-initialize defaults
    initialize_session_state()


def append_to_history(timestamp: str, model: str, option_type: str, price: float, params_summary: Dict[str, Any]):
    """Append a successful run to the history."""
    if "run_history" not in st.session_state:
        st.session_state.run_history = []

    entry = {
        "timestamp": timestamp,
        "model": model,
        "option_type": option_type,
        "price": price,
        "params_summary": params_summary,
    }

    st.session_state.run_history.append(entry)

