"""
Main Streamlit application for Monte Carlo options pricing.
Refactored as single-page, professional dashboard.
"""

import streamlit as st
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

# ...existing imports...
from config.settings import NUM_PATHS_PLOT, DEFAULT_NUM_SIMULATIONS, DEFAULT_NUM_TIMESTEPS
from config.schemas import MODEL_REGISTRY, OPTION_TYPES
from core.rng_engines import create_rng
from models.gbm import GBM
from models.heston import Heston
from models.heston_3_2 import Heston32
from models.merton_jump import MertonJump
from models.kou_jump import KouJump
from models.sabr import SABR
from models.bachelier import Bachelier
from models.local_volatility import LocalVolatility
from models.regime_switching import RegimeSwitchingGBM
from models.shifted_lognormal import ShiftedLognormal
from instruments.registry import create_instrument
from core.mc_engine import MonteCarloEngine
from analytics.greeks import GreeksComputer
from analytics.diagnostics import DiagnosticsAnalyzer
from visualization.charts_paths import chart_simulation_paths, chart_distribution_histogram
from visualization.charts_payoffs import chart_greeks_sensitivity
from visualization.charts_diagnostics import chart_convergence
from visualization.plotly_theme import apply_theme
from ui.components import (
    input_market_params, input_mc_settings, input_rng_settings,
    input_option_params, show_error, show_success
)
from ui.dynamic_forms import render_model_params_dynamic, render_option_type_selector, render_model_selector
from ui.state import initialize_session_state, save_config, get_config, build_pricing_config, display_config_summary
from utils.validation import validate_market_params, validate_mc_settings, validate_option_params
from utils.logging import log_pricing_event


# =====================
# LAYOUT HELPERS
# =====================

def render_header():
    """Render professional header section."""
    st.set_page_config(
        page_title="StochastiX - Derivatives Pricing",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Professional brand-aligned styling
    st.markdown("""
    <style>
    /* Color Palette */
    :root {
        --navy-dark: #0a0e1f;
        --navy-primary: #141829;
        --slate-secondary: #2a3f5f;
        --text-primary: #f5f7fa;
        --text-secondary: #d4dce8;
        --accent-gold: #d4af7c;
        --accent-gold-muted: #c4a070;
    }
    
    /* Global Styling */
    body {
        background-color: var(--navy-dark);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    
    .main {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, var(--navy-dark) 0%, var(--navy-primary) 100%);
    }

    .stApp {
        background: linear-gradient(135deg, var(--navy-dark) 0%, var(--navy-primary) 100%);
    }

    /* Header Styling */
    .header-container {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0 1rem 0;
        border-bottom: 2px solid var(--accent-gold);
    }

    .header-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: var(--accent-gold);
        letter-spacing: 1px;
        margin: 0;
        padding: 0;
    }

    .header-subtitle {
        font-size: 1rem;
        color: var(--accent-gold);
        font-weight: 400;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }

    /* Section Headers */
    h2 {
        color: var(--accent-gold) !important;
        font-weight: 600;
        font-size: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 3px solid var(--accent-gold);
        padding-left: 1rem;
    }
    
    h3 {
        color: var(--accent-gold);
        font-weight: 500;
        font-size: 1.1rem;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }
    
    /* Text Elements - Gold */
    p, label, .stLabel {
        color: var(--accent-gold) !important;
        font-weight: 400;
    }
    
    /* Metric Containers */
    [data-testid="metric-container"] {
        background: var(--slate-secondary);
        border-left: 3px solid var(--accent-gold);
        border-radius: 0.25rem;
    }

    .metric-container {
        background: var(--slate-secondary);
        padding: 1.25rem;
        border-radius: 0.5rem;
        border-left: 3px solid var(--accent-gold);
    }
    
    /* Metric Text - Gold */
    [data-testid="stMetricValue"] {
        color: var(--accent-gold) !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--accent-gold) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--accent-gold);
        color: #000000;
        font-weight: 600;
        border: none;
        border-radius: 0.375rem;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background-color: var(--accent-gold-muted);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(212, 175, 124, 0.3);
    }

    /* Input Elements - Dark background matching main background */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stSlider > div > div > div {
        background-color: var(--navy-dark) !important;
        color: var(--accent-gold) !important;
        border: 1px solid var(--accent-gold);
        border-radius: 0.375rem;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: var(--navy-dark);
        color: var(--accent-gold) !important;
    }

    .streamlit-expanderHeader:hover {
        background-color: rgba(212, 175, 124, 0.1);
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid var(--accent-gold);
        margin: 2rem 0;
    }

    /* Alerts & Messages */
    .stAlert {
        background-color: var(--slate-secondary);
        border-left: 4px solid var(--accent-gold);
        border-radius: 0.375rem;
    }
    
    .stSuccess, .stInfo, .stWarning {
        color: var(--text-primary);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--navy-primary) 0%, var(--slate-secondary) 100%);
    }

    .sidebar .stMarkdown {
        color: var(--accent-gold);
    }
    
    .sidebar h2 {
        color: var(--accent-gold);
        border: none;
        padding: 0;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        border-bottom: 2px solid transparent;
        color: var(--accent-gold);
    }
    
    .stTabs [aria-selected="true"] {
        color: var(--accent-gold);
        border-bottom-color: var(--accent-gold);
    }

    /* Data Tables */
    .dataframe {
        background-color: var(--navy-primary);
        color: var(--text-primary);
    }

    /* Label Styling */
    label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }

    .stCheckbox > label {
        color: var(--accent-gold) !important;
    }

    .stRadio > label {
        color: var(--accent-gold) !important;
    }
    
    /* Spinner & Loading */
    .stSpinner {
        color: var(--accent-gold) !important;
    }
    
    /* Markdown text */
    .stMarkdown p {
        color: var(--accent-gold);
    }
    
    /* Code blocks */
    pre {
        background-color: var(--navy-primary) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--accent-gold) !important;
    }
    
    code {
        background-color: var(--slate-secondary) !important;
        color: var(--accent-gold) !important;
        padding: 0.2rem 0.4rem;
        border-radius: 0.25rem;
    }

    /* Spacing Utilities */
    .spacer { margin: 2rem 0; }
    
    /* ===== COMPREHENSIVE BRAND STYLING ===== */
    
    /* Force all text to white/off-white on dark backgrounds */
    * {
        color: #ffffff !important;
    }
    
    /* Gold for headers and emphasis */
    h1, h2, h3, h4, h5, h6 {
        color: var(--accent-gold) !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Specific text elements */
    p, label, span, div, a {
        color: #ffffff !important;
    }
    
    /* Metrics - Gold values, white labels */
    [data-testid="stMetricValue"] {
        color: var(--accent-gold) !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    
    /* Button text - Black on gold */
    .stButton > button {
        color: #000000 !important;
    }
    
    /* ===== INPUT WIDGET STYLING (CRITICAL FIX) ===== */
    /* Input boxes must have dark background + white text for readability */
    input, 
    select, 
    textarea,
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stMultiSelect > div > div > select,
    .stTextArea > div > div > textarea,
    .stSlider > div > div > input {
        background-color: var(--navy-primary) !important;
        color: #ffffff !important;
        border: 2px solid var(--accent-gold) !important;
        border-radius: 0.375rem !important;
    }
    
    /* Placeholder text - lighter grey for contrast */
    input::placeholder,
    textarea::placeholder,
    select::placeholder {
        color: #b0b0b0 !important;
        opacity: 0.8 !important;
    }
    
    /* Input focus state - emphasize gold border */
    input:focus,
    select:focus,
    textarea:focus {
        outline: none !important;
        border-color: #e8c86d !important;
        box-shadow: 0 0 0 3px rgba(212, 175, 124, 0.2) !important;
    }
    
    /* ===== NUMBER INPUT PLUS/MINUS BUTTONS ===== */
    /* Streamlit number input increment/decrement buttons */
    .stNumberInput button {
        background-color: var(--navy-primary) !important;
        color: #ffffff !important;
        border: 2px solid var(--accent-gold) !important;
    }
    
    .stNumberInput button:hover {
        background-color: rgba(212, 175, 124, 0.15) !important;
        border-color: #e8c86d !important;
    }
    
    /* Plus/minus icon styling */
    .stNumberInput svg {
        color: #ffffff !important;
    }
    
    /* ===== DROPDOWN & SELECT STYLING ===== */
    /* Streamlit selectbox dropdown container */
    .stSelectbox > div {
        background-color: var(--navy-primary) !important;
    }
    
    /* Force all nested selectbox divs to dark navy */
    .stSelectbox > div > div {
        background-color: var(--navy-primary) !important;
    }
    
    .stSelectbox > div > div > div {
        background-color: var(--navy-primary) !important;
        color: var(--accent-gold) !important;
    }
    
    /* Override the actual select input field */
    .stSelectbox select {
        background-color: var(--navy-primary) !important;
    }
    
    /* Base web select component - force dark navy */
    [data-baseweb="select"] {
        background-color: var(--navy-primary) !important;
    }
    
    [data-baseweb="select"] > div {
        background-color: var(--navy-primary) !important;
    }
    
    [data-baseweb="select"] input {
        background-color: var(--navy-primary) !important;
        color: var(--accent-gold) !important;
    }
    
    /* Dropdown menu when open */
    [data-baseweb="menu"] {
        background-color: var(--navy-primary) !important;
    }
    
    [data-baseweb="menu"] > div {
        background-color: var(--navy-primary) !important;
    }
    
    [data-baseweb="menu"] ul {
        background-color: var(--navy-primary) !important;
    }
    
    /* Force dropdown container to dark navy */
    [role="listbox"] {
        background-color: var(--navy-primary) !important;
    }
    
    [role="option"] {
        background-color: var(--navy-primary) !important;
        color: #ffffff !important;
    }
    
    [role="option"]:hover {
        background-color: rgba(212, 175, 124, 0.2) !important;
        color: #ffffff !important;
    }
    
    /* Dropdown menu items */
    [data-baseweb="listbox"] {
        background-color: var(--navy-primary) !important;
    }
    
    [data-baseweb="listbox"] > div {
        background-color: var(--navy-primary) !important;
    }
    
    [data-baseweb="listbox"] li {
        background-color: var(--navy-primary) !important;
        color: #ffffff !important;
    }
    
    [data-baseweb="listbox"] li:hover {
        background-color: rgba(212, 175, 124, 0.2) !important;
        color: #ffffff !important;
    }
    
    /* Selected item highlight - gold background, white text */
    [data-baseweb="listbox"] li[aria-selected="true"] {
        background-color: rgba(212, 175, 124, 0.3) !important;
        color: #ffffff !important;
    }
    
    /* Generic dropdown/popover styling */
    div[class*="dropdown"],
    div[class*="popper"],
    div[class*="menu"],
    div[class*="select"] {
        background-color: var(--navy-primary) !important;
    }
    
    /* Multiselect styling */
    .stMultiSelect > div {
        background-color: var(--navy-primary) !important;
    }
    
    [data-baseweb="tag"] {
        background-color: rgba(212, 175, 124, 0.2) !important;
        color: var(--accent-gold) !important;
    }
    
    [data-baseweb="tag"] button {
        color: var(--accent-gold) !important;
    }
    
    /* Dropdown arrow/chevron icon */
    .stSelectbox [data-testid="stSelectbox"] svg,
    .stMultiSelect [data-testid="stMultiSelect"] svg {
        color: var(--accent-gold) !important;
    }
    
    /* Table cells - White text */
    table td, table th {
        color: #ffffff !important;
    }
    
    th {
        background-color: var(--navy-dark) !important;
        color: var(--accent-gold) !important;
    }
    
    /* ===== EXPANDER / DROPDOWN STYLING ===== */
    /* Risk Sensitivities dropdown background - GOLD when opened */
    [data-baseweb="baseButton"] {
        background-color: var(--accent-gold) !important;
    }
    
    /* Expander opened background - gold */
    .streamlit-expanderHeader {
        background-color: var(--accent-gold) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #e8c86d !important;
    }
    
    /* Expander text color - black on gold background */
    .streamlit-expanderHeader > p {
        color: #000000 !important;
    }
    
    /* Section header styling - white text, gold underline */
    .stMetric + .stMarkdown h2,
    .stMetric + .stMarkdown h3 {
        color: #ffffff !important;
        border-bottom: 3px solid var(--accent-gold) !important;
        padding-bottom: 0.5rem !important;
    }
    
    /* ===== UNIVERSAL DROPDOWN OVERRIDE ===== */
    /* Force ALL dropdown/menu/popup elements to dark navy background */
    /* This catches Popper overlays and other popup libraries */
    div[style*="background-color: rgb(255, 255, 255)"],
    div[style*="background-color:rgb(255, 255, 255)"],
    [style*="white"] {
        background-color: var(--navy-primary) !important;
        color: #ffffff !important;
    }
    
    /* Override any white background divs in popups */
    body > div[role="dialog"],
    body > div[class*="popper"],
    body > div[class*="popup"],
    body > div > div[style*="position: fixed"] {
        background-color: var(--navy-primary) !important;
    }
    
    /* Catch any remaining white menus */
    .popper {
        background-color: var(--navy-primary) !important;
    }
    
    .popper > div {
        background-color: var(--navy-primary) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown('<h1 class="header-title">StochastiX</h1>', unsafe_allow_html=True)
        st.markdown('<p class="header-subtitle">Advanced Exotic Derivatives Pricing</p>', unsafe_allow_html=True)
        st.markdown("---")

    # Add Volatility Surface button to the top-right of the header
    with col3:
        if st.button("Volatility Surface Analysis", use_container_width=False, key="vol_surface_header_btn"):
            st.session_state["page"] = "volatility_surface"
            st.rerun()


def render_inputs_section():
    """Render all input controls in organized columns."""
    st.markdown("## Pricing Configuration")

    # Initialize all variables
    model_params = {}
    option_type = None
    option_params = {}

    # Market Parameters Column
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Market Parameters")
        market_params = input_market_params()
        save_config("market_params", market_params)

    with col2:
        st.subheader("Monte Carlo Settings")
        mc_settings = input_mc_settings()
        save_config("mc_settings", mc_settings)

    # RNG & Model Selection
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Random Number Generator")
        rng_settings = input_rng_settings()
        save_config("rng_settings", rng_settings)

    with col2:
        st.subheader("Stochastic Model Selection")
        model_name = render_model_selector()
        save_config("model_name", model_name)

    # Model-specific and Option Parameters
    if model_name:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Model Parameters")
            model_params = render_model_params_dynamic(model_name)
            save_config("model_params", model_params)

        with col2:
            st.subheader("Derivative Specification")
            option_type = render_option_type_selector()
            save_config("option_type", option_type)

    # EXPLICIT PAYOFF PARAMETER RENDERING
    # This section VISIBLY changes when option_type changes
    if option_type:
        st.markdown("---")
        # Import the explicit renderer
        from ui.payoff_renderer import render_payoff_parameters

        # Render payoff-specific parameters - THIS IS WHERE THE UI VISIBLY CHANGES
        option_params = render_payoff_parameters(option_type)
        save_config("option_params", option_params)

    return market_params, mc_settings, rng_settings, model_name, model_params, option_type, option_params

def render_control_buttons():
    """Render action buttons."""
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        run_button = st.button("Run Pricing", key="run_pricing", use_container_width=True)
    with col2:
        config_button = st.button("View Configuration", key="show_config", use_container_width=True)
    with col3:
        reset_button = st.button("Clear State", key="reset_config", use_container_width=True)

    return run_button, config_button, reset_button


def render_metrics_section(mc_result):
    """Render key metrics in professional grid."""
    st.markdown("## Valuation Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Fair Value",
            f"${mc_result.price:.6f}",
            delta=None,
        )
    with col2:
        st.metric(
            "Standard Error",
            f"${mc_result.std_error:.6f}",
            delta=None,
        )
    with col3:
        st.metric(
            "Confidence Interval (95%) - Lower",
            f"${mc_result.ci_lower:.6f}",
            delta=None,
        )
    with col4:
        st.metric(
            "Confidence Interval (95%) - Upper",
            f"${mc_result.ci_upper:.6f}",
            delta=None,
        )


def render_visualizations_section(mc_result, market_params):
    """Render all chart visualizations."""
    st.markdown("## Path Analysis & Convergence")

    # Path visualization and Payoff Distribution side by side
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Simulated Asset Paths")
        fig_paths = chart_simulation_paths(mc_result.paths, num_to_plot=NUM_PATHS_PLOT)
        st.plotly_chart(fig_paths, use_container_width=True)

    with col2:
        st.markdown("#### Terminal Payoff Distribution")
        fig_hist = chart_distribution_histogram(mc_result.payoffs, "Option Payoffs")
        st.plotly_chart(fig_hist, use_container_width=True)

    # Convergence chart
    if mc_result.convergence_history is not None:
        st.markdown("#### Convergence Analysis")
        fig_conv = chart_convergence(mc_result.convergence_history)
        st.plotly_chart(fig_conv, use_container_width=True)


def render_greeks_diagnostics_section(mc_result, market_params, option_type, model=None, rng=None, mc_settings=None):
    """Render Greeks and Diagnostics in expandable sections."""
    st.markdown("## Risk Analysis & Simulation Diagnostics")

    col1, col2 = st.columns(2)

    # Greeks Expander
    with col1:
        with st.expander("Risk Sensitivities (Greeks)", expanded=False):
            with st.spinner("Computing Greeks..."):
                greeks_computer = GreeksComputer()
                instrument = create_instrument(
                    option_type,
                    strike=get_config("option_params", {}).get('strike', 100.0),
                )

                greeks = greeks_computer.compute_all(
                    spot=market_params['spot'],
                    price=mc_result.price,
                    paths=mc_result.paths,
                    payoff_func=instrument.payoff,
                    risk_free_rate=market_params['risk_free_rate'],
                    time_to_maturity=market_params['time_to_maturity'],
                    volatility=market_params['initial_volatility'],
                    model=model,
                    rng_engine=rng,
                    num_paths=mc_settings['num_simulations'] if mc_settings else None,
                    num_steps=mc_settings['num_timesteps'] if mc_settings else None,
                )

                # Display Greeks as metrics
                g_col1, g_col2, g_col3 = st.columns(3)
                with g_col1:
                    st.metric("Delta (∂V/∂S)", f"{greeks['delta']:.6f}")
                    st.metric("Gamma (∂²V/∂S²)", f"{greeks['gamma']:.6f}")
                with g_col2:
                    st.metric("Vega (∂V/∂σ)", f"{greeks['vega']:.6f}")
                    st.metric("Theta (∂V/∂t)", f"{greeks['theta']:.6f}")
                with g_col3:
                    st.metric("Rho (∂V/∂r)", f"{greeks['rho']:.6f}")

                # Greeks chart
                fig_greeks = chart_greeks_sensitivity(greeks)
                st.plotly_chart(fig_greeks, use_container_width=True)

    # Diagnostics Expander
    with col2:
        with st.expander("Simulation Quality Metrics", expanded=False):
            analyzer = DiagnosticsAnalyzer()

            # Path diagnostics
            diag = analyzer.path_diagnostics(mc_result.paths)

            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.metric("Mean Terminal Spot", f"${diag['mean_final_spot']:.2f}")
                st.metric("Terminal Spot Std Dev", f"${diag['std_final_spot']:.2f}")
            with d_col2:
                st.metric("Number of Paths", int(diag['num_paths']))
                st.metric("Steps Per Path", int(diag['num_steps']))

            # Convergence metrics
            if mc_result.convergence_history is not None:
                st.markdown("**Convergence Metrics**")
                conv_diag = analyzer.convergence_analysis(mc_result.convergence_history)
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    st.metric("Convergence Rate", f"{conv_diag.get('convergence_rate', 1.0):.4f}")
                with c_col2:
                    st.metric("Final Standard Error", f"${conv_diag.get('final_std', 0):.6f}")


# =====================
# MODEL BUILDER
# =====================

def build_model(model_name: str, market_params: Dict[str, float], model_params: Dict[str, Any]):
    """
    Factory to construct model instance.

    Defensive: Handles missing or None model_params with sensible defaults.
    """
    # Ensure model_params is a dict, not None
    if model_params is None:
        model_params = {}

    base_params = {
        'spot': market_params.get('spot', 100.0),
        'risk_free_rate': market_params.get('risk_free_rate', 0.05),
        'dividend_yield': market_params.get('dividend_yield', 0.0),
        'initial_volatility': market_params.get('initial_volatility', 0.20),
        'time_to_maturity': market_params.get('time_to_maturity', 1.0),
    }

    if model_name == "gbm":
        return GBM(**base_params)
    elif model_name == "heston":
        return Heston(
            **base_params,
            kappa=model_params.get('heston_kappa', 2.0),
            theta=model_params.get('heston_theta', 0.04),
            sigma=model_params.get('heston_sigma', 0.3),
            rho=model_params.get('heston_rho', -0.5),
        )
    elif model_name == "heston_3_2":
        return Heston32(
            **base_params,
            kappa=model_params.get('heston_32_kappa', 2.0),
            theta=model_params.get('heston_32_theta', 0.04),
            sigma=model_params.get('heston_32_sigma', 0.3),
            rho=model_params.get('heston_32_rho', -0.5),
        )
    elif model_name == "merton_jump":
        return MertonJump(
            **base_params,
            lambda_=model_params.get('merton_lambda', 0.5),
            mu_j=model_params.get('merton_mu_j', 0.0),
            sigma_j=model_params.get('merton_sigma_j', 0.2),
        )
    elif model_name == "kou_jump":
        return KouJump(
            **base_params,
            lambda_=model_params.get('kou_lambda', 0.5),
            p_up=model_params.get('kou_p_up', 0.5),
            eta_up=model_params.get('kou_eta_up', 20.0),
            eta_down=model_params.get('kou_eta_down', 10.0),
        )
    elif model_name == "sabr":
        return SABR(
            **base_params,
            alpha=model_params.get('sabr_alpha', 0.4),
            beta=model_params.get('sabr_beta', 0.5),
            nu=model_params.get('sabr_nu', 0.5),
            rho=model_params.get('sabr_rho', -0.5),
        )
    elif model_name == "bachelier":
        return Bachelier(**base_params)
    elif model_name == "local_volatility":
        return LocalVolatility(**base_params)
    elif model_name == "regime_switching":
        return RegimeSwitchingGBM(**base_params)
    elif model_name == "shifted_lognormal":
        return ShiftedLognormal(**base_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# =====================
# MAIN APPLICATION
# =====================

def run_pricing():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # Initialize page navigation
    if "page" not in st.session_state:
        st.session_state["page"] = "dashboard"

    # Route to appropriate page
    if st.session_state["page"] == "dashboard":
        render_dashboard()
    elif st.session_state["page"] == "volatility_surface":
        render_volatility_surface_page()


def render_dashboard():
    """Render the main pricing dashboard."""
    # Render header
    render_header()

    # Render inputs section
    market_params, mc_settings, rng_settings, model_name, model_params, option_type, option_params = render_inputs_section()

    st.markdown("---")

    # Navigation and Control buttons
    nav_col, btn_col1, btn_col2 = st.columns([1, 1, 1])

    # Control buttons
    run_button, config_button, reset_button = render_control_buttons()

    # Handle button actions
    if reset_button:
        st.session_state.clear()
        st.session_state["page"] = "dashboard"
        st.rerun()

    if config_button:
        display_config_summary(build_pricing_config())

    if run_button:
        # Validate that required selections were made
        if not model_name:
            show_error("Please select a stochastic model before running pricing")
            return

        if not option_type:
            show_error("Please select an option type before running pricing")
            return

        # Validate inputs
        valid, msg = validate_market_params(market_params)
        if not valid:
            show_error(msg)
            return

        valid, msg = validate_mc_settings(mc_settings)
        if not valid:
            show_error(msg)
            return

        valid, msg = validate_option_params(option_params)
        if not valid:
            show_error(msg)
            return

        # Build configuration
        pricing_config = build_pricing_config()

        with st.spinner("Running Monte Carlo simulation..."):
            try:
                # Create RNG
                rng = create_rng(
                    engine=rng_settings['engine'],
                    seed=rng_settings['seed'],
                )

                # Build model
                model = build_model(model_name, market_params, model_params)
                valid, msg = model.validate()
                if not valid:
                    show_error(f"Model validation error: {msg}")
                    return

                # Generate paths
                paths = model.generate_paths(
                    rng,
                    num_paths=mc_settings['num_simulations'],
                    num_steps=mc_settings['num_timesteps'],
                    distribution=rng_settings.get('distribution', 'normal'),
                    student_t_df=rng_settings.get('student_t_df', 3.0),
                    antithetic_variates=rng_settings.get('antithetic_variates', True),
                    use_sobol=rng_settings.get('use_sobol', False),
                )

                # Create instrument
                instrument = create_instrument(
                    option_type,
                    strike=option_params.get('strike', 100.0),
                    **{k: v for k, v in option_params.items() if k != 'strike'}
                )

                # Price with MC
                mc_engine = MonteCarloEngine(
                    rng,
                    num_simulations=mc_settings['num_simulations'],
                    num_timesteps=mc_settings['num_timesteps'],
                )

                # Detect if American option and use LSM pricing
                is_american = "american" in option_type.lower()
                lsm_config = {} if is_american else None

                mc_result = mc_engine.price(
                    paths,
                    instrument.payoff,
                    risk_free_rate=market_params['risk_free_rate'],
                    time_to_maturity=market_params['time_to_maturity'],
                    use_lsm=is_american,
                    lsm_config=lsm_config,
                )

                save_config("mc_result", mc_result)
                save_config("pricing_config", pricing_config)
                save_config("model", model)
                save_config("rng", rng)
                save_config("mc_settings", mc_settings)

                log_pricing_event(model_name, option_type, mc_result.price)
                show_success(f"Pricing calculation complete. Fair value: ${mc_result.price:.6f}")

            except Exception as e:
                show_error(f"Pricing error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
                return

    st.markdown("---")

    # Display results if available
    mc_result = get_config("mc_result")

    if mc_result is None:
        st.info("Configure pricing parameters and click 'Run Pricing' to initiate Monte Carlo simulation.")
    else:
        # Render results sections
        render_metrics_section(mc_result)
        st.markdown("")
        render_visualizations_section(mc_result, market_params)
        st.markdown("")
        # Retrieve saved model and rng for Greeks computation
        saved_model = get_config("model")
        saved_rng = get_config("rng")
        saved_mc_settings = get_config("mc_settings")
        render_greeks_diagnostics_section(mc_result, market_params, option_type, model=saved_model, rng=saved_rng, mc_settings=saved_mc_settings)


def render_volatility_surface_page():
    """Render the volatility surface analysis page with real market data."""
    st.set_page_config(
        page_title="Volatility Surface - Systematix",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Apply the same brand styling as main dashboard
    st.markdown("""
    <style>
    /* Color Palette */
    :root {
        --navy-dark: #0a0e1f;
        --navy-primary: #141829;
        --slate-secondary: #2a3f5f;
        --text-primary: #f5f7fa;
        --text-secondary: #d4dce8;
        --accent-gold: #d4af7c;
        --accent-gold-muted: #c4a070;
    }
    
    /* Global Styling */
    body {
        background-color: var(--navy-dark);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', sans-serif;
    }
    
    .main {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem 1rem;
        background: linear-gradient(135deg, var(--navy-dark) 0%, var(--navy-primary) 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--navy-dark) 0%, var(--navy-primary) 100%);
    }
    
    /* Header Styling */
    .header-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: var(--accent-gold);
        letter-spacing: 1px;
        margin: 0;
        padding: 0;
    }
    
    .header-subtitle {
        font-size: 1rem;
        color: var(--text-secondary);
        font-weight: 400;
        letter-spacing: 0.5px;
        margin-top: 0.5rem;
    }
    
    /* Section Headers */
    h2 {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 1.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 3px solid var(--accent-gold);
        padding-left: 1rem;
    }
    
    h3, h4 {
        color: var(--text-primary);
        font-weight: 500;
        margin-top: 1rem;
        margin-bottom: 0.75rem;
    }
    
    /* Text Elements */
    p, label, .stLabel {
        color: var(--text-secondary);
        font-weight: 400;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--accent-gold);
        color: #000000;
        font-weight: 600;
        border: none;
        border-radius: 0.375rem;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: var(--accent-gold-muted);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(212, 175, 124, 0.3);
    }
    
    /* Input Elements */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stSlider > div > div > div {
        background-color: var(--slate-secondary);
        color: var(--text-primary);
        border: 1px solid var(--accent-gold);
        border-radius: 0.375rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--navy-primary) 0%, var(--slate-secondary) 100%);
    }
    
    .sidebar .stMarkdown {
        color: var(--text-secondary);
    }
    
    .sidebar h2 {
        color: var(--accent-gold);
        border: none;
        padding: 0;
    }
    
    /* Data Tables */
    .dataframe {
        background-color: var(--navy-primary);
        color: var(--text-primary);
    }
    
    /* Label Styling */
    label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    /* Alerts & Messages */
    .stAlert {
        background-color: var(--slate-secondary);
        border-left: 4px solid var(--accent-gold);
        border-radius: 0.375rem;
    }
    
    .stSuccess, .stInfo, .stWarning {
        color: var(--text-primary);
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid var(--accent-gold);
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Back button
    if st.button("← Back to Pricing Dashboard", use_container_width=True):
        st.session_state["page"] = "dashboard"
        st.rerun()

    st.markdown('<h1 class="header-title" style="text-align:center;">Volatility Surface Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle" style="text-align:center;">European Option Market Quotes & Implied Volatility Surface Construction</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.markdown("## Market Data Configuration")

        ticker = st.text_input(
            "Underlying Equity Ticker",
            value="AAPL",
            help="Enter a valid stock ticker symbol (e.g., AAPL, MSFT, GOOGL)"
        ).upper()

        risk_free_rate = st.slider(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
            help="Annual risk-free rate (e.g., 10-year Treasury yield)"
        ) / 100.0

        dividend_yield = st.slider(
            "Dividend Yield (%)",
            min_value=0.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            help="Annual dividend yield of the underlying equity"
        ) / 100.0

        st.markdown("### Option Selection Filters")
        strike_range = st.slider(
            "Moneyness Range (% of Spot)",
            min_value=50,
            max_value=150,
            value=(80, 120),
            step=5,
            help="Include options within this moneyness range"
        )

        maturity_range = st.slider(
            "Time to Expiration (Days)",
            min_value=1,
            max_value=365,
            value=(7, 180),
            step=7,
            help="Include options within this maturity window"
        )

        st.markdown("### Data Handling")
        use_cache = st.checkbox("Use cached data (for testing)", value=True, help="Reduces API calls")

        fetch_button = st.button("Fetch Market Data", use_container_width=True)

    # Main content area
    if fetch_button or "vol_surface_data" in st.session_state:
        if fetch_button:
            with st.spinner(f"Fetching real market data for {ticker}..."):
                try:
                    # Fetch real market data
                    market_data = _fetch_real_market_data(
                        ticker,
                        strike_range,
                        maturity_range,
                        use_cache=use_cache
                    )

                    if market_data is None:
                        st.error(f"Could not fetch market data for {ticker}. Please check the ticker symbol.")
                        return

                    if market_data["empty_data"]:
                        st.warning(f"No option data available for {ticker} in the specified filters. Try adjusting the filters.")
                        return

                    st.session_state["vol_surface_data"] = market_data
                    st.success(f"Successfully fetched {len(market_data['df'])} market option quotes for {ticker}")

                except Exception as e:
                    st.error(f"Error fetching market data: {str(e)}")
                    return

        # Display results if data exists
        if "vol_surface_data" in st.session_state:
            market_data = st.session_state["vol_surface_data"]
            df = market_data["df"]
            spot_price = market_data["spot_price"]
            ticker = market_data["ticker"]
            data_source = market_data["data_source"]
            fetch_time = market_data["fetch_time"]

            # Display data source and timestamp
            st.info(f"**Data Source:** {data_source} | **Fetch Time:** {fetch_time}")

            # Display summary metrics
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Underlying Ticker", ticker)
            with m_col2:
                st.metric("Current Spot Price", f"${spot_price:.2f}")
            with m_col3:
                st.metric("Available Market Quotes", len(df))
            with m_col4:
                st.metric("Data Timestamp", pd.Timestamp.now().strftime('%H:%M:%S'))

            st.markdown("---")

            # Build and display surface from real market data
            try:
                _build_and_display_vol_surface(df, spot_price, risk_free_rate)
            except Exception as e:
                st.error(f"Error building volatility surface: {str(e)}")
    else:
        st.info("Configure parameters in the sidebar and click 'Fetch Market Data' to retrieve current option quotes from the market.")


@st.cache_data(ttl=3600)
def _fetch_real_market_data(ticker: str, strike_range: tuple, maturity_range: tuple, use_cache: bool = True) -> Optional[Dict]:
    """
    Fetch real European option quotes from Yahoo Finance.

    Returns dict with:
    - df: DataFrame of options
    - spot_price: current spot price
    - ticker: stock ticker
    - data_source: source label
    - fetch_time: timestamp
    - empty_data: whether data is empty after filtering
    """
    try:
        import yfinance as yf
    except ImportError:
        st.error("""
        **yfinance Module Not Installed**
        
        To enable real-time market data integration, please install yfinance:
        
        ```bash
        pip install yfinance
        ```
        
        Or if using Conda:
        ```bash
        conda install -c conda-forge yfinance
        ```
        
        After installation, restart the Streamlit application.
        """)
        return None

    try:
        # Fetch spot price
        stock = yf.Ticker(ticker)
        hist = stock.history(period='5d')

        if hist.empty:
            st.warning(f"⚠️ Could not fetch historical data for ticker: **{ticker}**")
            st.info("Please verify the ticker symbol and try again (e.g., AAPL, MSFT, TSLA)")
            return {"empty_data": True, "df": pd.DataFrame(), "spot_price": 0, "ticker": ticker, "data_source": "None", "fetch_time": "N/A"}

        spot_price = float(hist['Close'].iloc[-1])

        # Get available expirations
        try:
            expirations = stock.options
        except Exception as e:
            st.warning(f"Could not fetch option expirations for {ticker}: {str(e)}")
            return {"empty_data": True, "df": pd.DataFrame(), "spot_price": spot_price, "ticker": ticker, "data_source": "Yahoo Finance", "fetch_time": "N/A"}

        if not expirations:
            st.warning(f"No option expirations available for {ticker}")
            st.info("This underlying may not have actively traded options, or options data is not available from the data provider.")
            return {"empty_data": True, "df": pd.DataFrame(), "spot_price": spot_price, "ticker": ticker, "data_source": "Yahoo Finance", "fetch_time": "N/A"}

        # Collect all option chains
        all_options = []
        successful_expirations = 0

        for exp_date in expirations:
            try:
                opt_chain = stock.option_chain(exp_date)

                # Process calls
                calls = opt_chain.calls.copy()
                if not calls.empty:
                    calls['optionType'] = 'CALL'
                    calls['expiration'] = exp_date
                    calls['spot_price'] = spot_price
                    calls['dte'] = (pd.to_datetime(exp_date) - pd.Timestamp.now()).days
                    all_options.append(calls)

                # Process puts
                puts = opt_chain.puts.copy()
                if not puts.empty:
                    puts['optionType'] = 'PUT'
                    puts['expiration'] = exp_date
                    puts['spot_price'] = spot_price
                    puts['dte'] = (pd.to_datetime(exp_date) - pd.Timestamp.now()).days
                    all_options.append(puts)

                successful_expirations += 1

            except Exception as e:
                continue

        if not all_options:
            st.warning(f"⚠️ Could not retrieve option data for **{ticker}** from any expiration")
            return {"empty_data": True, "df": pd.DataFrame(), "spot_price": spot_price, "ticker": ticker, "data_source": "Yahoo Finance", "fetch_time": "N/A"}

        # Combine all options
        df = pd.concat(all_options, ignore_index=True)

        # Filter for valid data: positive prices, valid spreads, positive DTE
        df = df[
            (df['bid'] > 0) &
            (df['ask'] > 0) &
            (df['bid'] <= df['ask']) &
            (df['dte'] > 0) &
            (df['lastPrice'] > 0)
        ].copy()

        if df.empty:
            st.warning(f"⚠️ No valid option quotes found for **{ticker}** after data validation")
            return {"empty_data": True, "df": pd.DataFrame(), "spot_price": spot_price, "ticker": ticker, "data_source": "Yahoo Finance", "fetch_time": "N/A"}

        # Apply strike and maturity filters
        min_strike = spot_price * (strike_range[0] / 100.0)
        max_strike = spot_price * (strike_range[1] / 100.0)
        min_dte, max_dte = maturity_range

        df = df[
            (df['strike'] >= min_strike) &
            (df['strike'] <= max_strike) &
            (df['dte'] >= min_dte) &
            (df['dte'] <= max_dte)
        ]

        if df.empty:
            st.warning(f"No options matched your filter criteria for {ticker}")
            st.info("Adjust the moneyness range (% of spot) or time-to-expiration (days) filters and retry")
            return {"empty_data": True, "df": pd.DataFrame(), "spot_price": spot_price, "ticker": ticker, "data_source": "Yahoo Finance", "fetch_time": "N/A"}

        return {
            "df": df,
            "spot_price": spot_price,
            "ticker": ticker,
            "data_source": f"Yahoo Finance ({successful_expirations} expirations)",
            "fetch_time": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            "empty_data": False
        }

    except Exception as e:
        st.error(f"Error fetching market data from Yahoo Finance: {str(e)}")
        st.info("Please verify your internet connection and retry.")
        return None


def _build_and_display_vol_surface(df: pd.DataFrame, spot_price: float, risk_free_rate: float):
    """Build and display volatility surface from real market data."""
    try:
        from scipy.stats import norm
        from scipy.optimize import brentq
    except ImportError:
        st.error("The scipy module is required but not installed. Please run: pip install scipy")
        return

    df = df.copy()
    df['mid_price'] = (df['bid'] + df['ask']) / 2
    df['T'] = df['dte'] / 365.0

    # Black-Scholes functions
    def bs_call(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return None
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    def bs_put(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return None
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    # Implied volatility solver
    def implied_vol(price, S, K, T, r, is_call):
        if price <= 0 or T <= 0:
            return None
        try:
            def obj(sigma):
                if is_call:
                    theo = bs_call(S, K, T, r, sigma)
                else:
                    theo = bs_put(S, K, T, r, sigma)
                return (theo - price) if theo else float('inf')

            iv = brentq(obj, 0.001, 5.0, xtol=1e-4, maxiter=100)
            return max(0.001, min(iv, 5.0))
        except:
            return None

    # Compute implied volatilities from real market prices
    df['iv'] = df.apply(
        lambda row: implied_vol(
            row['mid_price'],
            row['spot_price'],
            row['strike'],
            row['T'],
            risk_free_rate,
            row['optionType'] == 'CALL'
        ),
        axis=1
    )

    # Remove invalid IVs
    df = df.dropna(subset=['iv'])
    df = df[df['iv'] > 0]

    if df.empty:
        st.warning("Could not compute implied volatilities from market prices")
        return

    # Display data coverage
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Valid IVs (Calls)", len(df[df['optionType'] == 'CALL']))
    with col2:
        st.metric("Valid IVs (Puts)", len(df[df['optionType'] == 'PUT']))

    st.markdown("---")

    # Build surfaces for calls and puts
    calls = df[df['optionType'] == 'CALL'].copy()
    puts = df[df['optionType'] == 'PUT'].copy()

    tab1, tab2 = st.tabs(["Call Options", "Put Options"])

    for tab, option_data, opt_type in [(tab1, calls, "CALL"), (tab2, puts, "PUT")]:
        with tab:
            if not option_data.empty:
                strikes = sorted(option_data['strike'].unique())
                maturities = sorted(option_data['T'].unique())

                # Build surface
                pivot = option_data.pivot_table(
                    values='iv',
                    index='strike',
                    columns='T',
                    aggfunc='mean'
                )

                # Interpolate missing points
                pivot = pivot.interpolate(method='linear', limit_direction='both', axis=1)
                pivot = pivot.interpolate(method='linear', limit_direction='both', axis=0)
                pivot = pivot.fillna(method='bfill').fillna(method='ffill')

                # 3D Surface plot
                try:
                    import plotly.graph_objects as go

                    X = np.array([m * 365 for m in pivot.columns])
                    Y = np.array(pivot.index)
                    Z = pivot.values

                    fig = go.Figure(data=[go.Surface(
                        x=X, y=Y, z=Z,
                        colorscale='Viridis',
                        colorbar=dict(title='Implied Vol', tickfont=dict(color='#f5f7fa')),
                        showscale=True
                    )])

                    fig.update_layout(
                        title=dict(text=f'Real Market Volatility Surface - {opt_type}'),
                        scene=dict(
                            xaxis=dict(title='Days to Expiration', tickfont=dict(color='#f5f7fa')),
                            yaxis=dict(title='Strike Price', tickfont=dict(color='#f5f7fa')),
                            zaxis=dict(title='Implied Volatility', tickfont=dict(color='#f5f7fa')),
                            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
                            bgcolor='#1a2332'
                        ),
                        paper_bgcolor='#0f1419',
                        plot_bgcolor='#1a2332',
                        height=700,
                        hovermode='closest'
                    )

                    fig = apply_theme(fig)

                    st.plotly_chart(fig, use_container_width=True)

                    # 2D slices
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### IV vs Strike")
                        mat_days = st.slider(
                            f"Select Maturity (days) - {opt_type}",
                            min_value=int(min(maturities) * 365),
                            max_value=int(max(maturities) * 365),
                            value=int(np.median([m * 365 for m in maturities])),
                            step=1,
                            key=f"mat_{opt_type}"
                        )

                        closest_T = min(maturities, key=lambda x: abs(x - mat_days / 365.0))
                        iv_slice = pivot[closest_T].dropna()

                        fig_strike = go.Figure()
                        fig_strike.add_trace(go.Scatter(
                            x=iv_slice.index,
                            y=iv_slice.values,
                            mode='lines+markers',
                            name='IV',
                            line=dict(color='#d4af7c', width=2),
                            marker=dict(size=8, color='#d4af7c')
                        ))
                        fig_strike.update_layout(
                            title=dict(text=f'{opt_type} IV vs Strike (T={mat_days} days)'),
                            xaxis=dict(title='Strike Price', tickfont=dict(color='#f5f7fa')),
                            yaxis=dict(title='Implied Volatility', tickfont=dict(color='#f5f7fa')),
                            hovermode='x unified',
                            height=500,
                            paper_bgcolor='#0f1419',
                            plot_bgcolor='#1a2332'
                        )
                        fig_strike = apply_theme(fig_strike)
                        st.plotly_chart(fig_strike, use_container_width=True)

                    with col2:
                        st.markdown("### IV vs Maturity")
                        strike_val = st.slider(
                            f"Select Strike - {opt_type}",
                            min_value=float(min(strikes)),
                            max_value=float(max(strikes)),
                            value=float(np.median(strikes)),
                            step=1.0,
                            key=f"strike_{opt_type}"
                        )

                        closest_K = min(strikes, key=lambda x: abs(x - strike_val))
                        iv_slice_mat = pivot.loc[closest_K].dropna()

                        fig_mat = go.Figure()
                        fig_mat.add_trace(go.Scatter(
                            x=np.array(iv_slice_mat.index) * 365,
                            y=iv_slice_mat.values,
                            mode='lines+markers',
                            name='IV',
                            line=dict(color='#d4af7c', width=2),
                            marker=dict(size=8, color='#d4af7c')
                        ))
                        fig_mat.update_layout(
                            title=dict(text=f'{opt_type} IV vs Maturity (K=${closest_K:.2f})'),
                            xaxis=dict(title='Days to Expiration', tickfont=dict(color='#f5f7fa')),
                            yaxis=dict(title='Implied Volatility', tickfont=dict(color='#f5f7fa')),
                            hovermode='x unified',
                            height=500,
                            paper_bgcolor='#0f1419',
                            plot_bgcolor='#1a2332'
                        )
                        fig_mat = apply_theme(fig_mat)
                        st.plotly_chart(fig_mat, use_container_width=True)

                except ImportError:
                    st.error("plotly not available. Please install: pip install plotly")

            else:
                st.info(f"No {opt_type} data available")

    # Data transparency
    with st.expander("📋 Raw Market Data"):
        st.dataframe(
            df[['strike', 'dte', 'optionType', 'bid', 'ask', 'lastPrice', 'iv']].sort_values(['optionType', 'dte', 'strike']),
            use_container_width=True,
            height=400
        )


def _generate_mock_vol_surface(ticker, strike_range, maturity_range):
                from scipy.optimize import brentq

                df = df.copy()
                df['mid_price'] = (df['bid'] + df['ask']) / 2
                df['T'] = df['dte'] / 365.0

                def bs_call(S, K, T, r, sigma):
                    if T <= 0 or sigma <= 0:
                        return None
                    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                    d2 = d1 - sigma * np.sqrt(T)
                    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

                def bs_put(S, K, T, r, sigma):
                    if T <= 0 or sigma <= 0:
                        return None
                    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                    d2 = d1 - sigma * np.sqrt(T)
                    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

                def implied_vol(price, S, K, T, r, is_call):
                    if price <= 0 or T <= 0:
                        return None
                    try:
                        def obj(sigma):
                            if is_call:
                                theo = bs_call(S, K, T, r, sigma)
                            else:
                                theo = bs_put(S, K, T, r, sigma)
                            return (theo - price) if theo else float('inf')

                        iv = brentq(obj, 0.001, 5.0, xtol=1e-4, maxiter=100)
                        return max(0.001, min(iv, 5.0))
                    except:
                        return None

                df['iv'] = df.apply(
                    lambda row: implied_vol(
                        row['mid_price'],
                        row['spot_price'],
                        row['strike'],
                        row['T'],
                        risk_free_rate,
                        row['optionType'] == 'CALL'
                    ),
                    axis=1
                )

                df = df.dropna(subset=['iv'])


def _generate_mock_vol_surface(ticker, strike_range, maturity_range):
    """Generate mock volatility surface data."""
    spot_price = 100.0
    strikes = np.linspace(spot_price * 0.7, spot_price * 1.3, 15)
    dtes = [7, 14, 30, 60, 90, 180]

    options = []
    today = datetime.now()

    for dte in dtes:
        T = dte / 365.0
        exp_date = (today + timedelta(days=dte)).strftime('%Y-%m-%d')

        for K in strikes:
            moneyness = np.log(spot_price / K)
            base_iv = 0.20
            smile_iv = base_iv + 0.05 * (moneyness ** 2)
            smile_iv = max(0.05, min(smile_iv, 1.0))

            from scipy.stats import norm
            d1 = (np.log(spot_price / K) + (0.05 + 0.5 * smile_iv ** 2) * T) / (smile_iv * np.sqrt(T))
            d2 = d1 - smile_iv * np.sqrt(T)

            call_price = spot_price * norm.cdf(d1) - K * np.exp(-0.05 * T) * norm.cdf(d2)
            put_price = K * np.exp(-0.05 * T) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)

            options.append({
                'strike': K,
                'expiration': exp_date,
                'optionType': 'CALL',
                'bid': call_price * 0.98,
                'ask': call_price * 1.02,
                'spot_price': spot_price,
                'dte': dte
            })
            options.append({
                'strike': K,
                'expiration': exp_date,
                'optionType': 'PUT',
                'bid': put_price * 0.98,
                'ask': put_price * 1.02,
                'spot_price': spot_price,
                'dte': dte
            })

    df = pd.DataFrame(options)
    st.session_state["vol_surface_data"] = {
        "df": df,
        "spot_price": spot_price,
        "ticker": ticker
    }
    st.success(f"Generated {len(df)} mock option quotes")


if __name__ == "__main__":
    run_pricing()
