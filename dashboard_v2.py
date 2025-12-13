"""
Professional 3-column options pricing dashboard with multiple models (GBM, Heston, Merton, Kou, G2++).
Layout: 1/4 custom payoff | 1/2 results/graphs/greeks | 1/4 history
"""
import streamlit as st
import numpy as np
import pandas as pd
import math
from typing import Dict, Any, Callable, Optional

st.set_page_config(page_title='Systematix Pro — Multi-Model Options Dashboard', layout='wide')

from simulator import simulate_paths
from payoffs import EuropeanCall, EuropeanPut
from pricing import price_mc_from_paths, price_heston, price_merton, price_kou, bs_price
from greeks import compute_greeks_mc
from viz import plot_terminal_histogram, plot_sample_paths
from payoff_utils import safe_compile_payoff

# ======================== Initialize Session State ========================
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'last_paths' not in st.session_state:
    st.session_state.last_paths = None

# ======================== Utility Functions ========================
def compile_custom_payoff(code_str: str) -> Optional[Callable]:
    """Safely compile custom payoff code."""
    try:
        fn = safe_compile_payoff(code_str)
        return fn
    except Exception as e:
        st.error(f'Payoff compilation failed: {e}')
        return None


def create_payoff_obj(payoff_type: str, strike: float, custom_fn: Optional[Callable] = None):
    """Create payoff object."""
    if custom_fn:
        # wrap custom function as payoff-like object
        class CustomPayoff:
            def __init__(self, fn):
                self.fn = fn
            def payoff(self, S):
                return self.fn(S)
        return CustomPayoff(custom_fn)
    if payoff_type == 'Call':
        return EuropeanCall(strike)
    else:
        return EuropeanPut(strike)


# ======================== Main Dashboard ========================
st.markdown('# Systematix Pro — Multi-Model Options Pricing Dashboard')
st.markdown('*Professional-grade pricing for European options across GBM, Heston, Merton, Kou, and G2++ models.*')

# Create 3-column layout
col_left, col_middle, col_right = st.columns([1, 2, 1])

# ======================== LEFT COLUMN: Custom Payoff ========================
with col_left:
    st.subheader('📝 Custom Payoff Function')
    st.markdown('*(Optional: if blank, use standard option)*')

    payoff_code = st.text_area(
        'Python code for payoff(S)',
        value='def custom_payoff(S):\n    return np.maximum(S - 100, 0)',
        height=150,
        key='payoff_code'
    )

    use_custom = st.checkbox('Use custom payoff', value=False)
    custom_payoff_fn = None
    if use_custom:
        custom_payoff_fn = compile_custom_payoff(payoff_code)

    st.divider()

    # Model selection
    st.subheader('🔧 Model Setup')
    model_key = st.selectbox(
        'Model',
        options=['gbm', 'heston', 'merton', 'kou', 'g2pp'],
        format_func=lambda x: {'gbm': 'GBM', 'heston': 'Heston', 'merton': 'Merton Jump',
                                'kou': 'Kou Jump', 'g2pp': 'G2++'}[x],
        key='model'
    )

    # Common parameters
    st.markdown('**Market Parameters**')
    S0 = st.number_input('Spot S0', value=100.0, step=1.0, key='S0')
    K = st.number_input('Strike K', value=100.0, step=1.0, key='K')
    r = st.number_input('Risk-free rate r', value=0.01, step=0.001, format='%.4f', key='r')
    sigma = st.number_input('Volatility σ', value=0.2, step=0.01, key='sigma')
    T = st.number_input('Time to maturity T (yrs)', value=1.0, step=0.1, key='T')

    st.markdown('**Simulation**')
    N = st.number_input('Time steps', value=12, step=1, key='N')
    n_paths = st.number_input('Monte Carlo paths', value=2000, step=100, key='n_paths')
    seed = st.number_input('RNG seed (0=random)', value=42, step=1, key='seed')

    # Model-specific parameters
    st.markdown('**Model Parameters**')
    if model_key == 'heston':
        st.markdown('*Heston SV Model*')
        v0 = st.number_input('v0 (initial variance)', value=0.04, step=0.001, key='v0')
        kappa = st.number_input('κ (reversion speed)', value=1.5, step=0.1, key='kappa')
        theta = st.number_input('θ (long-run variance)', value=0.04, step=0.001, key='theta')
        xi = st.number_input('ξ (vol of vol)', value=0.3, step=0.01, key='xi')
        rho = st.number_input('ρ (correlation)', value=-0.7, min_value=-1.0, max_value=1.0, step=0.1, key='rho')
        model_params = {'v0': v0, 'kappa': kappa, 'theta': theta, 'xi': xi, 'rho': rho}

    elif model_key == 'merton':
        st.markdown('*Merton Jump-Diffusion*')
        lambda_jump = st.number_input('λ (jump intensity)', value=0.1, step=0.01, key='lambda_jump')
        mu_jump = st.number_input('μ_J (jump mean)', value=0.0, step=0.01, key='mu_jump')
        sigma_jump = st.number_input('σ_J (jump vol)', value=0.3, step=0.01, key='sigma_jump')
        model_params = {'lambda_jump': lambda_jump, 'mu_jump': mu_jump, 'sigma_jump': sigma_jump}

    elif model_key == 'kou':
        st.markdown('*Kou Double Exponential Jump*')
        lambda_jump = st.number_input('λ (jump intensity)', value=0.1, step=0.01, key='lambda_jump_kou')
        p_up = st.number_input('p (up jump prob)', value=0.5, min_value=0.0, max_value=1.0, step=0.1, key='p_up')
        eta_up = st.number_input('η⁺ (up jump param)', value=1.0, step=0.1, key='eta_up')
        eta_down = st.number_input('η⁻ (down jump param)', value=2.0, step=0.1, key='eta_down')
        model_params = {'lambda_jump': lambda_jump, 'p_up': p_up, 'eta_up': eta_up, 'eta_down': eta_down}

    elif model_key == 'g2pp':
        st.markdown('*G2++ Interest Rate Model*')
        r0 = st.number_input('r0 (initial rate)', value=0.03, step=0.001, key='r0')
        a = st.number_input('a (mean reversion)', value=0.1, step=0.01, key='a')
        b = st.number_input('b (second factor reverting)', value=0.1, step=0.01, key='b')
        sigma_g2 = st.number_input('σ (vol factor 1)', value=0.015, step=0.001, key='sigma_g2')
        eta_g2 = st.number_input('η (vol factor 2)', value=0.025, step=0.001, key='eta_g2')
        rho_g2 = st.number_input('ρ_G2 (correlation)', value=0.8, min_value=-1.0, max_value=1.0, step=0.1, key='rho_g2')
        model_params = {'r0': r0, 'a': a, 'b': b, 'sigma': sigma_g2, 'eta': eta_g2, 'rho': rho_g2}

    else:  # gbm
        st.markdown('*Geometric Brownian Motion*')
        model_params = {}

    # Payoff type
    payoff_type = st.selectbox('Option type', ['Call', 'Put'], key='payoff_type')

    # Run button
    st.divider()
    run_button = st.button('▶️ RUN SIMULATION', use_container_width=True, key='run_btn')

# ======================== MIDDLE COLUMN: Results & Visualizations ========================
with col_middle:
    st.subheader('📊 Pricing Results & Analysis')

    # Placeholder for results
    result_placeholder = st.empty()
    metrics_placeholder = st.empty()
    viz1_placeholder = st.empty()
    viz2_placeholder = st.empty()
    greeks_placeholder = st.empty()

    if run_button:
        # Run simulation
        with st.spinner('Simulating paths and pricing...'):
            try:
                seed_val = None if seed == 0 else seed
                payoff_obj = create_payoff_obj(payoff_type, K, custom_payoff_fn)

                # Route to appropriate pricer
                if model_key == 'gbm':
                    from pricing import price_mc
                    result = price_mc(payoff_obj, S0, r, sigma, T, N, n_paths, seed=seed_val, antithetic=False)

                elif model_key == 'heston':
                    result = price_heston(payoff_obj, S0, r, model_params['v0'], model_params['kappa'],
                                          model_params['theta'], model_params['xi'], model_params['rho'],
                                          T, N, n_paths, seed=seed_val, antithetic=False)
                    # Also simulate paths for visualization
                    S, V = simulate_paths('heston', S0, r, model_params['v0'], model_params['kappa'],
                                          model_params['theta'], model_params['xi'], model_params['rho'],
                                          T, N, n_paths, seed=seed_val)
                    st.session_state.last_paths = S

                elif model_key == 'merton':
                    result = price_merton(payoff_obj, S0, r, sigma, T, N, n_paths,
                                          lambda_jump=model_params['lambda_jump'],
                                          mu_jump=model_params['mu_jump'],
                                          sigma_jump=model_params['sigma_jump'],
                                          seed=seed_val, antithetic=False)
                    S = simulate_paths('merton', S0, r, sigma, T, N, n_paths,
                                       lambda_jump=model_params['lambda_jump'],
                                       mu_jump=model_params['mu_jump'],
                                       sigma_jump=model_params['sigma_jump'],
                                       seed=seed_val)
                    st.session_state.last_paths = S

                elif model_key == 'kou':
                    result = price_kou(payoff_obj, S0, r, sigma, T, N, n_paths,
                                       lambda_jump=model_params['lambda_jump'],
                                       p_up=model_params['p_up'],
                                       eta_up=model_params['eta_up'],
                                       eta_down=model_params['eta_down'],
                                       seed=seed_val, antithetic=False)
                    S = simulate_paths('kou', S0, r, sigma, T, N, n_paths,
                                       lambda_jump=model_params['lambda_jump'],
                                       p_up=model_params['p_up'],
                                       eta_up=model_params['eta_up'],
                                       eta_down=model_params['eta_down'],
                                       seed=seed_val)
                    st.session_state.last_paths = S

                elif model_key == 'g2pp':
                    # G2++ returns rates, not equity prices; simplified for now
                    st.warning('G2++ pricing for equity options not fully implemented; showing rate simulation.')
                    r_paths = simulate_paths('g2pp', S0, model_params['r0'], model_params['a'], model_params['b'],
                                            model_params['sigma'], model_params['eta'], model_params['rho'],
                                            T, N, n_paths, seed=seed_val)
                    st.session_state.last_paths = r_paths
                    result = {'model': 'G2++', 'price': np.mean(r_paths[:, -1]), 'stderr': np.std(r_paths[:, -1]) / np.sqrt(n_paths), 'ci': (0, 0), 'n': n_paths}

                st.session_state.last_result = result

                # Display metrics
                with metrics_placeholder.container():
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric('Model', result.get('model', 'N/A'))
                    col_m2.metric('Price', f"{result['price']:.6f}")
                    col_m3.metric('Std Error', f"{result['stderr']:.6f}")
                    col_m4.metric('95% CI Low', f"{result['ci'][0]:.6f}")

                # Display paths if available
                if st.session_state.last_paths is not None and len(st.session_state.last_paths.shape) == 2:
                    with viz1_placeholder.container():
                        fig = plot_terminal_histogram(st.session_state.last_paths[:, -1])
                        st.plotly_chart(fig, use_container_width=True)

                    # Sample paths option
                    if st.checkbox('Show sample paths', value=False, key='show_paths'):
                        with viz2_placeholder.container():
                            fig2 = plot_sample_paths(st.session_state.last_paths, n_display=6, T=T)
                            st.plotly_chart(fig2, use_container_width=True)

                # Compute and display Greeks
                with greeks_placeholder.container():
                    if st.button('Compute Greeks', key='greeks_btn'):
                        try:
                            greeks_res = compute_greeks_mc(payoff_obj, model_key, S0, r, sigma, T, N, n_paths, seed=seed_val)
                            g_col1, g_col2, g_col3, g_col4, g_col5 = st.columns(5)
                            g_col1.metric('Δ (Delta)', f"{greeks_res.get('delta', float('nan')):.6f}")
                            g_col2.metric('Γ (Gamma)', f"{greeks_res.get('gamma', float('nan')):.6g}")
                            g_col3.metric('ν (Vega)', f"{greeks_res.get('vega', float('nan')):.6f}")
                            g_col4.metric('ρ (Rho)', f"{greeks_res.get('rho', float('nan')):.6f}")
                            g_col5.metric('Θ (Theta)', f"{greeks_res.get('theta', float('nan')):.6f}")
                            st.dataframe(pd.DataFrame([greeks_res]), use_container_width=True)
                        except Exception as e:
                            st.error(f'Greeks computation failed: {e}')

                # Add to history
                hist_entry = {
                    'Model': result.get('model', 'N/A'),
                    'Type': payoff_type if not use_custom else 'Custom',
                    'Price': result['price'],
                    'Stderr': result['stderr'],
                    'S0': S0,
                    'K': K,
                    'σ': sigma,
                    'T': T,
                }
                st.session_state.history.append(hist_entry)

            except Exception as e:
                st.error(f'Simulation failed: {e}')
                import traceback
                st.error(traceback.format_exc())

# ======================== RIGHT COLUMN: History ========================
with col_right:
    st.subheader('📈 Simulation History')

    if st.button('🗑️ Clear History', use_container_width=True, key='clear_history'):
        st.session_state.history = []

    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist.tail(10).reset_index(drop=True), use_container_width=True)

        # Download history
        csv = df_hist.to_csv(index=False)
        st.download_button(
            'Download history (CSV)',
            csv,
            file_name='pricing_history.csv',
            mime='text/csv',
            use_container_width=True,
            key='download_history'
        )
    else:
        st.info('No simulations yet. Run a simulation to populate history.')

    st.divider()
    st.subheader('ℹ️ About')
    st.markdown('''
    **Systematix Pro** pricing dashboard supports:
    - **GBM**: Geometric Brownian Motion
    - **Heston**: Stochastic volatility
    - **Merton**: Jump-diffusion
    - **Kou**: Double exponential jumps
    - **G2++**: Two-factor interest rates
    
    **Features**:
    - Custom payoff functions
    - Monte Carlo Greeks (CRN)
    - Terminal & path visualization
    - History tracking
    ''')

