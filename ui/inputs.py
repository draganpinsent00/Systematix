"""ui/inputs.py
Sidebar and input components for the dashboard.
Provides a single function get_sidebar_inputs() that returns a dictionary of inputs.
"""
from typing import Dict, Any
import streamlit as st


def get_sidebar_inputs() -> Dict[str, Any]:
    """Render sidebar inputs and return a dictionary of collected values.

    This isolates UI input logic from overall dashboard behavior for easier testing and maintenance.
    """
    with st.sidebar:
        st.subheader('Parameters')
        model_key = st.selectbox('Model', ['gbm', 'heston', 'sabr', 'merton', 'kou', 'g2++'], index=0, key='model_key')
        S0 = st.number_input('Initial asset price (S0)', value=st.session_state.get('S0', 100.0), step=1.0, format='%.4g', key='S0')
        K = st.number_input('Strike (K)', value=st.session_state.get('K', 100.0), step=1.0, format='%.4g', key='K')
        r = st.number_input('Risk-free rate (r)', value=st.session_state.get('r', 0.05), step=0.01, format='%.4g', help='Annual continuously compounded rate', key='r')
        sigma = st.number_input('Volatility (sigma)', value=st.session_state.get('sigma', 0.2), step=0.01, format='%.4g', key='sigma')
        T = st.number_input('Time to maturity (T, years)', value=st.session_state.get('T', 1.0), step=0.01, format='%.4g', key='T')
        N = st.number_input('Time steps (N)', value=st.session_state.get('N', 100), step=1, help='Number of discrete time steps over the maturity', key='N')
        n_paths = st.number_input('Monte Carlo paths (number of simulated paths)', value=st.session_state.get('n_paths', 2000), step=100, help='Total number of simulated sample paths.', key='n_paths')

        # Model-specific parameters
        st.subheader('Model-specific parameters')
        model_params = {}
        if model_key == 'heston':
            v0 = st.number_input('v0 (variance initial)', value=st.session_state.get('model_params', {}).get('v0', 0.04), step=0.01, key='v0')
            kappa = st.number_input('kappa', value=1.5, step=0.1)
            theta_h = st.number_input('theta (long-run var)', value=0.04, step=0.01)
            xi = st.number_input('xi (vol of vol)', value=0.3, step=0.01)
            rho = st.number_input('rho (corr)', value=-0.7, step=0.01)
            model_params.update({'v0': v0, 'kappa': kappa, 'theta': theta_h, 'xi': xi, 'rho': rho})
        elif model_key == 'g2++':
            a1 = st.number_input('a1 (mean reversion 1)', value=0.03, step=0.005)
            a2 = st.number_input('a2 (mean reversion 2)', value=0.02, step=0.005)
            sigma1 = st.number_input('sigma1', value=0.015, step=0.001)
            sigma2 = st.number_input('sigma2', value=0.01, step=0.001)
            rho_g2 = st.number_input('rho (factors corr)', value=0.5, step=0.05)
            x0 = st.number_input('x0 (initial factor 1)', value=0.0, step=0.001)
            y0 = st.number_input('y0 (initial factor 2)', value=0.0, step=0.001)
            model_params.update({'a1': a1, 'a2': a2, 'sigma1': sigma1, 'sigma2': sigma2, 'rho': rho_g2, 'x0': x0, 'y0': y0})
        elif model_key == 'merton':
            lamb = st.number_input('lambda (jump intensity)', value=0.5, step=0.1)
            mu_j = st.number_input('mu_j (jump mean)', value=-0.1, step=0.01)
            sigma_j = st.number_input('sigma_j (jump std)', value=0.25, step=0.01)
            model_params.update({'lamb': lamb, 'mu_j': mu_j, 'sigma_j': sigma_j})
        elif model_key == 'kou':
            lamb_k = st.number_input('lambda (jump intensity)', value=0.5, step=0.1)
            p_up = st.number_input('p_up (prob. positive jump)', value=0.3, step=0.05)
            eta1 = st.number_input('eta1 (up decay)', value=1.5, step=0.1)
            eta2 = st.number_input('eta2 (down decay)', value=0.5, step=0.1)
            model_params.update({'lamb': lamb_k, 'p_up': p_up, 'eta1': eta1, 'eta2': eta2})

        # Payoff controls
        st.subheader('Payoff')
        payoff_family = st.selectbox('Option family', ['European', 'Asian', 'Lookback', 'Digital', 'Barrier', 'Basket'], index=0)
        if payoff_family == 'European':
            payoff_type = st.selectbox('Payoff type', ['Call', 'Put'], index=0, key='payoff_type')
        elif payoff_family == 'Asian':
            payoff_type = st.selectbox('Payoff type', ['Arithmetic Call', 'Arithmetic Put', 'Geometric Call', 'Geometric Put', 'Floating Strike Call', 'Floating Strike Put'], index=0)
        elif payoff_family == 'Lookback':
            payoff_type = st.selectbox('Payoff type', ['Floating Call', 'Fixed Strike Call', 'Fixed Strike Put', 'Floating Put'], index=0)
        elif payoff_family == 'Digital':
            payoff_type = st.selectbox('Payoff type', ['Cash-or-nothing', 'Asset-or-nothing'], index=0, key='payoff_type')
            cash_amount = st.number_input('Cash payoff amount (for cash-or-nothing)', value=st.session_state.get('cash_amount', 1.0), step=0.01, format='%.4g', key='cash_amount')
            cash_at_maturity = st.checkbox('Pay at maturity', value=st.session_state.get('cash_at_maturity', True), key='cash_at_maturity')
        elif payoff_family == 'Barrier':
            payoff_type = st.selectbox('Barrier type', ['up-and-out', 'down-and-out', 'up-and-in', 'down-and-in', 'double-knock-out', 'double-knock-in'], index=0, key='payoff_type')
            barrier_low = st.number_input('Barrier low (leave 0 if not used)', value=st.session_state.get('barrier_low', 0.0), step=0.01, key='barrier_low')
            barrier_high = st.number_input('Barrier high (leave 0 if not used)', value=st.session_state.get('barrier_high', 0.0), step=0.01, key='barrier_high')
            rebate = st.number_input('Rebate on knock-out', value=st.session_state.get('rebate', 0.0), step=0.01, key='rebate')
            barrier_payoff_style = st.selectbox('Barrier payoff style', ['Vanilla (European payoff)', 'Digital - cash', 'Digital - asset'], index=0, key='barrier_payoff_style')
        else:
            payoff_type = st.selectbox('Payoff type', ['Call', 'Put'], index=0)

    inputs = {
         'model_key': model_key,
         'S0': S0,
         'K': K,
         'r': r,
         'sigma': sigma,
         'T': T,
         'N': N,
         'n_paths': n_paths,
         'model_params': model_params,
         'payoff_family': payoff_family,
         'payoff_type': payoff_type,
     }
    # optional fields
    if 'cash_amount' in locals():
        inputs['cash_amount'] = cash_amount
        inputs['cash_at_maturity'] = cash_at_maturity
    if 'barrier_low' in locals():
        inputs['barrier_low'] = barrier_low
        inputs['barrier_high'] = barrier_high
        inputs['rebate'] = rebate
        inputs['barrier_payoff_style'] = barrier_payoff_style

    return inputs
