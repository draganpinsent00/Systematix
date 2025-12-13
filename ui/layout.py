"""layout.py
High-level Streamlit layout helpers. For now, dashboard.py still holds the UI; this file will be the target for migration.
"""
import streamlit as st
import numpy as np
import pandas as pd
from pricing import price_mc_from_paths
from ui.inputs import get_sidebar_inputs
from ui.outputs import render_analysis
from models.monte_carlo import run_batched_simulation_adapter
from payoff_utils import safe_compile_payoff, run_payoff_in_sandbox
from history_utils import make_history_entry


def _safe_rerun():
    """Call st.experimental_rerun if available; otherwise no-op.

    Some Streamlit versions lack experimental_rerun. We guard against that.
    """
    try:
        func = getattr(st, 'experimental_rerun')
        func()
    except Exception:
        # Either attribute missing or rerun not allowed; ignore.
        return


def render_dashboard():
    """Render the whole dashboard using modular components.

    This function mirrors the previous `dashboard.py` behaviour but calls into
    ui.inputs and ui.outputs for separation of concerns.
    """
    # If a history entry was requested to be loaded, apply its params to session_state
    # before widgets are created so Streamlit widgets can pick up the values.
    if 'pending_load' in st.session_state:
        pending = st.session_state.pop('pending_load')
        for k, v in pending.items():
            # allow overriding or setting before widget instantiation
            st.session_state[k] = v
        # if pending paths were provided, set them too
        if 'pending_paths' in st.session_state:
            st.session_state['S_paths'] = st.session_state.pop('pending_paths')
        # clear any previously compiled custom payoff when loading a saved configuration
        st.session_state.pop('custom_payoff_compiled', None)
        st.session_state.pop('custom_payoff_code', None)

    # (no pre-render auto-run) Inputs are created after processing pending_load
    inputs = get_sidebar_inputs()

    model_key = inputs['model_key']
    S0 = inputs['S0']
    K = inputs['K']
    r = inputs['r']
    sigma = inputs['sigma']
    T = inputs['T']
    N = inputs['N']
    n_paths = inputs['n_paths']
    model_params = inputs['model_params']
    payoff_family = inputs['payoff_family']
    payoff_type = inputs['payoff_type']

    # Advanced inline box
    st.title('Options Pricing Dashboard — Professional')
    col1, col2, col3 = st.columns([1, 2, 1])

    # Ensure history exists
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    with col1:
        run_now_btn = st.button('Execute simulation')
        # treat run_now true if button clicked or a compile-trigger was set
        run_now = bool(run_now_btn)
        show_sample_paths = st.checkbox('Display sample paths')
        n_sample_plot = st.number_input('Sample paths to plot', value=5, min_value=1)
        with st.expander('Advanced (helpers & custom payoff)', expanded=False):
            # Show available runtime variables (do not show helper functions here)
            st.markdown('''
            Available runtime variables for custom payoff functions (vectorized):
            - paths: numpy array of shape (n_paths, steps+1) containing simulated prices
            - ST: terminal price (paths[:, -1])
            - S0: initial asset price
            - K: strike
            - r: risk-free rate
            - sigma: volatility
            - T: maturity (years)
            - N: number of time steps
            - dt: time step size
            - times: numpy array of time points
            ''')
            # Provide an empty editor for custom payoff; users must explicitly compile
            custom_code = st.text_area('Custom payoff code (define function custom_payoff(paths))', height=150, value='')
            # show a minimal example for custom payoff functions
            st.markdown('''
            Example custom payoff (enter and click Compile):
            ```python
            def custom_payoff(paths):
                import numpy as np
                ST = paths[:, -1]
                # simple European call
                return np.maximum(ST - K, 0.0)
            ```
            ''')
            if st.button('Compile custom payoff', key='compile_custom'):
                # require non-empty code
                if not custom_code or not custom_code.strip():
                    st.error('Please enter custom payoff code before compiling')
                else:
                    try:
                        fn = safe_compile_payoff(custom_code)
                        # mark compiled flag and store code
                        st.session_state['custom_payoff_code'] = custom_code
                        st.session_state['custom_payoff_compiled'] = True
                        st.success('Custom payoff compiled successfully — running simulation now')
                        # Immediately run simulation with compiled payoff
                        try:
                            S_all = run_batched_simulation_adapter(model_key, model_params, S0, r, sigma, T, int(N), int(n_paths), int(500), int(123), False, 'pseudo-random', 'normal', {}, False)
                            st.session_state['S_paths'] = S_all
                            if S_all is not None:
                                dt = float(T) / float(N) if N else None
                                payoffs = run_payoff_in_sandbox(custom_code, S_all, context={'S0': S0, 'K': K, 'r': r, 'sigma': sigma, 'T': T, 'N': N, 'dt': dt})
                                price = float(np.mean(payoffs))
                                stderr = float(np.std(payoffs, ddof=1) / (payoffs.shape[0] ** 0.5))
                                dfactor = np.exp(-r * T)
                                price_disc = price * dfactor
                                stderr_disc = stderr * dfactor
                            else:
                                price_disc = float('nan')
                                stderr_disc = float('nan')
                            # record history
                            entry = make_history_entry(model_key, {'S0': S0, 'K': K, 'r': r, 'sigma': sigma, 'T': T, 'N': N, 'n_paths': n_paths, 'payoff_family': payoff_family, 'payoff_type': payoff_type, 'compiled_custom': True}, price_disc, stderr_disc, S_all)
                            st.session_state.setdefault('history', []).append(entry)
                            st.session_state['last_price_disc'] = price_disc
                            st.session_state['last_stderr_disc'] = stderr_disc
                            # mark that we should display the custom-payoff result (takes precedence)
                            st.session_state['display_custom_result'] = True
                            # Render analysis immediately into the middle column so result shows even if rerun is unavailable
                            try:
                                with col2:
                                    st.subheader('Custom payoff result')
                                    render_analysis(S_all, K, r, T, payoff_family=payoff_family, payoff_type=payoff_type, show_sample_paths=show_sample_paths, n_sample_plot=n_sample_plot, price_disc=price_disc, stderr_disc=stderr_disc)
                            except Exception:
                                # If rendering now fails, fallback to scheduling a rerun
                                _safe_rerun()
                            st.success('Simulation with custom payoff completed')
                            # Stop further rendering to avoid later branches overwriting the custom-payoff display
                            return
                        except Exception as e:
                            import traceback
                            tb = traceback.format_exc()
                            st.error(f'Error running simulation with custom payoff: {e}')
                            st.text(tb)
                    except Exception as e:
                        # clear any previous compiled flags
                        st.session_state.pop('custom_payoff_code', None)
                        st.session_state.pop('custom_payoff_compiled', None)
                        st.error(f'Compilation error: {e}')

    # Run simulation
    if run_now:
        S_all = run_batched_simulation_adapter(model_key, model_params, S0, r, sigma, T, int(N), int(n_paths), int(500), int(123), False, 'pseudo-random', 'normal', {}, False)
        st.session_state['S_paths'] = S_all
        # manual run should clear any custom-payoff display preference
        st.session_state.pop('display_custom_result', None)
        # pricing and history
        # Only use custom payoff if it was successfully compiled in this session
        custom_code = st.session_state.get('custom_payoff_code', None) if st.session_state.get('custom_payoff_compiled', False) else None
        price_disc = float('nan')
        stderr_disc = float('nan')
        if S_all is not None:
            try:
                if custom_code:
                    dt = float(T) / float(N) if N else None
                    payoffs = run_payoff_in_sandbox(custom_code, S_all, context={'S0': S0, 'K': K, 'r': r, 'sigma': sigma, 'T': T, 'N': N, 'dt': dt})
                    price = float(np.mean(payoffs))
                    stderr = float(np.std(payoffs, ddof=1) / (payoffs.shape[0] ** 0.5))
                else:
                    payoff_obj = _build_payoff = None  # fallback to simple terminal call if no mapping
                    # attempt to price European call/put using terminal payoff
                    ST = S_all[:, -1]
                    if payoff_family.lower() == 'european':
                        if payoff_type == 'Call':
                            res_vals = np.maximum(ST - K, 0.0)
                        else:
                            res_vals = np.maximum(K - ST, 0.0)
                        price = float(res_vals.mean())
                        stderr = float(res_vals.std(ddof=1) / (res_vals.shape[0] ** 0.5))
                    else:
                        # Try to delegate to ui.outputs pricing if possible
                        from ui.outputs import _build_payoff
                        p_obj = _build_payoff(payoff_family, payoff_type, K, extra={'cash_amount': st.session_state.get('cash_amount', 1.0), 'cash_at_maturity': st.session_state.get('cash_at_maturity', True), 'barrier_low': st.session_state.get('barrier_low', None), 'barrier_high': st.session_state.get('barrier_high', None), 'rebate': st.session_state.get('rebate', 0.0), 'barrier_payoff_style': st.session_state.get('barrier_payoff_style', 'Vanilla (European payoff)')})
                        res = price_mc_from_paths(p_obj, S_all, r, T, discount=False)
                        price = float(res['price'])
                        stderr = float(res['stderr'])
                # discount
                dfactor = np.exp(-r * T)
                price_disc = price * dfactor
                stderr_disc = stderr * dfactor
            except Exception as e:
                st.error(f'Error computing price: {e}')
        # record into history
        entry = make_history_entry(model_key, {'S0': S0, 'K': K, 'r': r, 'sigma': sigma, 'T': T, 'N': N, 'n_paths': n_paths, 'payoff_family': payoff_family, 'payoff_type': payoff_type}, price_disc, stderr_disc, S_all)
        st.session_state['history'].append(entry)
        # store last price and stderr for display in outputs
        st.session_state['last_price_disc'] = price_disc
        st.session_state['last_stderr_disc'] = stderr_disc

    # Middle: analysis
    with col2:
        S_paths = st.session_state.get('S_paths', None)
        # If a custom payoff was compiled, executed, and explicitly marked for display, prefer the computed price/stderr
        if st.session_state.get('display_custom_result', False) and 'last_price_disc' in st.session_state:
            render_analysis(S_paths, K, r, T, show_sample_paths=show_sample_paths, n_sample_plot=n_sample_plot, price_disc=st.session_state.get('last_price_disc'), stderr_disc=st.session_state.get('last_stderr_disc'))
        else:
            render_analysis(S_paths, K, r, T, show_sample_paths=show_sample_paths, n_sample_plot=n_sample_plot)

    # Right: history
    with col3:
        st.subheader('History')
        history = st.session_state.get('history', [])
        if not history:
            st.write('No history yet')
        else:
            # show last-first
            for i, entry in enumerate(reversed(history)):
                idx = len(history) - 1 - i
                ts = entry.get('timestamp', '')
                try:
                    from datetime import datetime
                    dt_obj = datetime.fromisoformat(ts)
                    date_str = dt_obj.strftime('%Y-%m-%d')
                    time_str = dt_obj.strftime('%H:%M:%S')
                except Exception:
                    date_str = ts
                    time_str = ''
                params = entry.get('params', {})
                opt_family = params.get('payoff_family', payoff_family)
                opt_type = params.get('payoff_type', payoff_type)
                title = f"{date_str}, {time_str}, {opt_family}, {opt_type}, {entry.get('model', '')}"
                with st.expander(title, expanded=False):
                    st.write('Parameters:')
                    st.json(params)
                    st.write(f"Price (disc): {entry.get('price_disc', float('nan')):.6f}")
                    st.write(f"Std err (disc): {entry.get('stderr_disc', float('nan')):.6f}")
                    if st.button('Load into Analysis', key=f'load_{idx}'):
                        # Defer applying params until next run to avoid modifying widget-backed keys
                        st.session_state['pending_load'] = params
                        st.session_state['pending_paths'] = entry.get('paths')
                        _safe_rerun()
                    csv = pd.DataFrame(entry.get('paths')).to_csv(index=False)
                    st.download_button('Download CSV', csv, file_name=f"sim_paths_{idx}.csv", key=f'dl_{idx}')
                    if st.button('Delete entry', key=f'del_{idx}'):
                        # remove actual entry index
                        history_idx = len(history) - 1 - i
                        st.session_state['history'].pop(history_idx)
                        _safe_rerun()

__all__ = ['render_dashboard']
