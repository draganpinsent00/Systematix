"""ui/outputs.py
Components for rendering metrics, charts, and export controls.
Provides `render_analysis` used by the dashboard flow.
"""
from typing import Any, Optional
import streamlit as st
import pandas as pd
import numpy as np
from visualization.charts import plot_terminal_histogram, plot_sample_paths
from pricing import price_mc_from_paths
from payoffs import (
    EuropeanCall, EuropeanPut, AsianArithmeticCall, AsianArithmeticPut, AsianGeometricCall,
    AsianFloatingStrikeCall, AsianFloatingStrikePut,
    LookbackFloatingCall, LookbackFixedStrikeCall, LookbackFixedStrikePut, LookbackFloatingPut,
    DigitalCash, DigitalAsset, BarrierPayoff, BarrierDigitalCash, BarrierDigitalAsset, BasketCall
)


def _build_payoff(payoff_family: str, payoff_type: str, K: float, extra: dict = None):
    extra = extra or {}
    family = payoff_family.lower() if payoff_family else ''
    if family == 'european':
        return EuropeanCall(K) if payoff_type == 'Call' else EuropeanPut(K)
    if family == 'asian':
        if 'arithmetic' in payoff_type.lower():
            return AsianArithmeticCall(K) if 'call' in payoff_type.lower() else AsianArithmeticPut(K)
        if 'geometric' in payoff_type.lower():
            return AsianGeometricCall(K) if 'call' in payoff_type.lower() else AsianGeometricCall(K)
        if 'floating' in payoff_type.lower():
            return AsianFloatingStrikeCall() if 'call' in payoff_type.lower() else AsianFloatingStrikePut()
    if family == 'lookback':
        if 'floating' in payoff_type.lower() and 'call' in payoff_type.lower():
            return LookbackFloatingCall()
        if 'fixed' in payoff_type.lower():
            if 'call' in payoff_type.lower():
                return LookbackFixedStrikeCall(K)
            else:
                return LookbackFixedStrikePut(K)
        if 'floating' in payoff_type.lower() and 'put' in payoff_type.lower():
            return LookbackFloatingPut()
    if family == 'digital':
        if 'cash' in payoff_type.lower():
            return DigitalCash(K, cash=float(extra.get('cash_amount', 1.0)), at_maturity=bool(extra.get('cash_at_maturity', True)))
        else:
            return DigitalAsset(K, at_maturity=True)
    if family == 'barrier':
        low = extra.get('barrier_low', None)
        high = extra.get('barrier_high', None)
        bstyle = extra.get('barrier_payoff_style', 'Vanilla (European payoff)')
        if 'digital' in bstyle.lower() and 'cash' in bstyle.lower():
            return BarrierDigitalCash(K, barrier_low=low, barrier_high=high, barrier_type=payoff_type, cash=float(extra.get('cash_amount', 1.0)), at_maturity=True)
        if 'digital' in bstyle.lower() and 'asset' in bstyle.lower():
            return BarrierDigitalAsset(K, barrier_low=low, barrier_high=high, barrier_type=payoff_type, at_maturity=True)
        return BarrierPayoff(K, barrier_low=low, barrier_high=high, barrier_type=payoff_type, rebate=float(extra.get('rebate', 0.0)))
    if family == 'basket':
        return BasketCall(K)
    # fallback
    return EuropeanCall(K) if payoff_type == 'Call' else EuropeanPut(K)


def render_analysis(S_paths, K, r, T, payoff_family='European', payoff_type='Call', extra=None, show_sample_paths=False, n_sample_plot=5, price_disc: Optional[float]=None, stderr_disc: Optional[float]=None):
    """Render metrics, charts, pricing and export controls for given simulated paths.

    Calculates MC price and standard error by default using `price_mc_from_paths` and displays a 95% confidence interval.
    """
    if S_paths is None:
        st.info('No simulation results available. Run a simulation above.')
        return

    mean_ST = float(S_paths[:, -1].mean())
    median_ST = float(pd.Series(S_paths[:, -1]).median())

    # If price_disc and stderr_disc are provided use them; otherwise compute from paths
    if price_disc is None or stderr_disc is None:
        # Build payoff object
        payoff_obj = _build_payoff(payoff_family, payoff_type, K, extra=(extra or {}))

        # Price and stderr using path-based pricing
        try:
            mc_price_res = price_mc_from_paths(payoff_obj, S_paths, r, T, discount=False)
            price = float(mc_price_res['price'])
            stderr = float(mc_price_res['stderr'])

            # Discounted
            dfactor = float(np.exp(-r * T))
            price_disc = price * dfactor
            stderr_disc = stderr * dfactor
        except Exception:
            price_disc = float('nan')
            stderr_disc = float('nan')

    # 95% CI
    ci_low = price_disc - 1.96 * stderr_disc
    ci_high = price_disc + 1.96 * stderr_disc

    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Price (disc)', f'{price_disc:.6f}')
    c2.metric('Std err (disc)', f'{stderr_disc:.6f}')
    c3.metric('95% CI low', f'{ci_low:.6f}')
    c4.metric('95% CI high', f'{ci_high:.6f}')

    # Additional basic diagnostics
    m1, m2 = st.columns(2)
    m1.metric('Mean S_T', f'{mean_ST:.4f}')
    m2.metric('Median S_T', f'{median_ST:.4f}')

    # Visualizations
    fig_hist = plot_terminal_histogram(S_paths[:, -1])
    st.plotly_chart(fig_hist, use_container_width=True)

    if show_sample_paths:
        fig_paths = plot_sample_paths(S_paths, n_display=n_sample_plot, T=T)
        st.plotly_chart(fig_paths, use_container_width=True)

    with st.expander('Export simulated paths'):
        df_out = pd.DataFrame(S_paths)
        csv = df_out.to_csv(index=False)
        st.download_button('Download CSV', csv, file_name='simulated_paths.csv', mime='text/csv')
        st.dataframe(df_out.head())


__all__ = ['render_analysis']
