F"""
Volatility Surface Analysis Page

Pulls European option quotes from multiple data sources,
computes implied volatilities, and visualizes a vol surface.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import data sources (graceful fallback)
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

try:
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from scipy.optimize import brentq
    HAS_BRENTQ = True
except ImportError:
    HAS_BRENTQ = False


# ============================================================================
# BLACK-SCHOLES IMPLIED VOLATILITY CALCULATOR
# ============================================================================

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if not HAS_SCIPY or T <= 0 or sigma <= 0:
        return None
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if not HAS_SCIPY or T <= 0 or sigma <= 0:
        return None
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price


def implied_volatility(option_price: float, S: float, K: float, T: float,
                       r: float, is_call: bool, initial_guess: float = 0.3) -> Optional[float]:
    """
    Compute implied volatility using Brent's method.

    Args:
        option_price: Observed market price of the option
        S: Current spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        is_call: True for call, False for put
        initial_guess: Initial IV estimate

    Returns:
        Implied volatility, or None if calculation fails
    """
    if not HAS_BRENTQ or not HAS_SCIPY:
        return None

    if option_price <= 0 or T <= 0:
        return None

    def objective(sigma):
        if is_call:
            theo_price = black_scholes_call(S, K, T, r, sigma)
        else:
            theo_price = black_scholes_put(S, K, T, r, sigma)

        if theo_price is None:
            return float('inf')
        return theo_price - option_price

    try:
        # Intrinsic value check
        if is_call:
            intrinsic = max(S - K, 0)
        else:
            intrinsic = max(K - S, 0)

        if option_price < intrinsic * 0.99:
            return None

        # Brent's method
        iv = brentq(objective, 0.001, 5.0, xtol=1e-4, maxiter=100)
        return max(0.001, min(iv, 5.0))  # Clamp to reasonable range
    except Exception:
        return None


# ============================================================================
# DATA SOURCE INTEGRATIONS
# ============================================================================

def fetch_yfinance_options(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch European-style option quotes from Yahoo Finance.

    Returns DataFrame with columns:
    [contract_symbol, strike, expiration, last_price, bid, ask,
     optionType, implied_volatility_market, dte, spot_price]
    """
    if not HAS_YFINANCE:
        return None

    try:
        stock = yf.Ticker(ticker)

        # Get current spot price
        hist = stock.history(period='1d')
        if hist.empty:
            return None
        spot_price = hist['Close'].iloc[-1]

        # Get option chain
        expirations = stock.options
        if not expirations:
            return None

        all_options = []

        for exp_date in expirations[:10]:  # Limit to first 10 expirations
            try:
                opt_chain = stock.option_chain(exp_date)

                exp_dt = pd.to_datetime(exp_date)
                today = pd.Timestamp.now()
                dte = (exp_dt - today).days

                # Calls
                calls = opt_chain.calls.copy()
                calls['optionType'] = 'CALL'
                calls['expiration'] = exp_date
                calls['dte'] = dte
                calls['spot_price'] = spot_price

                # Puts
                puts = opt_chain.puts.copy()
                puts['optionType'] = 'PUT'
                puts['expiration'] = exp_date
                puts['dte'] = dte
                puts['spot_price'] = spot_price

                all_options.append(calls)
                all_options.append(puts)
            except Exception:
                continue

        if not all_options:
            return None

        df = pd.concat(all_options, ignore_index=True)

        # Filter for reasonable data
        df = df[(df['bid'] > 0) & (df['ask'] > 0)]
        df = df[(df['bid'] <= df['ask'])]
        df = df[df['dte'] > 0]

        return df

    except Exception as e:
        st.warning(f"Yahoo Finance error: {e}")
        return None


def fetch_mock_options(ticker: str, spot_price: Optional[float] = None) -> pd.DataFrame:
    """
    Generate mock European option data for demonstration.

    Returns realistic synthetic option quotes.
    """
    if spot_price is None:
        spot_price = 100.0

    # Generate strikes and maturities
    strikes = np.linspace(spot_price * 0.7, spot_price * 1.3, 15)
    expirations = [7, 14, 30, 60, 90, 180]  # Days to expiration

    options_data = []
    today = datetime.now()

    for dte in expirations:
        T = dte / 365.0
        exp_date = (today + timedelta(days=dte)).strftime('%Y-%m-%d')

        for K in strikes:
            # Generate synthetic IVs (smile effect)
            moneyness = np.log(spot_price / K)
            base_iv = 0.20
            smile_iv = base_iv + 0.05 * (moneyness ** 2)
            smile_iv = max(0.05, min(smile_iv, 1.0))

            # Call price
            call_price = black_scholes_call(spot_price, K, T, 0.05, smile_iv)
            call_bid = call_price * 0.98
            call_ask = call_price * 1.02

            # Put price
            put_price = black_scholes_put(spot_price, K, T, 0.05, smile_iv)
            put_bid = put_price * 0.98
            put_ask = put_price * 1.02

            options_data.append({
                'strike': K,
                'expiration': exp_date,
                'optionType': 'CALL',
                'bid': call_bid,
                'ask': call_ask,
                'lastPrice': call_price,
                'dte': dte,
                'spot_price': spot_price,
                'source': 'Mock Data'
            })

            options_data.append({
                'strike': K,
                'expiration': exp_date,
                'optionType': 'PUT',
                'bid': put_bid,
                'ask': put_ask,
                'lastPrice': put_price,
                'dte': dte,
                'spot_price': spot_price,
                'source': 'Mock Data'
            })

    return pd.DataFrame(options_data)


# ============================================================================
# VOLATILITY SURFACE CONSTRUCTION
# ============================================================================

def build_volatility_surface(df: pd.DataFrame, risk_free_rate: float) -> Dict:
    """
    Build volatility surface from option quotes.

    Returns dict with:
    - surface_grid: 2D array of IVs
    - strikes: unique strikes
    - maturities: unique maturities
    - iv_data: DataFrame with computed IVs
    """
    if df.empty:
        return None

    # Compute implied volatilities
    df = df.copy()
    df['mid_price'] = (df['bid'] + df['ask']) / 2
    df['T'] = df['dte'] / 365.0

    def compute_iv(row):
        return implied_volatility(
            row['mid_price'],
            row['spot_price'],
            row['strike'],
            row['T'],
            risk_free_rate,
            row['optionType'] == 'CALL'
        )

    df['iv'] = df.apply(compute_iv, axis=1)

    # Remove invalid IVs
    df = df.dropna(subset=['iv'])
    df = df[df['iv'] > 0]

    if df.empty:
        return None

    # Separate calls and puts
    calls = df[df['optionType'] == 'CALL'].copy()
    puts = df[df['optionType'] == 'PUT'].copy()

    # Build grid for each
    results = {}

    for option_type, data in [('CALL', calls), ('PUT', puts)]:
        if data.empty:
            continue

        strikes = sorted(data['strike'].unique())
        maturities = sorted(data['T'].unique())

        # Create pivot table
        pivot = data.pivot_table(
            values='iv',
            index='strike',
            columns='T',
            aggfunc='mean'
        )

        # Interpolate missing values
        pivot = pivot.interpolate(method='linear', limit_direction='both')
        pivot = pivot.fillna(method='bfill').fillna(method='ffill')

        results[option_type] = {
            'strikes': strikes,
            'maturities': maturities,
            'pivot': pivot,
            'data': data
        }

    return {
        'calls': results.get('CALL'),
        'puts': results.get('PUT'),
        'all_iv_data': df
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_volatility_surface_3d(surface_data: Dict, option_type: str = 'CALL') -> go.Figure:
    """Create 3D volatility surface plot."""

    data = surface_data.get(option_type)
    if data is None:
        return None

    X = data['maturities'] * 365  # Convert to days
    Y = data['strikes']
    Z = data['pivot'].values

    # Create surface with minimal colorbar (no font properties allowed)
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        colorbar=dict(title='Implied Vol')
    )])

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f'Volatility Surface - {option_type}',
            font=dict(color='white', size=18),
            x=0.5
        ),
        paper_bgcolor="#0e1117",
        margin=dict(t=80)
    )

    return fig


def plot_iv_by_strike(surface_data: Dict, maturity_days: float,
                      option_type: str = 'CALL') -> go.Figure:
    """IV vs Strike at fixed maturity."""

    data = surface_data.get(option_type)
    if data is None:
        return None

    # Find closest maturity
    T = maturity_days / 365.0
    closest_T = min(data['maturities'], key=lambda x: abs(x - T))

    iv_slice = data['pivot'][closest_T].dropna()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=iv_slice.index,
        y=iv_slice.values,
        mode='lines+markers',
        name='IV',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title=f'{option_type} IV vs Strike (T={maturity_days:.0f} days)',
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility',
        hovermode='x unified',
        height=500
    )

    fig.update_layout(title=dict(font=dict(color='white')))

    return fig


def plot_iv_by_maturity(surface_data: Dict, strike_price: float,
                        option_type: str = 'CALL') -> go.Figure:
    """IV vs Maturity at fixed strike."""

    data = surface_data.get(option_type)
    if data is None:
        return None

    # Find closest strike
    closest_K = min(data['strikes'], key=lambda x: abs(x - strike_price))

    iv_slice = data['pivot'].loc[closest_K].dropna()
    maturities_days = np.array(iv_slice.index) * 365

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=maturities_days,
        y=iv_slice.values,
        mode='lines+markers',
        name='IV',
        line=dict(color='green', width=2),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title=f'{option_type} IV vs Maturity (K=${closest_K:.2f})',
        xaxis_title='Days to Expiration',
        yaxis_title='Implied Volatility',
        hovermode='x unified',
        height=500
    )

    fig.update_layout(title=dict(font=dict(color='white')))

    return fig


# ============================================================================
# STREAMLIT PAGE
# ============================================================================

def main():
    st.set_page_config(
        page_title="Volatility Surface Analysis",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📈 Volatility Surface Analysis")
    st.markdown("Extract European option quotes and construct volatility surfaces")

    # ========================================================================
    # SIDEBAR: INPUT CONTROLS
    # ========================================================================

    with st.sidebar:
        st.markdown("## Configuration")

        # Ticker input
        ticker = st.text_input(
            "Stock Ticker",
            value="AAPL",
            help="Enter a valid stock ticker symbol"
        ).upper()

        # Risk-free rate
        risk_free_rate = st.slider(
            "Risk-Free Rate",
            min_value=0.0,
            max_value=0.1,
            value=0.05,
            step=0.001,
            help="Annual risk-free rate (continuous compounding)"
        )

        st.markdown("### Data Sources")

        # Data source selection
        use_yfinance = st.checkbox(
            "Yahoo Finance",
            value=HAS_YFINANCE,
            disabled=not HAS_YFINANCE,
            help="Pull live option quotes from Yahoo Finance"
        )

        use_mock = st.checkbox(
            "Mock Data (Demo)",
            value=True,
            help="Generate synthetic European option data for demo purposes"
        )

        st.markdown("### Filters")

        # Strike range
        strike_range = st.slider(
            "Strike Price Range (%)",
            min_value=50,
            max_value=150,
            value=(80, 120),
            step=5,
            help="Filter strikes relative to spot price"
        )

        # Maturity range
        maturity_range = st.slider(
            "Days to Expiration",
            min_value=1,
            max_value=365,
            value=(7, 180),
            step=7,
            help="Filter options by maturity range"
        )

        st.markdown("### Actions")

        fetch_button = st.button("🔄 Fetch & Analyze", use_container_width=True)

        st.markdown("---")
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ========================================================================
    # MAIN PANEL: DATA FETCHING & ANALYSIS
    # ========================================================================

    if fetch_button or 'vol_surface_data' not in st.session_state:
        with st.spinner("Fetching option data and building volatility surface..."):

            all_options = []
            sources_used = []

            # Fetch from Yahoo Finance
            if use_yfinance and HAS_YFINANCE:
                try:
                    yf_data = fetch_yfinance_options(ticker)
                    if yf_data is not None:
                        yf_data['source'] = 'Yahoo Finance'
                        all_options.append(yf_data)
                        sources_used.append('Yahoo Finance')
                        st.success(f"✓ Yahoo Finance: {len(yf_data)} quotes")
                except Exception as e:
                    st.warning(f"Yahoo Finance unavailable: {e}")

            # Fetch mock data
            if use_mock:
                try:
                    mock_data = fetch_mock_options(ticker)
                    all_options.append(mock_data)
                    sources_used.append('Mock Data (Synthetic)')
                    st.info(f"✓ Mock Data: {len(mock_data)} quotes")
                except Exception as e:
                    st.warning(f"Mock data error: {e}")

            # Combine and filter
            if all_options:
                df = pd.concat(all_options, ignore_index=True)

                # Get spot price from data
                spot_price = df['spot_price'].iloc[0]

                # Apply filters
                min_strike = spot_price * (strike_range[0] / 100.0)
                max_strike = spot_price * (strike_range[1] / 100.0)
                min_dte, max_dte = maturity_range

                df = df[
                    (df['strike'] >= min_strike) &
                    (df['strike'] <= max_strike) &
                    (df['dte'] >= min_dte) &
                    (df['dte'] <= max_dte)
                ]

                # Build surface
                surface_data = build_volatility_surface(df, risk_free_rate)

                if surface_data is not None:
                    st.session_state['vol_surface_data'] = surface_data
                    st.session_state['spot_price'] = spot_price
                    st.session_state['sources'] = sources_used
                    st.session_state['filtered_options'] = df
                    st.success(f"✓ Volatility surface built ({len(df)} options)")
                else:
                    st.error("Could not build volatility surface. Insufficient data.")
            else:
                st.error("No data sources available. Enable at least one data source.")

    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================

    if 'vol_surface_data' in st.session_state:
        surface_data = st.session_state['vol_surface_data']
        spot_price = st.session_state.get('spot_price', 100.0)
        sources = st.session_state.get('sources', [])
        df_filtered = st.session_state.get('filtered_options', pd.DataFrame())

        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Spot Price", f"${spot_price:.2f}")
        with col2:
            st.metric("Options Analyzed", len(df_filtered))
        with col3:
            st.metric("Data Sources", len(sources))
        with col4:
            st.metric("Valid IVs", len(st.session_state['vol_surface_data']['all_iv_data']))

        st.markdown("---")

        # Data source summary
        st.markdown(f"**Data Sources:** {', '.join(sources)}")
        st.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")

        # Option type selection
        tab1, tab2 = st.tabs(["Call Options", "Put Options"])

        with tab1:
            if surface_data['calls'] is not None:
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig_3d = plot_volatility_surface_3d(surface_data, 'CALL')
                    if fig_3d:
                        st.plotly_chart(fig_3d, use_container_width=True)

                with col2:
                    st.markdown("### 3D Surface")
                    st.markdown("Rotate and zoom to explore the volatility landscape")
                    st.markdown("")
                    st.markdown(f"**Strikes:** {len(surface_data['calls']['strikes'])} unique")
                    st.markdown(f"**Maturities:** {len(surface_data['calls']['maturities'])} unique")

                st.markdown("---")

                # 2D slices
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### IV vs Strike")
                    maturity_days = st.slider(
                        "Select Maturity (days)",
                        min_value=int(min(surface_data['calls']['maturities']) * 365),
                        max_value=int(max(surface_data['calls']['maturities']) * 365),
                        value=int(np.median([m * 365 for m in surface_data['calls']['maturities']])),
                        step=1,
                        key="call_maturity"
                    )
                    fig_strike = plot_iv_by_strike(surface_data, maturity_days, 'CALL')
                    if fig_strike:
                        st.plotly_chart(fig_strike, use_container_width=True)

                with col2:
                    st.markdown("### IV vs Maturity")
                    strike_price = st.slider(
                        "Select Strike ($)",
                        min_value=float(min(surface_data['calls']['strikes'])),
                        max_value=float(max(surface_data['calls']['strikes'])),
                        value=float(spot_price),
                        step=1.0,
                        key="call_strike"
                    )
                    fig_mat = plot_iv_by_maturity(surface_data, strike_price, 'CALL')
                    if fig_mat:
                        st.plotly_chart(fig_mat, use_container_width=True)
            else:
                st.warning("No call option data available")

        with tab2:
            if surface_data['puts'] is not None:
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig_3d = plot_volatility_surface_3d(surface_data, 'PUT')
                    if fig_3d:
                        st.plotly_chart(fig_3d, use_container_width=True)

                with col2:
                    st.markdown("### 3D Surface")
                    st.markdown("Rotate and zoom to explore the volatility landscape")
                    st.markdown("")
                    st.markdown(f"**Strikes:** {len(surface_data['puts']['strikes'])} unique")
                    st.markdown(f"**Maturities:** {len(surface_data['puts']['maturities'])} unique")

                st.markdown("---")

                # 2D slices
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### IV vs Strike")
                    maturity_days = st.slider(
                        "Select Maturity (days)",
                        min_value=int(min(surface_data['puts']['maturities']) * 365),
                        max_value=int(max(surface_data['puts']['maturities']) * 365),
                        value=int(np.median([m * 365 for m in surface_data['puts']['maturities']])),
                        step=1,
                        key="put_maturity"
                    )
                    fig_strike = plot_iv_by_strike(surface_data, maturity_days, 'PUT')
                    if fig_strike:
                        st.plotly_chart(fig_strike, use_container_width=True)

                with col2:
                    st.markdown("### IV vs Maturity")
                    strike_price = st.slider(
                        "Select Strike ($)",
                        min_value=float(min(surface_data['puts']['strikes'])),
                        max_value=float(max(surface_data['puts']['strikes'])),
                        value=float(spot_price),
                        step=1.0,
                        key="put_strike"
                    )
                    fig_mat = plot_iv_by_maturity(surface_data, strike_price, 'PUT')
                    if fig_mat:
                        st.plotly_chart(fig_mat, use_container_width=True)
            else:
                st.warning("No put option data available")

        # Data table
        with st.expander("📊 Raw IV Data"):
            st.dataframe(
                st.session_state['vol_surface_data']['all_iv_data'][
                    ['strike', 'T', 'optionType', 'iv', 'source', 'dte']
                ].sort_values(['optionType', 'dte', 'strike']),
                use_container_width=True,
                height=400
            )
    else:
        st.info("👈 Configure inputs and click 'Fetch & Analyze' to begin")


if __name__ == "__main__":
    main()

