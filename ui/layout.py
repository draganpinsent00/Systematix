"""
Streamlit layout and structure.
"""

import streamlit as st


def page_header():
    """Display application header."""
    st.set_page_config(
        page_title="Systematix - Monte Carlo Options Pricing",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    .main {max-width: 1400px; margin: 0 auto;}
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {margin-bottom: 0;}
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("📊 Systematix")
        st.markdown("**Production-Grade Monte Carlo Options Pricing Platform**")


def sidebar_header():
    """Display sidebar header with version info."""
    st.sidebar.markdown("## ⚙️ Configuration")


def tabs_structure():
    """Create main tab structure."""
    return st.tabs([
        "📝 Inputs",
        "💰 Results",
        "📈 Diagnostics",
        "📊 Greeks",
        "⚠️ Risk",
        "🎯 Scenarios",
    ])

