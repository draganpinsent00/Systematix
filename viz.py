# Plotly helpers for Streamlit visualizations (professional theme)
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# professional muted palette
PALETTE = {
    'primary': '#1F3251',  # deep navy
    'muted': '#6B7280',     # muted gray
    'accent': '#2C7A7B',    # teal/steel
    'highlight': '#2C6AA0',
    'bg': '#FFFFFF',
    'panel_bg': '#F7F9FB',
}


def _apply_plotly_style(fig, title=None, xaxis_title=None, yaxis_title=None):
    try:
        fig.update_layout(
            template='plotly_white',
            plot_bgcolor=PALETTE['bg'],
            paper_bgcolor=PALETTE['bg'],
            title=dict(text=title or '', x=0.01, xanchor='left', font=dict(size=14)),
            font=dict(family='Arial, sans-serif', color=PALETTE['primary'], size=12),
            margin=dict(l=40, r=24, t=48, b=40)
        )
        fig.update_xaxes(title_text=xaxis_title or '', showgrid=True, gridcolor='#ECEEEF', zeroline=False, tickfont=dict(color=PALETTE['primary']))
        fig.update_yaxes(title_text=yaxis_title or '', showgrid=True, gridcolor='#ECEEEF', zeroline=False, tickfont=dict(color=PALETTE['primary']))
    except Exception:
        pass
    return fig


def plot_terminal_histogram(S_T: np.ndarray, title: str = 'Terminal price distribution', nbins: int = 80):
    """Return a Plotly histogram of terminal prices (density).

    S_T: 1D array of terminal prices
    """
    df = pd.DataFrame({'S_T': np.asarray(S_T).ravel()})
    fig = px.histogram(df, x='S_T', nbins=nbins, histnorm='probability density')
    # muted bar color
    fig.update_traces(marker_color=PALETTE['primary'], marker_line_color='rgba(0,0,0,0)', opacity=0.9)
    fig = _apply_plotly_style(fig, title=title, xaxis_title='Terminal price', yaxis_title='Density')
    return fig


def plot_sample_paths(paths: np.ndarray, n_display: int = 10, T: float = 1.0, title: str = 'Sample Price Paths'):
    """Return a Plotly line chart of sample paths.

    paths: array shape (n_paths, steps+1)
    n_display: number of sample paths to draw (will sample first n_display)
    T: total time (years) to set x-axis
    """
    n_plot = min(int(n_display), int(paths.shape[0]))
    steps_plus = paths.shape[1]
    times = np.linspace(0, T, steps_plus)
    fig = go.Figure()
    # muted lines for individual paths
    for i in range(n_plot):
        fig.add_trace(go.Scatter(x=times, y=paths[i, :], mode='lines', line=dict(color=PALETTE['muted'], width=1), name=f'Path {i+1}'))
    # percentile ribbon (10-90)
    perc10 = np.percentile(paths, 10, axis=0)
    perc90 = np.percentile(paths, 90, axis=0)
    fig.add_trace(go.Scatter(x=times, y=perc90, fill=None, mode='lines', line=dict(color=PALETTE['accent'], width=1.5), showlegend=False))
    fig.add_trace(go.Scatter(x=times, y=perc10, fill='tonexty', mode='lines', line=dict(color=PALETTE['accent'], width=1.5), showlegend=False))
    fig = _apply_plotly_style(fig, title=title, xaxis_title='Time', yaxis_title='Price')
    return fig


def plot_overlay_histograms(arrays: dict, title: str = 'Overlayed payoff distributions'):
    """Plot overlayed histograms/KDE for multiple arrays.

    arrays: dict of label -> 1D numpy array
    """
    fig = go.Figure()
    colors = [PALETTE['primary'], PALETTE['accent'], PALETTE['highlight'], PALETTE['muted']]
    for i, (label, arr) in enumerate(arrays.items()):
        fig.add_trace(go.Histogram(x=np.asarray(arr).ravel(), name=label, histnorm='probability density', opacity=0.6, nbinsx=80, marker_color=colors[i % len(colors)]))
    fig.update_layout(barmode='overlay')
    fig = _apply_plotly_style(fig, title=title, xaxis_title='Value', yaxis_title='Density')
    return fig
