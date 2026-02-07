"""
Diagnostic and convergence charts.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict
from .plotly_theme import apply_theme, THEME_COLORS


def chart_convergence(
    convergence_history: np.ndarray,
    title: str = "Convergence Analysis",
) -> go.Figure:
    """
    Plot cumulative mean convergence.

    Args:
        convergence_history: Array of cumulative means
        title: Chart title

    Returns:
        Plotly figure
    """
    n = len(convergence_history)
    iterations = np.arange(1, n + 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=iterations,
        y=convergence_history,
        mode='lines',
        line=dict(color=THEME_COLORS["primary"], width=1),
        name='Cumulative Mean',
    ))

    # Add confidence band (simple: std error)
    window = max(100, n // 10)
    rolling_std = np.array([
        np.std(convergence_history[max(0, i-window):i])
        for i in range(1, n + 1)
    ])

    fig.add_trace(go.Scatter(
        x=iterations,
        y=convergence_history + rolling_std,
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=iterations,
        y=convergence_history - rolling_std,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='±1 Std Error',
        fillcolor='rgba(31, 119, 180, 0.2)',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Number of Paths",
        yaxis_title="Option Price (Cumulative Mean)",
        height=400,
    )

    fig = apply_theme(fig)
    fig.update_layout(title=dict(font=dict(color='white')), legend=dict(font=dict(color='white')))
    return fig


def chart_var_cvar(
    payoffs: np.ndarray,
    var: float,
    cvar: float,
    confidence: float = 0.95,
    title: str = "VaR / CVaR Analysis",
) -> go.Figure:
    """
    Plot payoff distribution with VaR/CVaR highlighted.

    Args:
        payoffs: Array of payoffs
        var: Value at Risk
        cvar: Conditional VaR
        confidence: Confidence level
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[
        go.Histogram(
            x=payoffs,
            nbinsx=50,
            marker=dict(color=THEME_COLORS["primary"]),
            name='Payoff Distribution',
        )
    ])

    # Mark VaR and CVaR
    var_threshold = np.percentile(payoffs, (1 - confidence) * 100)

    fig.add_vline(x=var_threshold, line_dash="dash", line_color=THEME_COLORS["danger"],
                  annotation_text=f"VaR: {var:.2f}", annotation_position="top left")

    fig.update_layout(
        title=title,
        xaxis_title="Payoff / P&L",
        yaxis_title="Frequency",
        height=400,
    )

    return apply_theme(fig)


def chart_diagnostic_table(diagnostics: Dict[str, float], title: str = "Path Diagnostics") -> go.Figure:
    """
    Create a table of diagnostic metrics.

    Args:
        diagnostics: Dict of metric names and values
        title: Table title

    Returns:
        Plotly table figure
    """
    metrics = list(diagnostics.keys())
    values = [f'{v:.6f}' if isinstance(v, float) else str(v) for v in diagnostics.values()]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Metric', 'Value'],
            fill_color=THEME_COLORS["primary"],
            align='left',
        ),
        cells=dict(
            values=[metrics, values],
            fill_color='lavender',
            align='left',
        )
    )])

    fig.update_layout(title=title, height=300)
    fig = apply_theme(fig)
    fig.update_layout(title=dict(font=dict(color='white')), legend=dict(font=dict(color='white')))
    return fig
