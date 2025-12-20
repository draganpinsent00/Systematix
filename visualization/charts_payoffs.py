"""
Payoff and Greeks visualization.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional, Dict
from .plotly_theme import apply_theme, THEME_COLORS


def chart_payoff_diagram(
    spot_range: np.ndarray,
    payoffs: np.ndarray,
    title: str = "Payoff Diagram",
    current_spot: Optional[float] = None,
) -> go.Figure:
    """
    Plot option payoff at expiration.

    Args:
        spot_range: Array of spot prices
        payoffs: Corresponding payoffs
        title: Chart title
        current_spot: Current spot price (vertical line)

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=spot_range,
        y=payoffs,
        mode='lines+markers',
        line=dict(color=THEME_COLORS["primary"], width=2),
        fill='tozeroy',
        name='Payoff',
        hovertemplate='Spot: %{x:.2f}<br>Payoff: %{y:.2f}<extra></extra>',
    ))

    if current_spot is not None:
        fig.add_vline(x=current_spot, line_dash="dash", line_color=THEME_COLORS["danger"],
                      annotation_text="Current Spot", annotation_position="top right")

    fig.update_layout(
        title=title,
        xaxis_title="Spot Price at Expiration",
        yaxis_title="Option Payoff",
        height=400,
    )

    return apply_theme(fig)


def chart_pnl_distribution(
    payoffs: np.ndarray,
    title: str = "P&L Distribution",
    premium: float = 0.0,
) -> go.Figure:
    """
    Plot P&L distribution from MC simulation.

    Args:
        payoffs: Option payoff values
        title: Chart title
        premium: Option premium (cost)

    Returns:
        Plotly figure
    """
    pnl = payoffs - premium

    fig = go.Figure(data=[
        go.Histogram(
            x=pnl,
            nbinsx=50,
            marker=dict(color=THEME_COLORS["primary"]),
            name='P&L',
        )
    ])

    fig.add_vline(x=0, line_dash="dash", line_color=THEME_COLORS["danger"],
                  annotation_text="Breakeven", annotation_position="top right")

    fig.update_layout(
        title=title,
        xaxis_title="P&L",
        yaxis_title="Frequency",
        height=400,
    )

    return apply_theme(fig)


def chart_greeks_sensitivity(
    greeks: Dict[str, float],
    title: str = "Greeks Sensitivity",
) -> go.Figure:
    """
    Bar chart of computed Greeks.

    Args:
        greeks: Dictionary of Greek names and values
        title: Chart title

    Returns:
        Plotly figure
    """
    greek_names = list(greeks.keys())
    greek_values = list(greeks.values())

    fig = go.Figure(data=[
        go.Bar(
            x=greek_names,
            y=greek_values,
            marker=dict(color=[THEME_COLORS["primary"] if v >= 0 else THEME_COLORS["danger"] for v in greek_values]),
            text=[f'{v:.4f}' for v in greek_values],
            textposition='outside',
        )
    ])

    fig.update_layout(
        title=title,
        yaxis_title="Greek Value",
        height=400,
    )

    return apply_theme(fig)

