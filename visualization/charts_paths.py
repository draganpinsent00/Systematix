"""
Path visualization charts.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional
from .plotly_theme import apply_theme, THEME_COLORS


def chart_simulation_paths(
    paths: np.ndarray,
    num_to_plot: int = 100,
    title: str = "Simulated Paths",
) -> go.Figure:
    """
    Plot sample simulation paths.

    Args:
        paths: Array of shape (num_paths, num_steps + 1)
        num_to_plot: Number of paths to display (for clarity)
        title: Chart title

    Returns:
        Plotly figure
    """
    num_paths = paths.shape[0]
    num_steps = paths.shape[1]

    # Sample paths for visualization
    indices = np.linspace(0, num_paths - 1, min(num_to_plot, num_paths), dtype=int)

    fig = go.Figure()

    time_steps = np.arange(num_steps)

    for idx in indices:
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=paths[idx, :],
            mode='lines',
            line=dict(color=THEME_COLORS["primary"], width=0.5),
            opacity=0.3,
            hoverinfo='y',
            showlegend=False,
        ))

    # Add mean path
    mean_path = np.mean(paths, axis=0)
    fig.add_trace(go.Scatter(
        x=time_steps,
        y=mean_path,
        mode='lines+markers',
        line=dict(color=THEME_COLORS["danger"], width=3),
        name='Mean Path',
        hovertemplate='Step: %{x}<br>Price: %{y:.2f}<extra></extra>',
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Time Step",
        yaxis_title="Spot Price",
        height=400,
    )

    fig = apply_theme(fig)
    fig.update_layout(title=dict(font=dict(color='white')), legend=dict(font=dict(color='white')))
    return fig


def chart_distribution_histogram(
    values: np.ndarray,
    title: str = "Distribution",
    xlabel: str = "Value",
) -> go.Figure:
    """
    Plot histogram of values.

    Args:
        values: Array of values
        title: Chart title
        xlabel: X-axis label

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[
        go.Histogram(
            x=values,
            nbinsx=50,
            marker=dict(color=THEME_COLORS["primary"]),
            opacity=0.7,
            name='Distribution',
        )
    ])

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title="Frequency",
        height=400,
    )

    fig = apply_theme(fig)
    fig.update_layout(title=dict(font=dict(color='white')), legend=dict(font=dict(color='white')))
    return fig

