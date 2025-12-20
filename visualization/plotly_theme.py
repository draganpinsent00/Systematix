"""
Plotly theme and styling.
"""

import plotly.graph_objects as go
from typing import Dict, Any


THEME_COLORS = {
    "primary": "#d4af7c",
    "secondary": "#2a3f5f",
    "accent": "#d4af7c",
    "success": "#7cb342",
    "danger": "#d32f2f",
    "neutral": "#d4dce8",
    "light": "#f5f7fa",
    "dark": "#0a0e1f",
    "navy_dark": "#0a0e1f",
    "navy_primary": "#141829",
    "slate_secondary": "#2a3f5f",
    "text_primary": "#f5f7fa",
    "text_secondary": "#d4dce8",
    "accent_gold": "#d4af7c",
}


def get_plotly_template() -> Dict[str, Any]:
    """Return Plotly template for professional charts with StochastiX branding."""
    return {
        "layout": go.Layout(
            template="plotly_dark",
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                     size=11,
                     color="#ffffff"),
            title_font_size=14,
            title_x=0.5,
            hovermode="x unified",
            plot_bgcolor=THEME_COLORS["navy_primary"],
            paper_bgcolor=THEME_COLORS["navy_dark"],
            xaxis=dict(
                showgrid=True,
                gridwidth=0.5,
                gridcolor="rgba(212, 175, 124, 0.1)",
                showline=True,
                linewidth=1,
                linecolor=THEME_COLORS["accent_gold"],
                title=dict(text="", font=dict(color="#ffffff", size=12)),
                tickfont=dict(color="#ffffff"),
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=0.5,
                gridcolor="rgba(212, 175, 124, 0.1)",
                showline=True,
                linewidth=1,
                linecolor=THEME_COLORS["accent_gold"],
                title=dict(text="", font=dict(color="#ffffff", size=12)),
                tickfont=dict(color="#ffffff"),
            ),
            margin=dict(l=60, r=60, t=60, b=60),
        )
    }


def apply_theme(fig: go.Figure) -> go.Figure:
    """Apply professional StochastiX theme to figure."""
    fig.update_layout(
        template="plotly_dark",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                 size=11,
                 color="#ffffff"),
        hovermode="x unified",
        plot_bgcolor=THEME_COLORS["navy_primary"],
        paper_bgcolor=THEME_COLORS["navy_dark"],
        title=dict(font=dict(color="#ffffff")),
        xaxis=dict(
            gridcolor="rgba(212, 175, 124, 0.1)",
            showline=True,
            linecolor=THEME_COLORS["accent_gold"],
            title=dict(font=dict(color="#ffffff")),
            tickfont=dict(color="#ffffff"),
        ),
        yaxis=dict(
            gridcolor="rgba(212, 175, 124, 0.1)",
            showline=True,
            linecolor=THEME_COLORS["accent_gold"],
            title=dict(font=dict(color="#ffffff")),
            tickfont=dict(color="#ffffff"),
        ),
    )

    # Update traces to use brand accent color (safely handle all trace types)
    for trace in fig.data:
        if hasattr(trace, 'line') and trace.line is not None:
            trace.line.color = THEME_COLORS["accent_gold"]

    return fig


