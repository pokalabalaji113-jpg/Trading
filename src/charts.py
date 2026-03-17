# src/charts.py

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ── 1. Candlestick Chart ─────────────────────────────────────────────────────
def plot_candlestick(df: pd.DataFrame) -> go.Figure:
    """OHLC Candlestick chart with volume bars."""

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25]
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ),
        row=1, col=1
    )

    # Volume bars
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["Close"], df["Open"])]

    fig.add_trace(
        go.Bar(
            x=df["Date"],
            y=df["Volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )

    fig.update_layout(
        title="📊 Candlestick Chart with Volume",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=550,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white")
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


# ── 2. Moving Averages Chart ─────────────────────────────────────────────────
def plot_moving_averages(df: pd.DataFrame) -> go.Figure:
    """Close price with MA7, MA14, MA21 overlaid."""

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        name="Close Price",
        line=dict(color="#ffffff", width=1.5),
        opacity=0.8
    ))

    ma_colors = {"MA_7": "#f39c12", "MA_14": "#3498db", "MA_21": "#e74c3c"}

    for ma, color in ma_colors.items():
        if ma in df.columns:
            fig.add_trace(go.Scatter(
                x=df["Date"], y=df[ma],
                name=ma.replace("_", " "),
                line=dict(color=color, width=1.8, dash="dot")
            ))

    fig.update_layout(
        title="📈 Moving Averages (MA7 / MA14 / MA21)",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=450,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig


# ── 3. Volume Chart ──────────────────────────────────────────────────────────
def plot_volume(df: pd.DataFrame) -> go.Figure:
    """Volume bar chart with 7-day average line."""

    fig = go.Figure()

    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(df["Close"], df["Open"])]

    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Volume"],
        name="Volume",
        marker_color=colors,
        opacity=0.85
    ))

    if "Volume_MA" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Volume_MA"],
            name="7-Day Avg Volume",
            line=dict(color="#f39c12", width=2)
        ))

    fig.update_layout(
        title="📦 Volume Analysis",
        xaxis_title="Date",
        yaxis_title="Volume",
        template="plotly_dark",
        height=400,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white")
    )

    return fig


# ── 4. RSI Chart ─────────────────────────────────────────────────────────────
def plot_rsi(df: pd.DataFrame) -> go.Figure:
    """RSI chart with overbought/oversold zones."""

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["RSI"],
        name="RSI",
        line=dict(color="#9b59b6", width=2)
    ))

    # Overbought zone (70)
    fig.add_hline(
        y=70, line_dash="dash",
        line_color="#ef5350",
        annotation_text="Overbought (70)",
        annotation_position="top right"
    )

    # Oversold zone (30)
    fig.add_hline(
        y=30, line_dash="dash",
        line_color="#26a69a",
        annotation_text="Oversold (30)",
        annotation_position="bottom right"
    )

    # Neutral line (50)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5)

    # Shade overbought region
    fig.add_hrect(
        y0=70, y1=100,
        fillcolor="#ef5350", opacity=0.07,
        layer="below", line_width=0
    )

    # Shade oversold region
    fig.add_hrect(
        y0=0, y1=30,
        fillcolor="#26a69a", opacity=0.07,
        layer="below", line_width=0
    )

    fig.update_layout(
        title="⚡ RSI — Relative Strength Index",
        xaxis_title="Date",
        yaxis_title="RSI Value",
        yaxis=dict(range=[0, 100]),
        template="plotly_dark",
        height=400,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white")
    )

    return fig


# ── 5. MACD Chart ────────────────────────────────────────────────────────────
def plot_macd(df: pd.DataFrame) -> go.Figure:
    """MACD line, Signal line, and Histogram."""

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, row_heights=[0.6, 0.4])

    # Close Price on top
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        name="Close",
        line=dict(color="#ffffff", width=1.5)
    ), row=1, col=1)

    # MACD Line
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MACD"],
        name="MACD",
        line=dict(color="#3498db", width=2)
    ), row=2, col=1)

    # Signal Line
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Signal_Line"],
        name="Signal Line",
        line=dict(color="#f39c12", width=2, dash="dot")
    ), row=2, col=1)

    # Histogram
    if "MACD" in df.columns and "Signal_Line" in df.columns:
        histogram = df["MACD"] - df["Signal_Line"]
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in histogram]
        fig.add_trace(go.Bar(
            x=df["Date"], y=histogram,
            name="Histogram",
            marker_color=colors,
            opacity=0.6
        ), row=2, col=1)

    fig.update_layout(
        title="📉 MACD — Moving Average Convergence Divergence",
        template="plotly_dark",
        height=550,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig


# ── 6. Prediction Result Chart ───────────────────────────────────────────────
def plot_prediction(df: pd.DataFrame, result: dict) -> go.Figure:
    """Last 30 days close price with prediction arrow."""

    last_30 = df.tail(30).copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=last_30["Date"], y=last_30["Close"],
        name="Close Price",
        line=dict(color="#3498db", width=2),
        fill="tozeroy",
        fillcolor="rgba(52, 152, 219, 0.1)"
    ))

    # Mark last point
    last_date  = last_30["Date"].iloc[-1]
    last_close = last_30["Close"].iloc[-1]

    is_bullish = result.get("signal", 0) == 1
    arrow_color = "#26a69a" if is_bullish else "#ef5350"
    arrow_text  = f"{'▲ BULLISH' if is_bullish else '▼ BEARISH'}<br>Conf: {result.get('confidence', 0)}%"

    fig.add_annotation(
        x=last_date,
        y=last_close,
        text=arrow_text,
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor=arrow_color,
        font=dict(color=arrow_color, size=13, family="Arial Black"),
        bgcolor="#0e1117",
        bordercolor=arrow_color,
        borderwidth=1.5,
        ax=0,
        ay=-60
    )

    fig.update_layout(
        title="🎯 Prediction — Last 30 Days + Next Day Signal",
        xaxis_title="Date",
        yaxis_title="Close Price",
        template="plotly_dark",
        height=420,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white")
    )

    return fig


# ── 7. Bollinger Bands Chart ─────────────────────────────────────────────────
def plot_bollinger_bands(df: pd.DataFrame) -> go.Figure:
    """Close price with Bollinger Bands."""

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["BB_Upper"],
        name="Upper Band",
        line=dict(color="#e74c3c", width=1.5, dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["BB_Lower"],
        name="Lower Band",
        line=dict(color="#26a69a", width=1.5, dash="dash"),
        fill="tonexty",
        fillcolor="rgba(52, 152, 219, 0.08)"
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"],
        name="Close Price",
        line=dict(color="#ffffff", width=2)
    ))

    fig.update_layout(
        title="🎯 Bollinger Bands",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=450,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig