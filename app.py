# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

# ── Page Config (MUST be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title = "AI Trading Assistant",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Load Custom CSS ───────────────────────────────────────────────────────
def load_css():
    css_path = os.path.join("assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Import src modules ────────────────────────────────────────────────────
from src.predictor      import predict_trend
from src.charts         import (
    plot_candlestick,
    plot_moving_averages,
    plot_volume,
    plot_rsi,
    plot_macd,
    plot_prediction,
    plot_bollinger_bands
)
from src.groq_explainer import get_groq_explanation, get_quick_summary


# ══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def validate_dataframe(df: pd.DataFrame):
    """Check if uploaded CSV has required columns."""
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing  = [col for col in required if col not in df.columns]
    return missing


def load_dataframe(uploaded_file) -> pd.DataFrame:
    """Load and preprocess uploaded CSV."""
    df = pd.read_csv(uploaded_file)

    # Normalize column names
    df.columns = [col.strip().title() for col in df.columns]

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Convert numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

    return df


def load_dataframe_from_path(file_path: str) -> pd.DataFrame:
    """Load and preprocess CSV from file path."""
    df = pd.read_csv(file_path)

    # Normalize column names
    df.columns = [col.strip().title() for col in df.columns]

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Convert numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)

    return df


def render_metric_card(label: str, value: str, delta: str = ""):
    """Render a styled metric card."""
    st.markdown(f"""
    <div class="metric-card fade-in">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {"<div style='color:#26a69a;font-size:0.8rem;margin-top:0.3rem;'>" + delta + "</div>" if delta else ""}
    </div>
    """, unsafe_allow_html=True)


def render_prediction_badge(result: dict):
    """Render the big prediction badge."""
    is_bullish   = result.get("signal", 0) == 1
    css_class    = "prediction-bullish" if is_bullish else "prediction-bearish"
    text_class   = "bullish" if is_bullish else "bearish"
    prediction   = result.get("prediction", "N/A")
    confidence   = result.get("confidence", 0)

    st.markdown(f"""
    <div class="{css_class} fade-in">
        <div class="prediction-text {text_class}">{prediction}</div>
        <div style="margin-top:0.8rem;color:#8b949e;font-size:0.95rem;">
            Confidence: <strong style="color:white;">{confidence}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_feature_bars(top_features: list):
    """Render feature importance bars."""
    st.markdown("#### 🏆 Top Influencing Indicators")
    for feat, imp in top_features:
        pct = round(imp * 100, 2)
        st.markdown(f"""
        <div class="feature-bar-container">
            <div class="feature-bar-label">
                <span>{feat.replace('_', ' ')}</span>
                <span>{pct}%</span>
            </div>
            <div class="feature-bar-track">
                <div class="feature-bar-fill" style="width:{min(pct*5, 100)}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  STOCK NAME MAPPER
# ══════════════════════════════════════════════════════════════════════════

STOCK_NAMES = {
    "AAPL"       : "🍎 Apple Inc",
    "TSLA"       : "🚗 Tesla Inc",
    "MSFT"       : "💻 Microsoft",
    "GOOGL"      : "🔍 Google",
    "AMZN"       : "📦 Amazon",
    "META"       : "👤 Meta",
    "NFLX"       : "🎬 Netflix",
    "NVDA"       : "🎮 Nvidia",
    "RELIANCE"   : "🇮🇳 Reliance",
    "TCS"        : "🇮🇳 TCS",
    "INFY"       : "🇮🇳 Infosys",
    "WIPRO"      : "🇮🇳 Wipro",
    "HDFCBANK"   : "🇮🇳 HDFC Bank",
    "ICICIBANK"  : "🇮🇳 ICICI Bank",
    "ADANIENT"   : "🇮🇳 Adani",
}


def get_stock_display_name(filename: str) -> str:
    """Convert filename to display name."""
    key = os.path.basename(filename).replace("_stock.csv", "").replace(".csv", "").upper()
    return STOCK_NAMES.get(key, f"📊 {key}")


# ══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0;'>
            <div style='font-size:2.5rem;'>📈</div>
            <h2 style='color:#e6edf3; margin:0.3rem 0;'>AI Trading</h2>
            <p style='color:#8b949e; font-size:0.85rem;'>Simulation Assistant</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Stock Data Section ────────────────────────────────────
        st.markdown("### 📁 Stock Data")

        # ── Input Mode Toggle ─────────────────────────────────────
        upload_mode = st.radio(
            "Choose Input Mode",
            ["📂 Select Sample Stock", "⬆️ Upload Your Own CSV"],
            horizontal=True
        )

        uploaded_file  = None
        selected_path  = None

        # ── Mode 1: Select from data/ folder ─────────────────────
        if upload_mode == "📂 Select Sample Stock":

            csv_files = sorted(glob.glob("data/*.csv"))

            if csv_files:
                # Build display options
                options      = {get_stock_display_name(f): f for f in csv_files}
                display_names = list(options.keys())

                selected_display = st.selectbox(
                    "🏢 Select Stock",
                    options=display_names
                )

                selected_path = options[selected_display]

                # Show file info
                ticker_name = os.path.basename(selected_path).replace("_stock.csv", "")
                st.markdown(f"""
                <div class="success-box">
                    ✅ Selected: <strong>{selected_display}</strong><br>
                    📄 File: <code>{os.path.basename(selected_path)}</code>
                </div>
                """, unsafe_allow_html=True)

            else:
                st.markdown("""
                <div class="warning-box">
                    ⚠️ No CSV files found in <code>data/</code> folder!<br><br>
                    Run this command first:<br>
                    <code>python download_all_stocks.py</code>
                </div>
                """, unsafe_allow_html=True)

        # ── Mode 2: Upload your own CSV ───────────────────────────
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV File",
                type=["csv"],
                help="CSV must have: Date, Open, High, Low, Close, Volume"
            )

            st.markdown("""
            <div class="info-box">
            📋 <strong>Required Columns:</strong><br>
            Date, Open, High, Low, Close, Volume
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Settings ──────────────────────────────────────────────
        st.markdown("### ⚙️ Settings")
        show_groq = st.toggle("🤖 Enable Groq AI Explanation", value=True)
        show_raw  = st.toggle("📋 Show Raw Data Table", value=False)

        st.divider()

        # ── Sample Data Download ──────────────────────────────────
        st.markdown("### 📥 Sample Data")
        sample_df = generate_sample_data()
        csv_data  = sample_df.to_csv(index=False)
        st.download_button(
            label     = "⬇️ Download Sample CSV",
            data      = csv_data,
            file_name = "sample_stock.csv",
            mime      = "text/csv"
        )

        st.divider()

        # ── About ─────────────────────────────────────────────────
        st.markdown("""
        <div style='color:#8b949e; font-size:0.8rem; text-align:center;'>
            <p>⚠️ <strong>Disclaimer</strong></p>
            <p>This is a simulation tool for educational purposes only.</p>
            <p>Not financial advice.</p>
            <br>
            <p>Built with ❤️ using Streamlit + Groq</p>
        </div>
        """, unsafe_allow_html=True)

    return uploaded_file, selected_path, show_groq, show_raw


# ══════════════════════════════════════════════════════════════════════════
#  SAMPLE DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════

def generate_sample_data() -> pd.DataFrame:
    """Generate realistic-looking sample stock data."""
    np.random.seed(42)
    days   = 200
    dates  = pd.date_range(end=datetime.today(), periods=days, freq="B")
    price  = 100.0
    data   = []

    for date in dates:
        change = np.random.normal(0.001, 0.02)
        open_p = round(price, 2)
        close  = round(price * (1 + change), 2)
        high   = round(max(open_p, close) * (1 + abs(np.random.normal(0, 0.005))), 2)
        low    = round(min(open_p, close) * (1 - abs(np.random.normal(0, 0.005))), 2)
        volume = int(np.random.randint(500000, 5000000))
        data.append([date.strftime("%Y-%m-%d"), open_p, high, low, close, volume])
        price  = close

    return pd.DataFrame(data, columns=["Date", "Open", "High", "Low", "Close", "Volume"])


# ══════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════

def main():

    # ── Header ────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🤖 AI Trading Assistant</h1>
        <p>Upload stock data · Predict trends · Understand the market · Simulation Only</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────
    uploaded_file, selected_path, show_groq, show_raw = render_sidebar()

    # ── Determine Data Source ─────────────────────────────────────
    df           = None
    stock_label  = ""

    if selected_path and os.path.exists(selected_path):
        # Load from data/ folder
        try:
            df          = load_dataframe_from_path(selected_path)
            stock_label = get_stock_display_name(selected_path)
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            return

    elif uploaded_file is not None:
        # Load from uploaded file
        try:
            df          = load_dataframe(uploaded_file)
            stock_label = uploaded_file.name.replace("_stock.csv", "").replace(".csv", "")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return

    # ── No File State ─────────────────────────────────────────────
    if df is None:
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="section-card fade-in">
                <div style='font-size:2rem; text-align:center;'>📁</div>
                <div class="section-title" style='text-align:center;'>Step 1: Select</div>
                <p style='color:#8b949e; font-size:0.9rem; text-align:center;'>
                    Select a stock from dropdown or upload your own CSV from the sidebar.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="section-card fade-in">
                <div style='font-size:2rem; text-align:center;'>🤖</div>
                <div class="section-title" style='text-align:center;'>Step 2: Predict</div>
                <p style='color:#8b949e; font-size:0.9rem; text-align:center;'>
                    AI runs Random Forest on 17 technical indicators to predict next-day trend.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="section-card fade-in">
                <div style='font-size:2rem; text-align:center;'>📊</div>
                <div class="section-title" style='text-align:center;'>Step 3: Analyze</div>
                <p style='color:#8b949e; font-size:0.9rem; text-align:center;'>
                    View interactive charts, RSI, MACD and get Groq AI explanation.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="warning-box" style='text-align:center; margin-top:2rem;'>
            ⬅️ Select a stock from the dropdown or upload your own CSV from the sidebar!
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Validate Data ─────────────────────────────────────────────
    missing = validate_dataframe(df)
    if missing:
        st.error(f"❌ Missing columns: {', '.join(missing)}")
        st.info("Required: Date, Open, High, Low, Close, Volume")
        return

    # ── Data Loaded Successfully ──────────────────────────────────
    st.markdown(f"""
    <div class="success-box fade-in">
        ✅ <strong>{stock_label}</strong> — {len(df)} rows loaded ·
        {df['Date'].min().strftime('%b %d, %Y')} → {df['Date'].max().strftime('%b %d, %Y')} ·
        Latest Close: <strong>${df['Close'].iloc[-1]:.2f}</strong>
    </div>
    """, unsafe_allow_html=True)

    # ── Raw Data Table ────────────────────────────────────────────
    if show_raw:
        st.markdown("### 📋 Raw Data")
        st.dataframe(df.tail(20), use_container_width=True)

    st.divider()

    # ── Run Prediction ────────────────────────────────────────────
    st.markdown(f"### 🤖 Running AI Prediction for {stock_label}...")

    with st.spinner("Analyzing stock data with Random Forest..."):
        result = predict_trend(df)

    if result.get("error"):
        st.error(f"❌ {result['error']}")
        return

    enriched_df = result.get("enriched_df", df)

    # ══════════════════════════════════════════════════════════════
    #  PREDICTION RESULT SECTION
    # ══════════════════════════════════════════════════════════════

    st.markdown(f"## 🎯 Prediction Result — {stock_label}")

    col_pred, col_metrics = st.columns([1, 2])

    with col_pred:
        render_prediction_badge(result)
        st.markdown("<br>", unsafe_allow_html=True)
        render_feature_bars(result.get("top_features", []))

    with col_metrics:
        st.markdown("#### 📊 Key Metrics")

        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric_card("Confidence",  f"{result['confidence']}%")
        with m2:
            render_metric_card("Model Accuracy", f"{result['accuracy']}%")
        with m3:
            render_metric_card("Close Price", f"${result['latest_close']}")

        st.markdown("<br>", unsafe_allow_html=True)

        m4, m5, m6 = st.columns(3)
        with m4:
            rsi_status = (
                "Overbought" if result['latest_rsi'] > 70
                else "Oversold" if result['latest_rsi'] < 30
                else "Neutral"
            )
            render_metric_card("RSI (14)", f"{result['latest_rsi']}", rsi_status)

        with m5:
            render_metric_card("MACD", f"{result['latest_macd']}")

        with m6:
            ma_signal = "Bullish" if result['latest_ma7'] > result['latest_ma21'] else "Bearish"
            render_metric_card("MA7 vs MA21", ma_signal)

    st.divider()

    # ══════════════════════════════════════════════════════════════
    #  CHARTS SECTION
    # ══════════════════════════════════════════════════════════════

    st.markdown("## 📊 Technical Charts")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🕯️ Candlestick",
        "📈 Moving Avg",
        "📦 Volume",
        "⚡ RSI",
        "📉 MACD",
        "🎯 Bollinger",
        "🔮 Prediction"
    ])

    with tab1:
        st.plotly_chart(plot_candlestick(df), use_container_width=True)

    with tab2:
        st.plotly_chart(plot_moving_averages(enriched_df), use_container_width=True)

    with tab3:
        st.plotly_chart(plot_volume(enriched_df), use_container_width=True)

    with tab4:
        st.plotly_chart(plot_rsi(enriched_df), use_container_width=True)

    with tab5:
        st.plotly_chart(plot_macd(enriched_df), use_container_width=True)

    with tab6:
        st.plotly_chart(plot_bollinger_bands(enriched_df), use_container_width=True)

    with tab7:
        st.plotly_chart(plot_prediction(enriched_df, result), use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════
    #  GROQ AI EXPLANATION SECTION
    # ══════════════════════════════════════════════════════════════

    if show_groq:
        st.markdown("## 🤖 Groq AI Explanation")

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])

        with col_btn1:
            full_explain = st.button("📝 Full Analysis", use_container_width=True)
        with col_btn2:
            quick_sum    = st.button("⚡ Quick Summary", use_container_width=True)

        if full_explain:
            with st.spinner("🤖 Groq AI is analyzing the market..."):
                explanation = get_groq_explanation(result)

            st.markdown("""
            <div class="section-card fade-in">
            """, unsafe_allow_html=True)
            st.markdown(explanation)
            st.markdown("</div>", unsafe_allow_html=True)

        if quick_sum:
            with st.spinner("⚡ Getting quick summary..."):
                summary = get_quick_summary(result)

            st.markdown(f"""
            <div class="success-box fade-in">
                <strong>⚡ Quick Summary:</strong><br><br>
                {summary}
            </div>
            """, unsafe_allow_html=True)

        if not full_explain and not quick_sum:
            st.markdown("""
            <div class="info-box">
                👆 Click <strong>Full Analysis</strong> for detailed AI explanation
                or <strong>Quick Summary</strong> for a 3-bullet overview.
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Footer ────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; color:#8b949e; font-size:0.85rem; padding:1rem 0 2rem 0;'>
        ⚠️ This is a <strong>simulation tool</strong> for educational purposes only.<br>
        Not financial advice. Always do your own research before investing.<br><br>
        Built with ❤️ using <strong>Streamlit · scikit-learn · Plotly · Groq</strong>
    </div>
    """, unsafe_allow_html=True)


# ── Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()