import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime

# ── Page Config ───────────────────────────────────────────────────────────
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
from src.chat           import get_chat_response, get_suggested_questions
from src.sentiment      import analyze_sentiment, get_sentiment_color, get_sentiment_emoji
from src.report         import generate_pdf_report


# ══════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

def validate_dataframe(df: pd.DataFrame):
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing  = [col for col in required if col not in df.columns]
    return missing


def load_dataframe(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip().title() for col in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    return df


def load_dataframe_from_path(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = [col.strip().title() for col in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["Open", "High", "Low", "Close", "Volume"], inplace=True)
    return df


def render_metric_card(label: str, value: str, delta: str = ""):
    st.markdown(f"""
    <div class="metric-card fade-in">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {"<div style='color:#26a69a;font-size:0.8rem;margin-top:0.3rem;'>" + delta + "</div>" if delta else ""}
    </div>
    """, unsafe_allow_html=True)


def render_prediction_badge(result: dict):
    is_bullish = result.get("signal", 0) == 1
    css_class  = "prediction-bullish" if is_bullish else "prediction-bearish"
    text_class = "bullish" if is_bullish else "bearish"
    prediction = result.get("prediction", "N/A")
    confidence = result.get("confidence", 0)
    st.markdown(f"""
    <div class="{css_class} fade-in">
        <div class="prediction-text {text_class}">{prediction}</div>
        <div style="margin-top:0.8rem;color:#8b949e;font-size:0.95rem;">
            Confidence: <strong style="color:white;">{confidence}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_feature_bars(top_features: list):
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
    "AAPL"      : "🍎 Apple Inc",
    "TSLA"      : "🚗 Tesla Inc",
    "MSFT"      : "💻 Microsoft",
    "GOOGL"     : "🔍 Google",
    "AMZN"      : "📦 Amazon",
    "META"      : "👤 Meta",
    "NFLX"      : "🎬 Netflix",
    "NVDA"      : "🎮 Nvidia",
    "RELIANCE"  : "🇮🇳 Reliance",
    "TCS"       : "🇮🇳 TCS",
    "INFY"      : "🇮🇳 Infosys",
    "WIPRO"     : "🇮🇳 Wipro",
    "HDFCBANK"  : "🇮🇳 HDFC Bank",
    "ICICIBANK" : "🇮🇳 ICICI Bank",
    "ADANIENT"  : "🇮🇳 Adani",
}


def get_stock_display_name(filename: str) -> str:
    key = os.path.basename(filename).replace("_stock.csv", "").replace(".csv", "").upper()
    return STOCK_NAMES.get(key, f"📊 {key}")


def get_ticker_from_path(file_path: str) -> str:
    return os.path.basename(file_path).replace("_stock.csv", "").replace(".csv", "").upper()


# ══════════════════════════════════════════════════════════════════════════
#  SAMPLE DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════

def generate_sample_data() -> pd.DataFrame:
    np.random.seed(42)
    days  = 200
    dates = pd.date_range(end=datetime.today(), periods=days, freq="B")
    price = 100.0
    data  = []
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

        st.markdown("### 📁 Stock Data")

        upload_mode = st.radio(
            "Choose Input Mode",
            ["📂 Select Sample Stock", "⬆️ Upload Your Own CSV"],
            horizontal=True
        )

        uploaded_file = None
        selected_path = None

        if upload_mode == "📂 Select Sample Stock":
            csv_files = sorted(glob.glob("data/*.csv"))
            if csv_files:
                options       = {get_stock_display_name(f): f for f in csv_files}
                display_names = list(options.keys())
                selected_display = st.selectbox("🏢 Select Stock", options=display_names)
                selected_path    = options[selected_display]
                st.markdown(f"""
                <div class="success-box">
                    ✅ Selected: <strong>{selected_display}</strong><br>
                    📄 File: <code>{os.path.basename(selected_path)}</code>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No CSV files in data/ folder! Run: python download_all_stocks.py")
        else:
            uploaded_file = st.file_uploader(
                "Upload CSV File", type=["csv"],
                help="CSV must have: Date, Open, High, Low, Close, Volume"
            )
            st.info("📋 Required Columns: Date, Open, High, Low, Close, Volume")

        st.divider()

        st.markdown("### ⚙️ Settings")
        show_groq      = st.toggle("🤖 Enable Groq AI Explanation", value=True)
        show_chat      = st.toggle("💬 Enable Stock Chat",          value=True)
        show_sentiment = st.toggle("📰 Enable News Sentiment",      value=True)
        show_report    = st.toggle("📄 Enable PDF Report",          value=True)
        show_raw       = st.toggle("📋 Show Raw Data Table",        value=False)

        st.divider()

        st.markdown("### 📥 Sample Data")
        sample_df = generate_sample_data()
        st.download_button(
            label     = "⬇️ Download Sample CSV",
            data      = sample_df.to_csv(index=False),
            file_name = "sample_stock.csv",
            mime      = "text/csv"
        )

        st.divider()

        st.markdown("""
        <div style='color:#8b949e; font-size:0.8rem; text-align:center;'>
            <p>⚠️ <strong>Disclaimer</strong></p>
            <p>Simulation tool for educational purposes only.</p>
            <p>Not financial advice.</p>
            <br>
            <p>Built with ❤️ using Streamlit + Groq</p>
        </div>
        """, unsafe_allow_html=True)

    return uploaded_file, selected_path, show_groq, show_chat, show_sentiment, show_report, show_raw


# ══════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════

def main():

    # ── Header ────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🤖 AI Trading Assistant</h1>
        <p>Upload stock data · Predict trends · Chat with AI · Simulation Only</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────
    uploaded_file, selected_path, show_groq, show_chat, show_sentiment, show_report, show_raw = render_sidebar()

    # ── Determine Data Source ─────────────────────────────────────
    df          = None
    stock_label = ""
    ticker      = ""

    if selected_path and os.path.exists(selected_path):
        try:
            df          = load_dataframe_from_path(selected_path)
            stock_label = get_stock_display_name(selected_path)
            ticker      = get_ticker_from_path(selected_path)
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")
            return

    elif uploaded_file is not None:
        try:
            df          = load_dataframe(uploaded_file)
            stock_label = uploaded_file.name.replace("_stock.csv", "").replace(".csv", "")
            ticker      = stock_label.upper()
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return

    # ── No File State ─────────────────────────────────────────────
    if df is None:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        cards = [
            ("📁", "Step 1: Select",        "Select stock from dropdown or upload your own CSV."),
            ("🤖", "Step 2: Predict",       "AI predicts next-day trend using Random Forest."),
            ("📊", "Step 3: Analyze",       "View 7 interactive charts with technical indicators."),
            ("💬", "Step 4: Chat & Report", "Chat with AI and download PDF report."),
        ]
        for col, (icon, title, desc) in zip([col1, col2, col3, col4], cards):
            with col:
                st.info(f"{icon} **{title}**\n\n{desc}")

        st.warning("⬅️ Select a stock from the dropdown or upload your CSV from the sidebar!")
        return
    # ── Validate ──────────────────────────────────────────────────
    missing = validate_dataframe(df)
    if missing:
        st.error(f"❌ Missing columns: {', '.join(missing)}")
        return

    # ── Data Loaded ───────────────────────────────────────────────
    st.success(f"✅ {stock_label} — {len(df)} rows loaded · {df['Date'].min().strftime('%b %d, %Y')} → {df['Date'].max().strftime('%b %d, %Y')} · Latest Close: ${df['Close'].iloc[-1]:.2f}")

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
    #  PREDICTION RESULT
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
            render_metric_card("Confidence",     f"{result['confidence']}%")
        with m2:
            render_metric_card("Model Accuracy", f"{result['accuracy']}%")
        with m3:
            render_metric_card("Close Price",    f"${result['latest_close']}")

        st.markdown("<br>", unsafe_allow_html=True)

        m4, m5, m6 = st.columns(3)
        with m4:
            rsi_status = (
                "Overbought" if result['latest_rsi'] > 70
                else "Oversold" if result['latest_rsi'] < 30
                else "Neutral"
            )
            render_metric_card("RSI (14)",    f"{result['latest_rsi']}", rsi_status)
        with m5:
            render_metric_card("MACD",        f"{result['latest_macd']}")
        with m6:
            ma_signal = "Bullish" if result['latest_ma7'] > result['latest_ma21'] else "Bearish"
            render_metric_card("MA7 vs MA21", ma_signal)

    st.divider()

    # ══════════════════════════════════════════════════════════════
    #  CHARTS
    # ══════════════════════════════════════════════════════════════

    st.markdown("## 📊 Technical Charts")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🕯️ Candlestick", "📈 Moving Avg", "📦 Volume",
        "⚡ RSI", "📉 MACD", "🎯 Bollinger", "🔮 Prediction"
    ])

    with tab1:
        st.plotly_chart(plot_candlestick(df),               use_container_width=True)
    with tab2:
        st.plotly_chart(plot_moving_averages(enriched_df),  use_container_width=True)
    with tab3:
        st.plotly_chart(plot_volume(enriched_df),           use_container_width=True)
    with tab4:
        st.plotly_chart(plot_rsi(enriched_df),              use_container_width=True)
    with tab5:
        st.plotly_chart(plot_macd(enriched_df),             use_container_width=True)
    with tab6:
        st.plotly_chart(plot_bollinger_bands(enriched_df),  use_container_width=True)
    with tab7:
        st.plotly_chart(plot_prediction(enriched_df, result), use_container_width=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════
    #  GROQ AI EXPLANATION
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
            st.markdown(explanation)

        if quick_sum:
            with st.spinner("⚡ Getting quick summary..."):
                summary = get_quick_summary(result)
            st.success(f"⚡ Quick Summary:\n\n{summary}")

        if not full_explain and not quick_sum:
            st.info("👆 Click Full Analysis for detailed AI explanation or Quick Summary for a 3-bullet overview.")

        st.divider()

    # ══════════════════════════════════════════════════════════════
    #  NEWS SENTIMENT
    # ══════════════════════════════════════════════════════════════

    sentiment_result = None

    if show_sentiment:
        st.markdown("## 📰 News Sentiment Analysis")

        if st.button("🔍 Analyze News Sentiment"):
            with st.spinner(f"📰 Fetching and analyzing {stock_label} news..."):
                sentiment_result = analyze_sentiment(ticker, stock_label, result)
                st.session_state["sentiment_result"] = sentiment_result

        if "sentiment_result" in st.session_state:
            sentiment_result = st.session_state["sentiment_result"]

            if sentiment_result.get("error"):
                st.error(sentiment_result["error"])
            else:
                sentiment  = sentiment_result.get("sentiment", "NEUTRAL")
                score      = sentiment_result.get("score", 50)
                headlines  = sentiment_result.get("headlines", [])
                analysis   = sentiment_result.get("analysis", "")

                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Overall Sentiment", sentiment)
                with col_s2:
                    st.metric("Sentiment Score", f"{score}/100")
                with col_s3:
                    st.metric("Headlines Found", len(headlines))

                if headlines:
                    st.markdown("#### 📰 Latest Headlines:")
                    for headline in headlines:
                        st.info(f"📌 {headline}")

                if analysis:
                    st.markdown(analysis)
        else:
            st.info("👆 Click Analyze News Sentiment to fetch latest news!")

        st.divider()

    # ══════════════════════════════════════════════════════════════
    #  STOCK CHAT
    # ══════════════════════════════════════════════════════════════

    if show_chat:
        st.markdown("## 💬 Stock Chat Assistant")
        st.info("🧠 Chat has memory — it remembers your full conversation!")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        suggestions = get_suggested_questions(result, stock_label)
        st.markdown("#### 💡 Suggested Questions:")

        cols = st.columns(4)
        for i, suggestion in enumerate(suggestions[:4]):
            with cols[i % 4]:
                if st.button(
                    suggestion[:40] + "..." if len(suggestion) > 40 else suggestion,
                    key=f"suggest_{i}",
                    use_container_width=True
                ):
                    st.session_state["chat_input_value"] = suggestion

        if st.session_state["chat_history"]:
            st.markdown("#### 💬 Conversation:")
            for chat in st.session_state["chat_history"]:
                if chat["role"] == "user":
                    st.markdown(f"👤 **You:** {chat['content']}")
                else:
                    st.markdown(f"🤖 **AI:** {chat['content']}")

        user_input = st.text_input(
            "💬 Ask anything about this stock...",
            value=st.session_state.get("chat_input_value", ""),
            placeholder=f"e.g. Why is {stock_label} showing this signal?",
            key="chat_input"
        )

        col_send, col_clear = st.columns([1, 1])
        with col_send:
            send  = st.button("📤 Send Message", use_container_width=True)
        with col_clear:
            clear = st.button("🗑️ Clear Chat",   use_container_width=True)

        if clear:
            st.session_state["chat_history"]     = []
            st.session_state["chat_input_value"] = ""
            st.rerun()

        if send and user_input.strip():
            with st.spinner("🤖 AI is thinking..."):
                response = get_chat_response(
                    user_message = user_input,
                    chat_history = st.session_state["chat_history"],
                    result       = result,
                    stock_label  = stock_label
                )
            st.session_state["chat_history"].append({"role": "user",      "content": user_input})
            st.session_state["chat_history"].append({"role": "assistant", "content": response})
            st.session_state["chat_input_value"] = ""
            st.rerun()

        st.divider()

    # ══════════════════════════════════════════════════════════════
    #  PDF REPORT
    # ══════════════════════════════════════════════════════════════

    if show_report:
        st.markdown("## 📄 Download AI Report")
        st.info("📄 Generate a professional PDF report with full AI analysis!")

        if st.button("📄 Generate PDF Report"):
            with st.spinner("📄 AI is writing your report..."):
                sentiment_data = st.session_state.get("sentiment_result", None)
                pdf_bytes      = generate_pdf_report(result, stock_label, sentiment_data)

            if pdf_bytes:
                st.success("✅ Report generated successfully!")
                st.download_button(
                    label     = "⬇️ Download PDF Report",
                    data      = pdf_bytes,
                    file_name = f"{ticker}_AI_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime      = "application/pdf",
                )
            else:
                st.error("❌ Report generation failed! Check your Groq API key.")

        st.divider()

    # ── Footer ────────────────────────────────────────────────────
    st.markdown("""
    ---
    ⚠️ This is a **simulation tool** for educational purposes only.
    Not financial advice. Always do your own research before investing.

    Built with ❤️ using **Streamlit · scikit-learn · Plotly · Groq · LLaMA 3.3**
    """)


# ── Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()