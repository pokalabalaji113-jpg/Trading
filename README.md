# 🤖 AI Trading Assistant (Simulation)

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.3+-orange?style=for-the-badge&logo=scikit-learn)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.3-purple?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-green?style=for-the-badge&logo=plotly)

> ⚠️ **Disclaimer:** This is a simulation tool for educational purposes only. Not financial advice.

---

## 📌 Overview

An AI-powered stock market simulation assistant built with **Streamlit**, **scikit-learn**, **Plotly** and **Groq's LLaMA 3.3** model.

Upload any stock CSV or select from 15 pre-loaded real world stocks, get next-day trend predictions, view interactive technical charts, chat with AI, analyze news sentiment and download professional PDF reports.

---

## 🚀 Features

- 📁 **Stock Dropdown** — Select from 15 real US + Indian stocks instantly
- ⬆️ **Upload Your Own CSV** — Supports any OHLCV CSV file
- 🤖 **AI Trend Prediction** — Random Forest with 17 technical indicators
- 📊 **7 Interactive Charts** — Candlestick, MA, Volume, RSI, MACD, Bollinger Bands, Prediction
- 🧠 **Groq AI Explanation** — LLaMA 3.3 explains prediction in plain English
- 💬 **Stock Chat Assistant** — Conversational AI with full memory
- 📰 **News Sentiment Analysis** — Fetches latest headlines + AI sentiment score
- 📄 **PDF Report Generator** — Professional downloadable AI written report
- 🎯 **Confidence Score** — Know how sure the model is
- 🏆 **Feature Importance** — See what indicators drove the prediction
- 🌑 **Dark Trading Terminal UI** — Professional dark theme

---

## 🛠️ Tech Stack

| Technology       | Purpose                          |
|-----------------|----------------------------------|
| Python 3.9+     | Core language                    |
| Streamlit       | Web UI framework                 |
| scikit-learn    | Random Forest ML model           |
| Pandas & NumPy  | Data processing                  |
| Plotly          | Interactive charts               |
| Groq (LLaMA3.3) | AI explanation, chat, sentiment  |
| yfinance        | Real stock data fetching         |
| BeautifulSoup4  | News headline scraping           |
| ReportLab       | PDF report generation            |
| python-dotenv   | API key management               |

---

## 📁 Project Structure

```
ai-trading-assistant/
│
├── app.py                   ← Main Streamlit App
├── download_stock.py        ← Single stock downloader
├── download_all_stocks.py   ← All stocks downloader
├── requirements.txt         ← All dependencies
├── .env.example             ← API key template
├── .gitignore               ← Git ignore rules
├── README.md                ← You are here
│
├── src/
│   ├── __init__.py          ← Package init
│   ├── predictor.py         ← ML trend prediction
│   ├── charts.py            ← Plotly chart functions
│   ├── groq_explainer.py    ← Groq AI explanation
│   ├── chat.py              ← Chat with memory
│   ├── sentiment.py         ← News sentiment analysis
│   └── report.py            ← PDF report generator
│
├── data/
│   ├── AAPL_stock.csv       ← Apple
│   ├── TSLA_stock.csv       ← Tesla
│   ├── MSFT_stock.csv       ← Microsoft
│   ├── GOOGL_stock.csv      ← Google
│   ├── AMZN_stock.csv       ← Amazon
│   ├── META_stock.csv       ← Meta
│   ├── NFLX_stock.csv       ← Netflix
│   ├── NVDA_stock.csv       ← Nvidia
│   ├── RELIANCE_stock.csv   ← Reliance 🇮🇳
│   ├── TCS_stock.csv        ← TCS 🇮🇳
│   ├── WIPRO_stock.csv      ← Wipro 🇮🇳
│   ├── HDFCBANK_stock.csv   ← HDFC Bank 🇮🇳
│   ├── ICICIBANK_stock.csv  ← ICICI Bank 🇮🇳
│   └── ADANIENT_stock.csv   ← Adani 🇮🇳
│
└── assets/
    └── style.css            ← Custom dark UI styling
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/pokalabalaji113-jpg/Trading.git
cd Trading
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup API Key
```bash
# Create .env file
# Add your Groq API key inside it:
GROQ_API_KEY=your_groq_api_key_here
```

> 🔑 Get your **FREE** Groq API key at: [https://console.groq.com](https://console.groq.com)

### 5. Download Stock Data
```bash
python download_all_stocks.py
```

### 6. Run the App
```bash
streamlit run app.py
```

---

## 📊 How to Use

```
1. Run → streamlit run app.py
2. Select stock from dropdown in sidebar
   OR upload your own CSV file
3. Wait for AI prediction to complete
4. View prediction result and confidence score
5. Explore 7 interactive chart tabs
6. Click Full Analysis for Groq AI explanation
7. Click Analyze News Sentiment for news analysis
8. Chat with AI about the stock
9. Generate and download PDF report
```

---

## 📋 CSV Format

```csv
Date,Open,High,Low,Close,Volume
2024-01-01,150.00,155.00,148.00,153.00,1000000
2024-01-02,153.00,157.00,151.00,156.00,1200000
```

| Column  | Type   | Description              |
|---------|--------|--------------------------|
| Date    | String | YYYY-MM-DD format        |
| Open    | Float  | Opening price            |
| High    | Float  | Highest price of the day |
| Low     | Float  | Lowest price of the day  |
| Close   | Float  | Closing price            |
| Volume  | Int    | Number of shares traded  |

> ✅ Minimum **50 rows** required for prediction

---

## 🧠 ML Model Details

| Property         | Value                     |
|------------------|---------------------------|
| Algorithm        | Random Forest Classifier  |
| Trees            | 200 estimators            |
| Features Used    | 17 technical indicators   |
| Train/Test Split | 80% / 20%                 |
| Target           | Next-day price direction  |
| Scaling          | StandardScaler            |

### 17 Technical Indicators:
```
MA_7, MA_14, MA_21          → Moving Averages
RSI                         → Relative Strength Index
MACD, Signal_Line           → Momentum Indicator
BB_Upper, BB_Lower, BB_Width → Bollinger Bands
Daily_Return, Price_Change  → Price Movement
High_Low_Diff               → Volatility
Open_Close_Diff             → Daily Range
Volume_MA, Volume_Change    → Volume Analysis
Momentum_5, Momentum_10     → Price Momentum
```

---

## 📈 Charts Available

| Chart           | Description                             |
|----------------|-----------------------------------------|
| 🕯️ Candlestick | OHLC chart with volume bars             |
| 📈 Moving Avg  | MA7, MA14, MA21 overlaid on close price |
| 📦 Volume      | Volume bars with 7-day average          |
| ⚡ RSI         | RSI with overbought/oversold zones      |
| 📉 MACD        | MACD, Signal line and Histogram         |
| 🎯 Bollinger   | Bollinger Bands with price              |
| 🔮 Prediction  | Last 30 days + next day signal arrow    |

---

## 🤖 Gen AI Features

### 💬 Stock Chat Assistant
- Conversational AI powered by LLaMA 3.3
- Full conversation memory
- Knows current stock prediction context
- Smart suggested questions
- Ask anything about stocks, indicators, strategies

### 📰 News Sentiment Analysis
- Fetches latest stock headlines
- AI analyzes sentiment using LLaMA 3.3
- Returns POSITIVE / NEGATIVE / NEUTRAL
- Sentiment score from 0 to 100
- Compares news sentiment vs ML prediction

### 📄 PDF Report Generator
- Professional downloadable PDF
- Executive Summary
- Technical Analysis
- Prediction Analysis
- Sentiment Analysis
- Risk Assessment
- Trading Recommendation
- All written by LLaMA 3.3 in real time

---

## 🇮🇳 Supported Stocks

### 🇺🇸 US Stocks:
```
AAPL  → Apple        TSLA  → Tesla
MSFT  → Microsoft    GOOGL → Google
AMZN  → Amazon       META  → Meta
NFLX  → Netflix      NVDA  → Nvidia
```

### 🇮🇳 Indian Stocks:
```
RELIANCE → Reliance      TCS      → TCS
WIPRO    → Wipro          HDFCBANK → HDFC Bank
ICICIBANK → ICICI Bank   ADANIENT → Adani
```

---

## 🔧 Environment Variables

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 📦 Requirements

```txt
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.18.0
groq>=0.5.0
python-dotenv>=1.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
reportlab>=4.0.0
yfinance>=0.2.0
```

---

## 🚀 GitHub Setup

```bash
git init
git add .
git commit -m "🚀 Initial commit - AI Trading Assistant"
git remote add origin https://github.com/pokalabalaji113-jpg/Trading.git
git push -u origin main
```

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

```
MIT License — feel free to use, modify and distribute.
```

---

## 👨‍💻 Author

**Bala**
- GitHub: [@pokalabalaji113-jpg](https://github.com/pokalabalaji113-jpg)

---

<div align="center">

⭐ **Star this repo if you found it useful!** ⭐

Built with ❤️ using Streamlit · scikit-learn · Plotly · Groq · LLaMA 3.3

</div>
