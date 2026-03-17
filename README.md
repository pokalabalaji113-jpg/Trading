# 🤖 AI Trading Assistant (Simulation)

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=for-the-badge&logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.3+-orange?style=for-the-badge&logo=scikit-learn)
![Groq](https://img.shields.io/badge/Groq-LLaMA3-purple?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-5.18+-green?style=for-the-badge&logo=plotly)

> ⚠️ **Disclaimer:** This is a simulation tool for educational purposes only. Not financial advice.

---

## 📌 Overview

An AI-powered stock market simulation assistant built with **Streamlit**, **scikit-learn**, **Plotly**, and **Groq LLaMA3**. Upload any stock CSV, get next-day trend predictions, view interactive technical charts, and receive AI-generated market explanations.

---

## 🚀 Features

- 📁 **Upload Stock Data** — Supports any OHLCV CSV file
- 🤖 **AI Trend Prediction** — Random Forest with 17 technical indicators
- 📊 **7 Interactive Charts** — Candlestick, MA, Volume, RSI, MACD, Bollinger Bands, Prediction
- 🧠 **Groq AI Explanation** — LLaMA3-70B explains the prediction in plain English
- 🎯 **Confidence Score** — Know how sure the model is
- 🏆 **Feature Importance** — See what indicators drove the prediction
- 📥 **Sample CSV** — Built-in sample data generator
- 🌑 **Dark Trading Terminal UI** — Professional dark theme

---

## 🛠️ Tech Stack

| Technology       | Purpose                        |
|-----------------|-------------------------------|
| Python 3.9+     | Core language                  |
| Streamlit       | Web UI framework               |
| scikit-learn    | Random Forest ML model         |
| Pandas & NumPy  | Data processing                |
| Plotly          | Interactive charts             |
| Groq (LLaMA3)   | AI market explanation          |
| python-dotenv   | API key management             |

---

## 📁 Project Structure
```
ai-trading-assistant/
│
├── app.py                  ← Main Streamlit App
├── requirements.txt        ← All dependencies
├── .env.example            ← API key template
├── .gitignore              ← Git ignore rules
├── README.md               ← You are here
│
├── src/
│   ├── __init__.py         ← Package init
│   ├── predictor.py        ← ML trend prediction
│   ├── charts.py           ← Plotly chart functions
│   └── groq_explainer.py   ← Groq AI explanation
│
├── data/
│   └── sample_stock.csv    ← Sample test data
│
└── assets/
    └── style.css           ← Custom dark UI styling
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-trading-assistant.git
cd ai-trading-assistant
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
# Copy the example env file
cp .env.example .env

# Open .env and add your Groq API key
GROQ_API_KEY=your_groq_api_key_here
```

> 🔑 Get your **FREE** Groq API key at: [https://console.groq.com](https://console.groq.com)

### 5. Run the App
```bash
streamlit run app.py
```

---

## 📊 How to Use
```
1. Launch the app with: streamlit run app.py
2. Download sample CSV from the sidebar (or use your own)
3. Upload your stock CSV file
4. Wait for AI prediction to complete
5. View prediction result, confidence score & metrics
6. Explore 7 interactive chart tabs
7. Click "Full Analysis" for Groq AI explanation
```

---

## 📋 CSV Format

Your CSV file must have these exact columns:
```csv
Date,Open,High,Low,Close,Volume
2024-01-01,150.00,155.00,148.00,153.00,1000000
2024-01-02,153.00,157.00,151.00,156.00,1200000
...
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

| Property          | Value                        |
|-------------------|------------------------------|
| Algorithm         | Random Forest Classifier     |
| Trees             | 200 estimators               |
| Features Used     | 17 technical indicators      |
| Train/Test Split  | 80% / 20%                    |
| Target            | Next-day price direction     |
| Scaling           | StandardScaler               |

### 17 Technical Indicators:
```
MA_7, MA_14, MA_21         → Moving Averages
RSI                        → Relative Strength Index
MACD, Signal_Line          → Momentum Indicator
BB_Upper, BB_Lower, BB_Width → Bollinger Bands
Daily_Return, Price_Change → Price Movement
High_Low_Diff              → Volatility
Open_Close_Diff            → Daily Range
Volume_MA, Volume_Change   → Volume Analysis
Momentum_5, Momentum_10    → Price Momentum
```

---

## 📈 Charts Available

| Chart            | Description                              |
|-----------------|------------------------------------------|
| 🕯️ Candlestick  | OHLC chart with volume bars              |
| 📈 Moving Avg   | MA7, MA14, MA21 overlaid on close price  |
| 📦 Volume       | Volume bars with 7-day average           |
| ⚡ RSI          | RSI with overbought/oversold zones       |
| 📉 MACD         | MACD, Signal line & Histogram            |
| 🎯 Bollinger    | Bollinger Bands with price               |
| 🔮 Prediction   | Last 30 days + next day signal arrow     |

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
```

---

## 🚀 GitHub Setup
```bash
# Initialize git
git init

# Add all files
git add .

# First commit
git commit -m "🚀 Initial commit - AI Trading Assistant"

# Add remote origin
git remote add origin https://github.com/yourusername/ai-trading-assistant.git

# Push to GitHub
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
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [your-linkedin](https://linkedin.com/in/yourprofile)

---

<div align="center">

⭐ **Star this repo if you found it useful!** ⭐

Built with ❤️ using Streamlit · scikit-learn · Plotly · Groq

</div>
```

---

✅ **README includes:**

| Section | Details |
|---------|---------|
| 🏷️ Badges | Python, Streamlit, Sklearn, Groq, Plotly |
| 📌 Overview | Clear project description |
| 🚀 Features | All 8 features listed |
| 🛠️ Tech Stack | Full table |
| 📁 Structure | Folder tree |
| ⚙️ Setup | Step by step install guide |
| 📋 CSV Format | Example + column table |
| 🧠 ML Details | Model info + all 17 indicators |
| 📈 Charts | All 7 charts described |
| 🚀 GitHub | Push commands ready |

---

## 🎉 ALL 11 FILES COMPLETE!

Here's the **final checklist** before running:
```
✅ src/__init__.py
✅ src/predictor.py
✅ src/charts.py
✅ src/groq_explainer.py
✅ assets/style.css
✅ app.py
✅ README.md
⬜ .gitignore        ← still needed!
⬜ .env.example      ← still needed!
⬜ requirements.txt  ← still needed!
⬜ data/sample_stock.csv ← auto generated in app!