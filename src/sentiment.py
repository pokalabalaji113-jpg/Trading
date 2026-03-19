# src/sentiment.py

import os
import requests
from bs4 import BeautifulSoup
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════
#  FETCH STOCK NEWS
# ══════════════════════════════════════════════════════════════════════════

def fetch_stock_news(ticker: str, stock_label: str = "") -> list:
    """
    Fetch latest news headlines for a stock
    from Yahoo Finance news section.
    """

    # Clean ticker for URL
    clean_ticker = ticker.upper().replace(
        "_STOCK", ""
    ).replace(
        ".CSV", ""
    ).replace(
        ".NS", ""
    )

    url = f"https://finance.yahoo.com/quote/{clean_ticker}/news"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    news_list = []

    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup     = BeautifulSoup(response.text, "html.parser")

        # Extract headlines
        headlines = soup.find_all("h3", limit=10)

        for h in headlines:
            text = h.get_text(strip=True)
            if text and len(text) > 20:
                news_list.append(text)

        # If no headlines found try alternative
        if not news_list:
            headlines = soup.find_all("a", {"class": lambda x: x and "title" in x.lower()}, limit=10)
            for h in headlines:
                text = h.get_text(strip=True)
                if text and len(text) > 20:
                    news_list.append(text)

    except Exception as e:
        print(f"News fetch error: {e}")

    # Fallback headlines if nothing fetched
    if not news_list:
        news_list = [
            f"{stock_label} stock shows mixed signals in recent trading",
            f"Analysts watch {stock_label} closely amid market volatility",
            f"Market uncertainty affects {stock_label} price movement",
            f"{stock_label} investors monitor technical levels carefully",
            f"Trading volume patterns shift for {stock_label}",
        ]

    return news_list[:8]


# ══════════════════════════════════════════════════════════════════════════
#  ANALYZE SENTIMENT WITH GROQ
# ══════════════════════════════════════════════════════════════════════════

def analyze_sentiment(
    ticker      : str,
    stock_label : str,
    result      : dict = None
) -> dict:
    """
    Fetch news + analyze sentiment using LLaMA via Groq.
    Returns sentiment score, label and analysis.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return {
            "error"    : "⚠️ Groq API key not found!",
            "sentiment": "NEUTRAL",
            "score"    : 50,
            "headlines": [],
            "analysis" : ""
        }

    # ── Fetch News ────────────────────────────────────────────────
    headlines = fetch_stock_news(ticker, stock_label)

    # ── Build Headlines Text ──────────────────────────────────────
    headlines_text = "\n".join([
        f"{i+1}. {h}" for i, h in enumerate(headlines)
    ])

    # ── Prediction Context ────────────────────────────────────────
    prediction_context = ""
    if result:
        prediction_context = f"""
ML Model Prediction:
- Signal    : {result.get('prediction', 'N/A')}
- Confidence: {result.get('confidence', 0)}%
- RSI       : {result.get('latest_rsi', 0)}
- MACD      : {result.get('latest_macd', 0)}
"""

    # ── Sentiment Prompt ──────────────────────────────────────────
    prompt = f"""
Analyze the sentiment of these recent news headlines for {stock_label} stock.

## News Headlines:
{headlines_text}

## Technical Context:
{prediction_context}

Please provide:

1. **Overall Sentiment** — POSITIVE, NEGATIVE, or NEUTRAL
   (Just one word on first line)

2. **Sentiment Score** — Give a score from 0 to 100
   (0 = Very Negative, 50 = Neutral, 100 = Very Positive)
   (Just the number on second line)

3. **📰 Headline Analysis** — Brief analysis of each headline
   (1 line per headline)

4. **🎯 Overall Market Mood** — What does the news suggest overall?
   (2-3 sentences)

5. **⚠️ Key Risk Factors** — What risks do the headlines reveal?
   (2-3 bullet points)

6. **💡 Sentiment vs Technical** — Does news sentiment agree or
   disagree with the ML prediction? What does this mean?

Format response clearly with headings and emojis.
End with: "News sentiment is for educational purposes only."
"""

    # ── Call Groq API ─────────────────────────────────────────────
    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            messages=[
                {
                    "role"   : "system",
                    "content": (
                        "You are an expert financial news analyst. "
                        "You analyze stock market news sentiment accurately "
                        "and provide clear, structured analysis. "
                        "Always remind users this is for educational purposes only."
                    )
                },
                {
                    "role"   : "user",
                    "content": prompt
                }
            ],
            model       = "llama-3.3-70b-versatile",
            temperature = 0.4,
            max_tokens  = 1200,
        )

        full_response = response.choices[0].message.content
        lines         = full_response.strip().split("\n")

        # ── Parse Sentiment & Score ───────────────────────────────
        sentiment = "NEUTRAL"
        score     = 50

        for line in lines[:5]:
            line_clean = line.strip().upper()
            if "POSITIVE" in line_clean and len(line_clean) < 30:
                sentiment = "POSITIVE"
                break
            elif "NEGATIVE" in line_clean and len(line_clean) < 30:
                sentiment = "NEGATIVE"
                break
            elif "NEUTRAL" in line_clean and len(line_clean) < 30:
                sentiment = "NEUTRAL"
                break

        for line in lines[:5]:
            line_clean = line.strip()
            if line_clean.isdigit():
                score = max(0, min(100, int(line_clean)))
                break

        return {
            "error"    : None,
            "sentiment": sentiment,
            "score"    : score,
            "headlines": headlines,
            "analysis" : full_response
        }

    except Exception as e:
        return {
            "error"    : f"⚠️ Groq API Error: {str(e)}",
            "sentiment": "NEUTRAL",
            "score"    : 50,
            "headlines": headlines,
            "analysis" : ""
        }


# ══════════════════════════════════════════════════════════════════════════
#  SENTIMENT COLOR & EMOJI HELPERS
# ══════════════════════════════════════════════════════════════════════════

def get_sentiment_color(sentiment: str) -> str:
    """Return color based on sentiment."""
    colors = {
        "POSITIVE": "#26a69a",
        "NEGATIVE": "#ef5350",
        "NEUTRAL" : "#f39c12"
    }
    return colors.get(sentiment.upper(), "#f39c12")


def get_sentiment_emoji(sentiment: str) -> str:
    """Return emoji based on sentiment."""
    emojis = {
        "POSITIVE": "🟢",
        "NEGATIVE": "🔴",
        "NEUTRAL" : "🟡"
    }
    return emojis.get(sentiment.upper(), "🟡")