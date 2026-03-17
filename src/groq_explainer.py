import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def get_groq_explanation(result: dict, df_tail: dict = None) -> str:
    """
    Send prediction result to Groq LLM and get
    a detailed trading explanation back.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return (
            "⚠️ **Groq API key not found!**\n\n"
            "Please add your `GROQ_API_KEY` in the `.env` file.\n"
            "Get your free key at: https://console.groq.com"
        )

    # ── Build Context from Prediction Result ─────────────────────
    prediction    = result.get("prediction", "N/A")
    confidence    = result.get("confidence", 0)
    accuracy      = result.get("accuracy", 0)
    latest_close  = result.get("latest_close", 0)
    latest_rsi    = result.get("latest_rsi", 0)
    latest_macd   = result.get("latest_macd", 0)
    latest_ma7    = result.get("latest_ma7", 0)
    latest_ma21   = result.get("latest_ma21", 0)
    top_features  = result.get("top_features", [])

    # Format top features nicely
    features_text = "\n".join([
        f"  - {feat}: {round(imp * 100, 2)}% importance"
        for feat, imp in top_features
    ])

    # ── Build the Prompt ─────────────────────────────────────────
    prompt = f"""
You are an expert stock market analyst and trading advisor.
Analyze the following AI model prediction results and provide
a detailed, professional trading explanation.

## Prediction Summary:
- **Signal**      : {prediction}
- **Confidence**  : {confidence}%
- **Model Accuracy** : {accuracy}%

## Latest Technical Indicators:
- **Close Price** : {latest_close}
- **RSI (14)**    : {latest_rsi}
- **MACD**        : {latest_macd}
- **MA 7**        : {latest_ma7}
- **MA 21**       : {latest_ma21}

## Top 5 Features That Influenced Prediction:
{features_text}

---

Please provide:

1. **📊 Market Analysis** — What do these indicators say about the current market condition?

2. **🎯 Prediction Explanation** — Why is the model predicting {prediction}? Explain in simple terms.

3. **⚡ RSI Interpretation** — Is the stock overbought, oversold, or neutral? What does it mean?

4. **📈 MACD Signal** — What does the MACD value indicate? Bullish or bearish crossover?

5. **📉 Moving Average Trend** — What does MA7 vs MA21 relationship tell us?

6. **⚠️ Risk Factors** — What risks should a trader be aware of?

7. **💡 Trading Suggestion** — Based on all indicators, what is the overall trading suggestion?

⚠️ Always end with: "This is a simulation for educational purposes only. Not financial advice."

Keep the tone professional, clear, and beginner-friendly.
Use emojis to make it engaging. Format with proper headings.
"""

    # ── Call Groq API ─────────────────────────────────────────────
    try:
        client = Groq(api_key=api_key)

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional stock market analyst. "
                        "You explain AI trading predictions clearly, "
                        "concisely, and in a beginner-friendly way. "
                        "You always remind users this is for simulation only."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",   # Fast + powerful free model on Groq
            temperature=0.7,
            max_tokens=1500,
        )

        explanation = chat_completion.choices[0].message.content
        return explanation

    except Exception as e:
        return (
            f"⚠️ **Groq API Error:**\n\n"
            f"```\n{str(e)}\n```\n\n"
            f"Please check your API key and internet connection."
        )


def get_quick_summary(result: dict) -> str:
    """
    Get a short 3-line summary from Groq for dashboard display.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "⚠️ Groq API key not found."

    prediction = result.get("prediction", "N/A")
    confidence = result.get("confidence", 0)
    rsi        = result.get("latest_rsi", 0)
    macd       = result.get("latest_macd", 0)

    prompt = f"""
Given:
- Prediction: {prediction}
- Confidence: {confidence}%
- RSI: {rsi}
- MACD: {macd}

Give exactly 3 short bullet points (1 line each) summarizing
the trading signal. Be direct and professional. Use emojis.
No extra text, just 3 bullets.
"""

    try:
        client = Groq(api_key=api_key)

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=200,
        )

        return chat_completion.choices[0].message.content

    except Exception as e:
        return f"⚠️ Error: {str(e)}"