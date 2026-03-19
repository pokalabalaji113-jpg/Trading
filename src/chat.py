# src/chat.py

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


# ══════════════════════════════════════════════════════════════════════════
#  CHAT WITH MEMORY
# ══════════════════════════════════════════════════════════════════════════

def get_chat_response(
    user_message: str,
    chat_history: list,
    result: dict = None,
    stock_label: str = ""
) -> str:
    """
    Send user message to Groq LLaMA with full chat history (memory).
    Knows about current stock prediction context.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return "⚠️ Groq API key not found! Please add it to .env file."

    # ── Build Stock Context ───────────────────────────────────────
    stock_context = ""
    if result:
        stock_context = f"""
Current Stock Analysis Context:
- Stock         : {stock_label}
- Prediction    : {result.get('prediction', 'N/A')}
- Confidence    : {result.get('confidence', 0)}%
- Model Accuracy: {result.get('accuracy', 0)}%
- Close Price   : ${result.get('latest_close', 0)}
- RSI (14)      : {result.get('latest_rsi', 0)}
- MACD          : {result.get('latest_macd', 0)}
- MA7           : {result.get('latest_ma7', 0)}
- MA21          : {result.get('latest_ma21', 0)}
- Top Indicators: {result.get('top_features', [])}
"""

    # ── System Prompt ─────────────────────────────────────────────
    system_prompt = f"""
You are an expert AI Stock Market Assistant with deep knowledge of:
- Technical Analysis (RSI, MACD, Bollinger Bands, Moving Averages)
- Stock market trends and patterns
- Trading strategies and risk management
- Financial markets worldwide

{stock_context}

Your job:
- Answer user questions about stocks, trading, indicators clearly
- Reference the current stock analysis context when relevant
- Be concise, professional and beginner friendly
- Use emojis to make responses engaging
- Always remind this is simulation — not financial advice

Important:
- Remember the full conversation history
- Build on previous messages naturally
- If user asks "why?" refer back to previous answer
"""

    # ── Build Messages with History ───────────────────────────────
    messages = [{"role": "system", "content": system_prompt}]

    # Add chat history (memory!)
    for chat in chat_history:
        messages.append({
            "role"   : chat["role"],
            "content": chat["content"]
        })

    # Add current user message
    messages.append({
        "role"   : "user",
        "content": user_message
    })

    # ── Call Groq API ─────────────────────────────────────────────
    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            messages    = messages,
            model       = "llama-3.3-70b-versatile",
            temperature = 0.7,
            max_tokens  = 800,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"⚠️ **Error:** {str(e)}"


# ══════════════════════════════════════════════════════════════════════════
#  SUGGESTED QUESTIONS
# ══════════════════════════════════════════════════════════════════════════

def get_suggested_questions(result: dict, stock_label: str) -> list:
    """Return smart suggested questions based on prediction."""

    prediction = result.get("prediction", "")
    rsi        = result.get("latest_rsi", 50)
    is_bullish = result.get("signal", 0) == 1

    base_questions = [
        f"Why is {stock_label} showing {prediction}?",
        f"What does RSI {rsi} mean for {stock_label}?",
        f"Is it safe to {'buy' if is_bullish else 'sell'} {stock_label} now?",
        f"What are the risks of trading {stock_label}?",
        "Explain MACD in simple terms",
        "What is Bollinger Band strategy?",
        "How reliable is this prediction?",
        "What is a good stop loss strategy?",
    ]

    return base_questions