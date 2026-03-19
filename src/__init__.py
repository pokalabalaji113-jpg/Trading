# src/__init__.py

from .predictor import predict_trend
from .charts import (
    plot_candlestick,
    plot_moving_averages,
    plot_volume,
    plot_rsi,
    plot_prediction
)
from .groq_explainer import get_groq_explanation

__all__ = [
    "predict_trend",
    "plot_candlestick",
    "plot_moving_averages",
    "plot_volume",
    "plot_rsi",
    "plot_prediction",
    "get_groq_explanation"
]# src/__init__.py

from .predictor import predict_trend
from .charts import (
    plot_candlestick,
    plot_moving_averages,
    plot_volume,
    plot_rsi,
    plot_macd,
    plot_bollinger_bands,
    plot_prediction
)
from .groq_explainer import get_groq_explanation, get_quick_summary
from .chat import get_chat_response, get_suggested_questions
from .sentiment import (
    analyze_sentiment,
    get_sentiment_color,
    get_sentiment_emoji
)
from .report import generate_pdf_report

__all__ = [
    # Predictor
    "predict_trend",

    # Charts
    "plot_candlestick",
    "plot_moving_averages",
    "plot_volume",
    "plot_rsi",
    "plot_macd",
    "plot_bollinger_bands",
    "plot_prediction",

    # Groq Explainer
    "get_groq_explanation",
    "get_quick_summary",

    # Chat
    "get_chat_response",
    "get_suggested_questions",

    # Sentiment
    "analyze_sentiment",
    "get_sentiment_color",
    "get_sentiment_emoji",

    # Report
    "generate_pdf_report",
]