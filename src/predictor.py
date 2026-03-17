# src/predictor.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators as features for prediction."""

    # ── Moving Averages ──────────────────────────────────────────
    df["MA_7"]  = df["Close"].rolling(window=7).mean()
    df["MA_14"] = df["Close"].rolling(window=14).mean()
    df["MA_21"] = df["Close"].rolling(window=21).mean()

    # ── Price Change & Returns ───────────────────────────────────
    df["Daily_Return"]  = df["Close"].pct_change()
    df["Price_Change"]  = df["Close"].diff()
    df["High_Low_Diff"] = df["High"] - df["Low"]
    df["Open_Close_Diff"] = df["Close"] - df["Open"]

    # ── RSI (Relative Strength Index) ────────────────────────────
    delta     = df["Close"].diff()
    gain      = delta.clip(lower=0)
    loss      = -delta.clip(upper=0)
    avg_gain  = gain.rolling(window=14).mean()
    avg_loss  = loss.rolling(window=14).mean()
    rs        = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── MACD ─────────────────────────────────────────────────────
    ema_12      = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26      = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]  = ema_12 - ema_26
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # ── Bollinger Bands ──────────────────────────────────────────
    rolling_mean      = df["Close"].rolling(window=20).mean()
    rolling_std       = df["Close"].rolling(window=20).std()
    df["BB_Upper"]    = rolling_mean + (2 * rolling_std)
    df["BB_Lower"]    = rolling_mean - (2 * rolling_std)
    df["BB_Width"]    = df["BB_Upper"] - df["BB_Lower"]

    # ── Volume Indicators ────────────────────────────────────────
    df["Volume_MA"]    = df["Volume"].rolling(window=7).mean()
    df["Volume_Change"] = df["Volume"].pct_change()

    # ── Momentum ─────────────────────────────────────────────────
    df["Momentum_5"]  = df["Close"] - df["Close"].shift(5)
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)

    return df


def prepare_features(df: pd.DataFrame):
    """Prepare feature matrix X and target vector y."""

    df = add_technical_indicators(df.copy())

    # Target: 1 = price goes UP tomorrow, 0 = DOWN or SAME
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    feature_cols = [
        "MA_7", "MA_14", "MA_21",
        "Daily_Return", "Price_Change",
        "High_Low_Diff", "Open_Close_Diff",
        "RSI", "MACD", "Signal_Line",
        "BB_Upper", "BB_Lower", "BB_Width",
        "Volume_MA", "Volume_Change",
        "Momentum_5", "Momentum_10"
    ]

    df.dropna(inplace=True)

    X = df[feature_cols]
    y = df["Target"]

    return X, y, df


def predict_trend(df: pd.DataFrame) -> dict:
    """
    Main prediction function.
    Returns prediction result, confidence, accuracy, and enriched dataframe.
    """

    if len(df) < 50:
        return {
            "error": "Need at least 50 rows of data for prediction.",
            "prediction": None
        }

    X, y, enriched_df = prepare_features(df)

    if len(X) < 30:
        return {
            "error": "Not enough clean data after feature engineering.",
            "prediction": None
        }

    # ── Train / Test Split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # ── Scale Features ───────────────────────────────────────────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Train Random Forest ──────────────────────────────────────
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # ── Predict Next Day ─────────────────────────────────────────
    last_features = scaler.transform(X.iloc[[-1]])
    prediction    = model.predict(last_features)[0]
    probabilities = model.predict_proba(last_features)[0]
    confidence    = round(float(max(probabilities)) * 100, 2)

    # ── Feature Importance ───────────────────────────────────────
    feature_importance = dict(
        zip(X.columns, model.feature_importances_)
    )
    top_features = sorted(
        feature_importance.items(), key=lambda x: x[1], reverse=True
    )[:5]

    # ── Summary Stats ────────────────────────────────────────────
    latest_close  = round(float(df["Close"].iloc[-1]), 2)
    latest_rsi    = round(float(enriched_df["RSI"].iloc[-1]), 2)
    latest_macd   = round(float(enriched_df["MACD"].iloc[-1]), 4)
    latest_ma7    = round(float(enriched_df["MA_7"].iloc[-1]), 2)
    latest_ma21   = round(float(enriched_df["MA_21"].iloc[-1]), 2)

    return {
        "error"         : None,
        "prediction"    : "BULLISH 📈" if prediction == 1 else "BEARISH 📉",
        "signal"        : int(prediction),
        "confidence"    : confidence,
        "accuracy"      : round(accuracy * 100, 2),
        "latest_close"  : latest_close,
        "latest_rsi"    : latest_rsi,
        "latest_macd"   : latest_macd,
        "latest_ma7"    : latest_ma7,
        "latest_ma21"   : latest_ma21,
        "top_features"  : top_features,
        "enriched_df"   : enriched_df
    }