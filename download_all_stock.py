# download_all_stocks.py

import yfinance as yf
import pandas as pd
import os

# ── Create data folder if not exists ─────────────────────────────────────
os.makedirs("data", exist_ok=True)

# ── All Stocks List ───────────────────────────────────────────────────────
stocks = {
    # 🇺🇸 US Stocks
    "AAPL"  : "Apple",
    "TSLA"  : "Tesla",
    "MSFT"  : "Microsoft",
    "GOOGL" : "Google",
    "AMZN"  : "Amazon",
    "META"  : "Meta Facebook",
    "NFLX"  : "Netflix",
    "NVDA"  : "Nvidia",

    # 🇮🇳 Indian Stocks
    "RELIANCE.NS" : "Reliance",
    "TCS.NS"      : "TCS",
    "INFY.NS"     : "Infosys",
    "WIPRO.NS"    : "Wipro",
    "HDFCBANK.NS" : "HDFC Bank",
    "ICICIBANK.NS": "ICICI Bank",
    "ADANIENT.NS" : "Adani Enterprise",
}

# ── Download Each Stock ───────────────────────────────────────────────────
print("=" * 50)
print("   📥 Downloading All Stock Data...")
print("=" * 50)

success = []
failed  = []

for ticker, name in stocks.items():
    try:
        print(f"\n⏳ Downloading {name} ({ticker})...")

        df = yf.download(
            ticker,
            start="2024-01-01",
            end="2026-03-17",
            progress=False
        )

        if len(df) < 50:
            print(f"❌ {name} — Not enough data!")
            failed.append(ticker)
            continue

        # Reset index
        df.reset_index(inplace=True)

        # Keep only needed columns
        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        # Flatten columns
        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

        # Remove empty rows
        df.dropna(inplace=True)

        # Save CSV
        filename = f"data/{ticker.replace('.NS', '')}_stock.csv"
        df.to_csv(filename, index=False)

        print(f"✅ {name} — {len(df)} rows saved → {filename}")
        success.append(f"{name} ({ticker})")

    except Exception as e:
        print(f"❌ {name} failed: {e}")
        failed.append(ticker)

# ── Summary ───────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("   📊 Download Summary")
print("=" * 50)
print(f"\n✅ Success ({len(success)}):")
for s in success:
    print(f"   → {s}")

if failed:
    print(f"\n❌ Failed ({len(failed)}):")
    for f in failed:
        print(f"   → {f}")

print("\n🎉 All done! Upload any CSV to your app!")
print("=" * 50)