# ================= SYSTEM =================
import os
import io
import time
import logging
import datetime
import requests
import asyncio

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes
)

# ================= CONFIG =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN не встановлено")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ================= CACHE =================
DATA_CACHE = None
LAST_LOAD = 0
CACHE_TTL = 300  # 5 хвилин

# ================= COINGECKO =================
def load_crypto_data():
    global DATA_CACHE, LAST_LOAD

    if DATA_CACHE is not None and time.time() - LAST_LOAD < CACHE_TTL:
        logging.info("📦 Using cached CoinGecko data")
        return DATA_CACHE

    logging.info("🌐 Loading data from CoinGecko")

    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "7",
        "interval": "hourly"
    }

    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    if "prices" not in data or len(data["prices"]) < 50:
        raise RuntimeError("CoinGecko не повернув достатньо даних")

    rows = []
    for ts, price in data["prices"]:
        rows.append({
            "Date": datetime.datetime.fromtimestamp(ts / 1000),
            "Price": price
        })

    df = pd.DataFrame(rows)

    df["MA10"] = df["Price"].rolling(10).mean()
    df["MA30"] = df["Price"].rolling(30).mean()
    df["Volatility"] = df["Price"].pct_change().rolling(10).std()
    df["Target"] = df["Price"].shift(-1)

    df = df.dropna()
    if len(df) < 30:
        raise RuntimeError("Недостатньо даних після обробки")

    DATA_CACHE = df
    LAST_LOAD = time.time()
    return df

# ================= ML =================
def train_model(df):
    features = ["Price", "MA10", "MA30", "Volatility"]
    X = df[features]
    y = df["Target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    model = TransformedTargetRegressor(
        regressor=LinearRegression(),
        transformer=StandardScaler()
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return df.iloc[-len(y_test):], y_test, predictions, mse

# ================= PLOT =================
def plot_prediction(df_test, y_test, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(df_test["Date"], y_test.values, label="Real")
    plt.plot(df_test["Date"], predictions, label="Predicted")
    plt.legend()
    plt.title("BTC Price Prediction (CoinGecko)")
    plt.xticks(rotation=45)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

# ================= CORE =================
def make_prediction(user_id):
    start = time.time()

    df = load_crypto_data()
    df_test, y_test, preds, mse = train_model(df)
    plot = plot_prediction(df_test, y_test, preds)

    elapsed = time.time() - start
    next_price = preds[-1]

    text = (
        f"📈 BTC прогноз (CoinGecko)\n"
        f"Прогноз наступної години: {next_price:.2f} USD\n"
        f"MSE: {mse:.2f}\n"
        f"Час розрахунку: {elapsed:.2f} сек"
    )

    return text, plot

# ================= TELEGRAM =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Crypto прогнозатор (CoinGecko, реальні дані)\n"
        "/predict — отримати прогноз"
    )

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    await update.message.reply_text("⏳ Розрахунок прогнозу...")

    try:
        loop = asyncio.get_running_loop()
        text, plot = await loop.run_in_executor(None, make_prediction, user_id)
        await update.message.reply_text(text)
        await update.message.reply_photo(photo=plot)
    except Exception as e:
        logging.exception("Prediction failed")
        await update.message.reply_text(f"❌ Помилка:\n{e}")

# ================= MAIN =================
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))

    logging.info("🤖 Bot started (polling)")
    app.run_polling()

if __name__ == "__main__":
    main()
