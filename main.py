import os
import io
import time
import json
import logging
import datetime
import requests
import asyncio

from dotenv import load_dotenv  # <-- ДОДАНО

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
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ================= LOAD ENV =================
load_dotenv()  # <-- ПІДКЛЮЧЕННЯ .env

# ================= CONFIG =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("❌ TELEGRAM_TOKEN не встановлено (.env або system env)")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ================= CACHE =================
DATA_CACHE = None
LAST_LOAD = 0
CACHE_TTL = 3600  # 1 година
CACHE_FILE = "btc_cache.json"

# ================= USER COOLDOWN =================
USER_COOLDOWN = 300  # 5 хв
last_call = {}

# ================= COINGECKO FETCH WITH BACKOFF =================
def fetch_coingecko_data(url, params):
    for wait in (1, 3, 5):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:
                logging.warning(f"429 Too Many Requests, sleeping {wait}s")
                time.sleep(wait)
            else:
                r.raise_for_status()
        except requests.RequestException as e:
            logging.warning(f"CoinGecko request failed: {e}, sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError("CoinGecko тимчасово недоступний (429 або помилка мережі)")

# ================= LOAD CRYPTO DATA =================
def load_crypto_data():
    global DATA_CACHE, LAST_LOAD
    now = time.time()

    if DATA_CACHE is not None and now - LAST_LOAD < CACHE_TTL:
        logging.info("📦 Using in-memory cache")
        return DATA_CACHE

    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            df["Date"] = pd.to_datetime(df["Date"])
            logging.info("📦 Using local cache file")
            DATA_CACHE = df
            LAST_LOAD = now
            return df
        except Exception:
            logging.warning("Не вдалося завантажити локальний cache file")

    logging.info("🌐 Loading data from CoinGecko")
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {
        "vs_currency": "usd",
        "days": "7",
        "interval": "hourly"
    }

    r = fetch_coingecko_data(url, params)
    data = r.json()

    if "prices" not in data or len(data["prices"]) < 50:
        raise RuntimeError("CoinGecko не повернув достатньо даних")

    rows = [
        {"Date": datetime.datetime.fromtimestamp(ts / 1000), "Price": price}
        for ts, price in data["prices"]
    ]

    df = pd.DataFrame(rows)
    df["MA10"] = df["Price"].rolling(10).mean()
    df["MA30"] = df["Price"].rolling(30).mean()
    df["Volatility"] = df["Price"].pct_change().rolling(10).std()
    df["Target"] = df["Price"].shift(-1)
    df = df.dropna()

    if len(df) < 30:
        raise RuntimeError("Недостатньо даних після обробки")

    DATA_CACHE = df
    LAST_LOAD = now
    try:
        df.to_json(CACHE_FILE, orient="records", date_format="iso")
    except Exception:
        logging.warning("Не вдалося зберегти локальний кеш")

    return df

# ================= ML TRAIN =================
def train_model(df):
    features = ["Price", "MA10", "MA30", "Volatility"]
    X = df[features]
    y = df["Target"]

    X_scaled = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    model = TransformedTargetRegressor(
        regressor=LinearRegression(),
        transformer=StandardScaler()
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)

    return df.iloc[-len(y_test):], y_test, preds, mse

# ================= PLOT =================
def plot_prediction(df_test, y_test, preds):
    plt.figure(figsize=(10, 5))
    plt.plot(df_test["Date"], y_test.values, label="Real")
    plt.plot(df_test["Date"], preds, label="Predicted")
    plt.legend()
    plt.title("BTC Price Prediction (CoinGecko)")
    plt.xticks(rotation=45)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

# ================= CORE PREDICTION =================
def make_prediction():
    start = time.time()
    df = load_crypto_data()
    df_test, y_test, preds, mse = train_model(df)
    plot = plot_prediction(df_test, y_test, preds)

    text = (
        f"📈 BTC прогноз (CoinGecko)\n"
        f"Прогноз наступної години: {preds[-1]:.2f} USD\n"
        f"MSE: {mse:.2f}\n"
        f"Час розрахунку: {time.time() - start:.2f} сек"
    )
    return text, plot

# ================= TELEGRAM HANDLERS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    logging.info(f"START | user_id={user.id} username=@{user.username}")
    await update.message.reply_text(
        "🤖 Crypto прогнозатор (CoinGecko)\n"
        "/predict — отримати прогноз"
    )

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    now = time.time()

    if user.id in last_call and now - last_call[user.id] < USER_COOLDOWN:
        await update.message.reply_text(
            f"⏳ Зачекай {USER_COOLDOWN // 60} хв між запитами"
        )
        return

    last_call[user.id] = now
    logging.info(f"PREDICT | user_id={user.id} username=@{user.username}")

    await update.message.reply_text("⏳ Розрахунок прогнозу...")

    try:
        loop = asyncio.get_running_loop()
        text, plot = await loop.run_in_executor(None, make_prediction)
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
