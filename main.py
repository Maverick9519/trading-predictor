# ===== ВАЖЛИВО ДЛЯ RENDER (headless) =====
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

import logging
import io
import json
import os
import datetime
import time
import requests
import asyncio

# ===== TELEGRAM TOKEN (ENV) =====
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN not set in environment variables")

# ===== ФАЙЛИ =====
MODEL_FILE = "user_models.json"
LOG_FILE = "prediction_log.csv"

# ===== ЛОГУВАННЯ =====
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# ===== ЗАВАНТАЖЕННЯ МОДЕЛЕЙ =====
def load_user_models():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "r") as f:
            return json.load(f)
    return {}

def save_user_models(data):
    with open(MODEL_FILE, "w") as f:
        json.dump(data, f)

user_models = load_user_models()

# ===== ЛОГ ПРОГНОЗІВ =====
def log_prediction(user_id, model_type, mse, predictions, elapsed_time, total_prediction):
    df_log = pd.DataFrame([{
        "timestamp": datetime.datetime.now().isoformat(),
        "user_id": user_id,
        "model_type": model_type,
        "mse": round(mse, 4),
        "prediction_preview": list(np.round(predictions[:3], 2)),
        "prediction_sum": round(total_prediction, 2),
        "elapsed_time": round(elapsed_time, 2)
    }])

    if os.path.exists(LOG_FILE):
        df_log.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df_log.to_csv(LOG_FILE, index=False)

# ===== ЗАВАНТАЖЕННЯ КРИПТО-ДАНИХ =====
def load_crypto_data():
    url = (
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
        "?vs_currency=usd&days=100"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    prices = np.array([item[1] for item in data["prices"]])
    timestamps = [
        datetime.datetime.fromtimestamp(item[0] / 1000)
        for item in data["prices"]
    ]

    df = pd.DataFrame({"Date": timestamps, "Price": prices})

    df["Moving_Avg_10"] = df["Price"].rolling(10).mean()
    df["Moving_Avg_50"] = df["Price"].rolling(50).mean()
    df["Volatility"] = df["Price"].pct_change().rolling(10).std()

    delta = df["Price"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Target"] = df["Price"].shift(-1)
    return df.dropna()

# ===== ТРЕНУВАННЯ МОДЕЛІ =====
def train_model(df, model_type):
    features = ["Price", "Moving_Avg_10", "Moving_Avg_50", "Volatility", "RSI"]
    X = df[features]
    y = df["Target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, shuffle=False, test_size=0.2
    )

    if model_type == "SVR":
        base_model = SVR()
    elif model_type == "RandomForest":
        base_model = RandomForestRegressor(n_estimators=100)
    else:
        base_model = LinearRegression()

    model = TransformedTargetRegressor(
        regressor=base_model,
        transformer=StandardScaler()
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return model, df.loc[y_test.index], y_test, predictions, mse

# ===== ГРАФІК =====
def plot_prediction(df_test, y_test, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(df_test["Date"], y_test.values, label="Real")
    plt.plot(df_test["Date"], predictions, label="Predicted")
    plt.legend()
    plt.title("Bitcoin Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

# ===== ГОЛОВНИЙ ПРОГНОЗ =====
def get_prediction_text_and_plot(model_type, user_id):
    start = time.time()

    df = load_crypto_data()
    _, df_test, y_test, predictions, mse = train_model(df, model_type)
    plot_buf = plot_prediction(df_test, y_test, predictions)

    elapsed = time.time() - start
    total_prediction = np.sum(predictions)

    log_prediction(user_id, model_type, mse, predictions, elapsed, total_prediction)

    text = (
        f"Модель: {model_type}\n"
        f"MSE: {mse:.2f}\n"
        f"Сума прогнозу: {total_prediction:.2f}\n"
        f"Час обчислення: {elapsed:.2f} сек"
    )

    return text, plot_buf

# ===== TELEGRAM HANDLERS =====
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 Crypto прогнозатор\n\n"
        "/predict — прогноз\n"
        "/model [LinearRegression|SVR|RandomForest]\n"
        "/log — історія"
    )

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    model_type = user_models.get(user_id, "LinearRegression")

    await update.message.reply_text(
        f"⏳ Розрахунок прогнозу ({model_type})..."
    )

    try:
        loop = asyncio.get_running_loop()
        text, plot_buf = await loop.run_in_executor(
            None,
            get_prediction_text_and_plot,
            model_type,
            user_id
        )

        await update.message.reply_text(text)
        await update.message.reply_photo(photo=plot_buf)

    except Exception:
        logging.exception("Prediction error")
        await update.message.reply_text("❌ Помилка під час прогнозу.")

async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)

    if not context.args:
        await update.message.reply_text(
            "Використання: /model LinearRegression | SVR | RandomForest"
        )
        return

    model = context.args[0]
    if model not in ["LinearRegression", "SVR", "RandomForest"]:
        await update.message.reply_text("❌ Невідома модель")
        return

    user_models[user_id] = model
    save_user_models(user_models)
    await update.message.reply_text(f"✅ Модель встановлено: {model}")

async def show_log(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)

    if not os.path.exists(LOG_FILE):
        await update.message.reply_text("Журнал порожній.")
        return

    df = pd.read_csv(LOG_FILE)
    df_user = df[df["user_id"] == user_id].tail(5)

    if df_user.empty:
        await update.message.reply_text("Записів немає.")
        return

    text = "📊 Останні прогнози:\n"
    for _, row in df_user.iterrows():
        text += (
            f"\n🕒 {row['timestamp'][:19]}"
            f"\nМодель: {row['model_type']}"
            f"\nMSE: {row['mse']}"
            f"\nСума: {row['prediction_sum']}"
            f"\nЧас: {row['elapsed_time']} сек\n"
        )

    await update.message.reply_text(text)

# ===== MAIN =====
def main():
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .read_timeout(30)
        .write_timeout(30)
        .connect_timeout(30)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("model", set_model))
    app.add_handler(CommandHandler("log", show_log))

    logging.info("🤖 Bot started")
    app.run_polling()

if __name__ == "__main__":
    main()
