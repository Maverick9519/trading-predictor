import os
import threading
import logging
import asyncio
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from prophet import Prophet

# --- Налаштування
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)

# --- Завантаження історичних даних BTC
def fetch_historical_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1d",
        "limit": 100
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    raw_data = response.json()
    df = pd.DataFrame(raw_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df["close"] = df["close"].astype(float)
    df = df[["timestamp", "close"]]
    df.rename(columns={"timestamp": "ds", "close": "y"}, inplace=True)
    return df

# --- Графік Prophet
def plot_forecast(model, forecast):
    fig = model.plot(forecast)
    plt.title("Bitcoin (прогноз Prophet)")
    plt.xlabel("Дата")
    plt.ylabel("Ціна (USD)")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# --- Графік RandomForest
def plot_rf_forecast(df, predicted_price):
    plt.figure(figsize=(10, 5))
    plt.plot(df["ds"], df["y"], label="Історія")
    plt.scatter(pd.Timestamp.now() + pd.Timedelta(days=1), predicted_price, color="red", label="Прогноз")
    plt.xlabel("Дата")
    plt.ylabel("Ціна (USD)")
    plt.title("Bitcoin: Прогноз (Random Forest)")
    plt.legend()
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# --- Підготовка фічей для RandomForest
def prepare_features(df):
    df["day"] = df["ds"].dt.day
    df["month"] = df["ds"].dt.month
    df["year"] = df["ds"].dt.year
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["lag1"] = df["y"].shift(1)
    df["lag2"] = df["y"].shift(2)
    df = df.dropna()
    features = ["day", "month", "year", "dayofweek", "lag1", "lag2"]
    return df[features], df["y"], features

# --- Людський фактор
def apply_human_factor(price: float, mood: str = "neutral") -> float:
    mood = mood.lower()
    factors = {
        "fear": -0.03,
        "greed": 0.04,
        "panic": -0.07,
        "euphoria": 0.08,
        "neutral": 0.0
    }
    adjustment = factors.get(mood, 0.0)
    return price * (1 + adjustment)

# --- Команда /predict
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        model_name = "prophet"
        mood = "neutral"

        if context.args:
            for arg in context.args:
                if arg.startswith("model="):
                    model_name = arg.split("=")[-1].lower()
                elif arg.startswith("mood="):
                    mood = arg.split("=")[-1].lower()

        df = fetch_historical_data()
        now_price = df["y"].iloc[-1]

        if model_name == "randomforest":
            X, y, feature_cols = prepare_features(df)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_scaled, y)

            last_row = df.iloc[-1:]
            last_features = {
                "day": last_row["ds"].dt.day.values[0],
                "month": last_row["ds"].dt.month.values[0],
                "year": last_row["ds"].dt.year.values[0],
                "dayofweek": last_row["ds"].dt.dayofweek.values[0],
                "lag1": last_row["y"].values[0],
                "lag2": df.iloc[-2]["y"]
            }
            X_pred = pd.DataFrame([last_features])[feature_cols]
            X_pred_scaled = scaler.transform(X_pred)

            predicted_price = model.predict(X_pred_scaled)[0]
            model_label = "Random Forest"
            plot_buf = plot_rf_forecast(df, predicted_price)

        else:
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=1)
            forecast = model.predict(future)
            predicted_price = forecast.iloc[-1]["yhat"]
            model_label = "Prophet"
            plot_buf = plot_forecast(model, forecast)

        predicted_price = apply_human_factor(predicted_price, mood)

        text = (
            f"\U0001F4CA Поточна ціна: ${now_price:.2f}\n"
            f"\U0001F52E Прогноз ({model_label}): ${predicted_price:.2f}\n"
            f"Фактор настрою: {mood}"
        )

        await update.message.reply_text(text)
        await update.message.reply_photo(photo=plot_buf)

    except Exception as e:
        logging.exception("❗️Помилка прогнозу")
        await update.message.reply_text(f"❌ Помилка: {e}")

# --- Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Я трейдинг-прогнозатор бот.\n"
        "Команди:\n"
        "/predict — прогноз (model=prophet або model=randomforest, mood=greed/fear/neutral...)\n"
        "/auto [хв] — авто-прогноз\n"
        "/stop — зупинити авто-прогноз"
    )

# --- Авто-прогноз (тільки Prophet)
auto_tasks = {}

async def auto_predict(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id
    try:
        df = fetch_historical_data()
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)

        predicted_price = forecast.iloc[-1]["yhat"]
        now_price = df["y"].iloc[-1]
        plot_buf = plot_forecast(model, forecast)

        text = (
            f"\U0001F4CA Поточна ціна: ${now_price:.2f}\n"
            f"\U0001F52E Прогноз (Prophet): ${predicted_price:.2f}"
        )
        await context.bot.send_message(chat_id=chat_id, text=text)
        await context.bot.send_photo(chat_id=chat_id, photo=plot_buf)
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Помилка: {e}")

async def auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) != 1:
            await update.message.reply_text("❗️ Формат: /auto 5 (хвилин)")
            return
        minutes = int(context.args[0])
        chat_id = update.effective_chat.id
        if chat_id in auto_tasks:
            auto_tasks[chat_id].schedule_removal()
        job = context.job_queue.run_repeating(auto_predict, interval=minutes * 60, first=0, chat_id=chat_id)
        auto_tasks[chat_id] = job
        await update.message.reply_text(f"✅ Авто-прогноз кожні {minutes} хвилин.")
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка: {e}")

# --- Команда /stop
async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in auto_tasks:
        auto_tasks[chat_id].schedule_removal()
        del auto_tasks[chat_id]
        await update.message.reply_text("⛔️ Авто-прогноз зупинено.")
    else:
        await update.message.reply_text("❗️ Авто-прогноз не запущено.")

# --- Flask сервер
flask_app = Flask(__name__)

@flask_app.route('/')
def index():
    return "✅ Бот працює!"

# --- Telegram запуск
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

async def run_bot():
    logging.info("🚀 Запуск Telegram бота")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("auto", auto))
    app.add_handler(CommandHandler("stop", stop))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await asyncio.Event().wait()

# --- Основний запуск
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=port), daemon=True).start()
    asyncio.run(run_bot())
