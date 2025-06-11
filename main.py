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
from prophet import Prophet

# === Налаштування логування та середовища
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)

# === Завантаження історичних даних BTC з Binance
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

# === Побудова графіка прогнозу
def plot_forecast(model, forecast):
    fig = model.plot(forecast)
    plt.title("Bitcoin (прогноз)")
    plt.xlabel("Дата")
    plt.ylabel("Ціна (USD)")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# === Команда /predict
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        df = fetch_historical_data()
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=1)
        forecast = model.predict(future)

        predicted_price = forecast.iloc[-1]["yhat"]
        now_price = df["y"].iloc[-1]
        change = predicted_price - now_price
        change_pct = (change / now_price) * 100

        plot_buf = plot_forecast(model, forecast)

        text = (
            f"\U0001F4CA Поточна ціна: ${now_price:.2f}\n"
            f"\U0001F52E Прогноз: ${predicted_price:.2f}\n"
            f"\U0001F4C8 Зміна: ${change:.2f} ({change_pct:.2f}%)"
        )
        await update.message.reply_text(text)
        await update.message.reply_photo(photo=plot_buf)
    except Exception as e:
        logging.exception("❗️Помилка прогнозу")
        await update.message.reply_text(f"❌ Помилка: {e}")

# === Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Я трейдинг-прогнозатор бот.\n"
        "Команди:\n"
        "/predict — отримати прогноз\n"
        "/auto [хв] — авто-прогноз\n"
        "/stop — зупинити авто-прогноз"
    )

# === Авто-прогноз
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
        change = predicted_price - now_price
        change_pct = (change / now_price) * 100

        plot_buf = plot_forecast(model, forecast)

        text = (
            f"\U0001F4CA Поточна ціна: ${now_price:.2f}\n"
            f"\U0001F52E Прогноз: ${predicted_price:.2f}\n"
            f"\U0001F4C8 Зміна: ${change:.2f} ({change_pct:.2f}%)"
        )
        await context.bot.send_message(chat_id=chat_id, text=text)
        await context.bot.send_photo(chat_id=chat_id, photo=plot_buf)
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Помилка: {e}")

# === Команда /auto
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

# === Команда /stop
async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in auto_tasks:
        auto_tasks[chat_id].schedule_removal()
        del auto_tasks[chat_id]
        await update.message.reply_text("⛔️ Авто-прогноз зупинено.")
    else:
        await update.message.reply_text("❗️ Авто-прогноз не запущено.")

# === Flask-сервер для Render
flask_app = Flask(__name__)

@flask_app.route('/')
def index():
    return "✅ Бот працює!"

# === Запуск Telegram бота
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

# === Головний запуск
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=port), daemon=True).start()
    asyncio.run(run_bot())