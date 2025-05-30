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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)

# === Формула

def custom_algorithm(x: float, y: float, a: float, n: int) -> float:
    return sum((x**i + y + a) for i in range(n, 0, -1))

# === Синтетичний сигнал на основі алгоритму

def generate_signal(length=128, x=0.5, y=1.0, a=2.0, n=10):
    return np.array([custom_algorithm(x + i * 0.01, y, a, n) for i in range(length)])

# === Перетворення Фур'є

def apply_fft(signal):
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal))
    return frequencies, np.abs(fft_result)

# === Побудова графіка Фур'є

def plot_fft(frequencies, magnitude):
    fig, ax = plt.subplots()
    ax.plot(frequencies[:len(frequencies) // 2], magnitude[:len(magnitude) // 2])
    ax.set_title("Частотний спектр (FFT)")
    ax.set_xlabel("Частота")
    ax.set_ylabel("Амплітуда")
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# === Ціна BTC

def fetch_latest_data():
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    api_key = os.environ.get("COINMARKETCAP_API_KEY")
    if not api_key:
        raise RuntimeError("❌ Не вказано COINMARKETCAP_API_KEY у змінних середовища!")

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": api_key
    }
    params = {"symbol": "BTC", "convert": "USD"}

    response = requests.get(url, headers=headers, params=params)
    try:
        response.raise_for_status()
        data = response.json()
        price = data["data"]["BTC"]["quote"]["USD"]["price"]
        timestamp = pd.Timestamp.now()
        df = pd.DataFrame([[timestamp, price]], columns=["timestamp", "price"])
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        raise RuntimeError(f"❌ API помилка: {e}")

# === Побудова графіка

def plot_latest_data(df):
    fig, ax = plt.subplots()
    df.plot(ax=ax, legend=False)
    plt.title("Bitcoin (поточна ціна)")
    plt.xlabel("Дата")
    plt.ylabel("Ціна (USD)")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# === Telegram-команди

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        signal = generate_signal()
        frequencies, magnitude = apply_fft(signal)
        plot_buf = plot_fft(frequencies, magnitude)
        await update.message.reply_photo(photo=plot_buf, caption="🔮 Прогноз на основі твоєї формули + FFT")
    except Exception as e:
        logging.exception("❗️Помилка прогнозу")
        await update.message.reply_text(f"❌ Помилка: {e}")

async def custom(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) != 4:
            await update.message.reply_text("❗️ Введи 4 аргументи: /custom x y a n")
            return
        x, y, a = float(context.args[0]), float(context.args[1]), float(context.args[2])
        n = int(context.args[3])
        result = custom_algorithm(x, y, a, n)
        await update.message.reply_text(f"🔢 Результат A = {result:.4f}")
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Я трейдинг-прогнозатор бот.\n"
        "Команди:\n"
        "/predict — прогноз через твою формулу + FFT\n"
        "/custom x y a n — обрахунок власної формули\n"
    )

# === Flask-сервер

flask_app = Flask(__name__)

@flask_app.route('/')
def index():
    return "✅ Бот працює!"

# === Telegram бот

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

async def run_bot():
    logging.info("🚀 Запуск Telegram бота")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("custom", custom))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await asyncio.Event().wait()

# === Головний запуск

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=port), daemon=True).start()
    asyncio.run(run_bot())
