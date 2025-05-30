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

def custom_algorithm(x: float, y: float, a: float, n: int) -> np.ndarray:
    return np.array([x ** k + y + a for k in range(1, n + 1)])

# === Ціна BTC з історією

def fetch_historical_data(n=64):
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    api_key = os.environ.get("COINMARKETCAP_API_KEY")
    if not api_key:
        raise RuntimeError("❌ Не вказано COINMARKETCAP_API_KEY у змінних середовища!")

    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": api_key
    }
    params = {"symbol": "BTC", "convert": "USD"}

    prices = []
    for _ in range(n):
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        price = data["data"]["BTC"]["quote"]["USD"]["price"]
        prices.append(price)
        asyncio.sleep(0.5)  # Пауза для обмеження API

    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='H')
    df = pd.DataFrame({"timestamp": timestamps, "price": prices})
    df.set_index("timestamp", inplace=True)
    return df

# === Побудова графіка цін

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

# === FFT-обробка

def fft_filtered_signal(signal: np.ndarray, keep: int = 5) -> np.ndarray:
    fft_vals = np.fft.fft(signal)
    fft_filtered = fft_vals.copy()
    fft_filtered[keep:-keep] = 0
    return np.fft.ifft(fft_filtered).real

# === Telegram-команди

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        x, y, a, n = 1.1, 2.0, 1.0, 64
        signal = custom_algorithm(x, y, a, n)
        signal_fft = fft_filtered_signal(signal)

        fig, ax = plt.subplots()
        ax.plot(signal, label='Оригінал')
        ax.plot(signal_fft, label='Фур\'є прогноз', linestyle='--')
        ax.set_title("Custom сигнал + FFT прогноз")
        ax.set_xlabel("Час (k)")
        ax.set_ylabel("Значення")
        ax.grid(True)
        ax.legend()

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        await update.message.reply_text("🔍 Сигнал та його прогноз після Фур'є")
        await update.message.reply_photo(photo=buf)

    except Exception as e:
        logging.exception("❗️Помилка прогнозу")
        await update.message.reply_text(f"❌ Помилка: {e}")

async def fftbtc(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        df = fetch_historical_data(n=64)
        prices = df['price'].values
        prices_fft = fft_filtered_signal(prices)

        fig, ax = plt.subplots()
        ax.plot(df.index, prices, label='Оригінальні дані')
        ax.plot(df.index, prices_fft, label='Фур\'є прогноз', linestyle='--')
        ax.set_title("Bitcoin FFT Прогноз")
        ax.set_xlabel("Дата")
        ax.set_ylabel("Ціна (USD)")
        ax.grid(True)
        ax.legend()

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        await update.message.reply_text("📉 Ціни BTC та прогноз після Фур'є")
        await update.message.reply_photo(photo=buf)

    except Exception as e:
        logging.exception("❗️Помилка FFT BTC")
        await update.message.reply_text(f"❌ Помилка: {e}")

async def custom(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) != 4:
            await update.message.reply_text("❗️ Введи 4 аргументи: /custom x y a n")
            return
        x, y, a = float(context.args[0]), float(context.args[1]), float(context.args[2])
        n = int(context.args[3])
        result = custom_algorithm(x, y, a, n).sum()
        await update.message.reply_text(f"🔢 Результат A = {result:.4f}")
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Я трейдинг-прогнозатор бот.\n"
        "Команди:\n"
        "/predict — FFT на custom формулі\n"
        "/fftbtc — FFT прогноз на реальних даних BTC\n"
        "/custom x y a n — власна формула"
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
    app.add_handler(CommandHandler("fftbtc", fftbtc))
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
