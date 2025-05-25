import os
import threading
import logging
import asyncio
from flask import Flask
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

# Для Render: прибрати зайві TensorFlow логування
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# === Алгоритм ===
def custom_algorithm(x: float, y: float, a: float, n: int) -> float:
    result = 0
    for i in range(n, 0, -1):
        result += (x**i + y) + a
    return result

# === Прогнозування (простий приклад для демонстрації) ===
def plot_latest_data(df):
    fig, ax = plt.subplots()
    df.tail(30).plot(ax=ax)
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        now_price = 123.45  # тестове значення
        predicted_price = 130.00
        change = predicted_price - now_price
        change_pct = (change / now_price) * 100
        df = pd.DataFrame({'Price': np.random.rand(100)})

        plot_buf = plot_latest_data(df)
        text = (
            f"Поточна ціна: ${now_price:.2f}\n"
            f"Прогноз: ${predicted_price:.2f}\n"
            f"Зміна: ${change:.2f} ({change_pct:.2f}%)"
        )
        await update.message.reply_text(text)
        await update.message.reply_photo(photo=plot_buf)
    except Exception as e:
        logging.exception("Prediction error")
        await update.message.reply_text(f"Помилка: {e}")

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
        await update.message.reply_text(f"Помилка: {e}")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привіт! Надішли /custom x y a n для обчислення або /predict для прогнозу.")

# === Flask для Render Keep-Alive ===
flask_app = Flask(__name__)
@flask_app.route('/')
def index():
    return "Бот працює!"

def keep_alive():
    threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=10000)).start()

# === Telegram Bot Start ===
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

async def run_bot():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("custom", custom))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await app.updater.idle()

def start_all():
    keep_alive()
    threading.Thread(target=lambda: asyncio.run(run_bot()), daemon=True).start()

if name == '__main__':
    start_all()
