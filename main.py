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

# --- Прибрати зайві логи TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)

# === Алгоритм з фото: A = ((x^n + y) + a) + ((x^{n-1} + y) + a) + ... + ((x^1 + y) + a)
def custom_algorithm(x: float, y: float, a: float, n: int) -> float:
    return sum((x**i + y + a) for i in range(n, 0, -1))

# === API запит до CoinMarketCap
def fetch_latest_data():
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": os.environ.get("COINMARKETCAP_API_KEY")
    }
    params = {
        "symbol": "BTC",
        "convert": "USD"
    }
    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    try:
        price = data["data"]["BTC"]["quote"]["USD"]["price"]
        timestamp = pd.Timestamp.now()
        df = pd.DataFrame([[timestamp, price]], columns=["timestamp", "price"])
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        raise RuntimeError("❌ Помилка обробки відповіді від CoinMarketCap") from e

# === Побудова графіка (поки просто один запис, як заглушка)
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

# === Telegram Команди ===
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        df = fetch_latest_data()
        now_price = df["price"].iloc[-1]
        predicted_price = now_price * 1.05
        change = predicted_price - now_price
        change_pct = (change / now_price) * 100
        plot_buf = plot_latest_data(df)

        text = (
            f"📊 Поточна ціна: ${now_price:.2f}\n"
            f"🔮 Прогноз: ${predicted_price:.2f}\n"
            f"📈 Зміна: ${change:.2f} ({change_pct:.2f}%)"
        )
        await update.message.reply_text(text)
        await update.message.reply_photo(photo=plot_buf)
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
        "/predict — отримати прогноз\n"
        "/custom x y a n — власна формула\n"
        "/auto [хв] — авто-прогноз\n"
        "/stop — зупинити авто-прогноз"
    )

# === Автопрогнозування ===
auto_tasks = {}

async def auto_predict(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id
    df = fetch_latest_data()
    now_price = df["price"].iloc[-1]
    predicted_price = now_price * 1.05
    change = predicted_price - now_price
    change_pct = (change / now_price) * 100
    plot_buf = plot_latest_data(df)

    text = (
        f"📊 Поточна ціна: ${now_price:.2f}\n"
        f"🔮 Прогноз: ${predicted_price:.2f}\n"
        f"📈 Зміна: ${change:.2f} ({change_pct:.2f}%)"
    )
    await context.bot.send_message(chat_id=chat_id, text=text)
    await context.bot.send_photo(chat_id=chat_id, photo=plot_buf)

async def auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) != 1:
            await update.message.reply_text("❗️ Формат: /auto 5 (хвилин)")
            return

        minutes = int(context.args[0])
        chat_id = update.effective_chat.id

        if chat_id in auto_tasks:
            auto_tasks[chat_id].schedule_removal()

        job = context.job_queue.run_repeating(auto_predict, interval=minutes*60, first=0, chat_id=chat_id)
        auto_tasks[chat_id] = job
        await update.message.reply_text(f"✅ Авто-прогноз кожні {minutes} хвилин.")
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка: {e}")

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in auto_tasks:
        auto_tasks[chat_id].schedule_removal()
        del auto_tasks[chat_id]
        await update.message.reply_text("⛔️ Авто-прогноз зупинено.")
    else:
        await update.message.reply_text("❗️ Авто-прогноз не запущено.")

# === Flask для Render keep-alive ===
flask_app = Flask(__name__)
@flask_app.route('/')
def index():
    return "✅ Бот працює!"

# === Telegram Bot запуск ===
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

async def run_bot():
    logging.info("🚀 Запуск Telegram бота")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("custom", custom))
    app.add_handler(CommandHandler("auto", auto))
    app.add_handler(CommandHandler("stop", stop))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()

# === Головний запуск: Flask + Telegram ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=port), daemon=True).start()

    loop = asyncio.get_event_loop()
    loop.create_task(run_bot())
    loop.run_forever()
