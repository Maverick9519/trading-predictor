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

import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# === Налаштування
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)

# === Формула
def custom_algorithm(x: float, y: float, a: float, n: int) -> float:
    return sum((x**i + y + a) for i in range(n, 0, -1))

# === Поточна ціна BTC
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
    response.raise_for_status()
    data = response.json()
    price = data["data"]["BTC"]["quote"]["USD"]["price"]
    timestamp = pd.Timestamp.now()
    df = pd.DataFrame([[timestamp, price]], columns=["timestamp", "price"])
    df.set_index("timestamp", inplace=True)
    return df

# === Побудова графіка ціни BTC
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

# === Linear Regression
def linear_predict_price():
    df = yf.download("BTC-USD", period="2d", interval="1h")
    df.dropna(inplace=True)

    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values

    model = LinearRegression()
    model.fit(X, y)

    next_index = np.array([[len(df)]])
    next_price = model.predict(next_index)[0]

    fig, ax = plt.subplots()
    ax.plot(df.index, y, label="Історія")
    ax.plot(df.index[-1] + pd.Timedelta(hours=1), next_price, 'go', label="Прогноз (Linear)")
    plt.title("Прогноз BTC (Linear Regression)")
    plt.xlabel("Час")
    plt.ylabel("Ціна (USD)")
    plt.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return y[-1], next_price, buf

# === Random Forest
def rf_predict_price(return_current_price=False):
    df = yf.download("BTC-USD", period="2d", interval="1h")
    df.dropna(inplace=True)

    X = np.arange(len(df)).reshape(-1, 1)
    y = df["Close"].values

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    next_index = np.array([[len(df)]])
    next_price = model.predict(next_index)[0]

    fig, ax = plt.subplots()
    ax.plot(df.index, y, label="Історія")
    ax.plot(df.index[-1] + pd.Timedelta(hours=1), next_price, 'ro', label="Прогноз (RF)")
    plt.title("Прогноз BTC (Random Forest)")
    plt.xlabel("Час")
    plt.ylabel("Ціна (USD)")
    plt.legend()
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)

    if return_current_price:
        return y[-1], next_price, buf
    return next_price, buf

# === Telegram-команди
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Я трейдинг-прогнозатор бот.\n"
        "Команди:\n"
        "/predict [linear|rf] — прогноз ціни BTC\n"
        "/custom x y a n — власна формула\n"
        "/custom_predict y a n — формула з ціною BTC\n"
        "/auto [хв] — авто-прогноз\n"
        "/stop — зупинити авто-прогноз"
    )

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        model_type = context.args[0].lower() if context.args else "linear"

        if model_type == "rf":
            now_price, predicted_price, plot_buf = rf_predict_price(return_current_price=True)
            model_name = "Random Forest"
        else:
            now_price, predicted_price, plot_buf = linear_predict_price()
            model_name = "Linear Regression"

        change = predicted_price - now_price
        change_pct = (change / now_price) * 100

        text = (
            f"📊 Поточна ціна: ${now_price:.2f}\n"
            f"🔮 Прогноз ({model_name}): ${predicted_price:.2f}\n"
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
            await update.message.reply_text("❗️ Формат: /custom x y a n")
            return
        x, y, a = float(context.args[0]), float(context.args[1]), float(context.args[2])
        n = int(context.args[3])
        result = custom_algorithm(x, y, a, n)
        await update.message.reply_text(f"🔢 Результат A = {result:.4f}")
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка: {e}")

async def custom_predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        if len(context.args) != 3:
            await update.message.reply_text("❗️ Формат: /custom_predict y a n")
            return
        df = fetch_latest_data()
        x = df["price"].iloc[-1]
        y = float(context.args[0])
        a = float(context.args[1])
        n = int(context.args[2])
        result = custom_algorithm(x, y, a, n)
        text = (
            f"📊 Поточна ціна BTC (x) = {x:.2f}\n"
            f"🔧 y = {y}, a = {a}, n = {n}\n"
            f"🧮 Результат A = {result:.4f}"
        )
        await update.message.reply_text(text)
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка: {e}")

# === Авто-прогноз
auto_tasks = {}

async def auto_predict(context: ContextTypes.DEFAULT_TYPE):
    chat_id = context.job.chat_id
    try:
        now_price, predicted_price, plot_buf = linear_predict_price()
        change = predicted_price - now_price
        change_pct = (change / now_price) * 100
        text = (
            f"📊 Поточна ціна: ${now_price:.2f}\n"
            f"🔮 Прогноз: ${predicted_price:.2f}\n"
            f"📈 Зміна: ${change:.2f} ({change_pct:.2f}%)"
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

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in auto_tasks:
        auto_tasks[chat_id].schedule_removal()
        del auto_tasks[chat_id]
        await update.message.reply_text("⛔️ Авто-прогноз зупинено.")
    else:
        await update.message.reply_text("❗️ Авто-прогноз не запущено.")

# === Flask
flask_app = Flask(__name__)
@flask_app.route('/')
def index():
    return "✅ Бот працює!"

# === Telegram запуск
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

async def run_bot():
    logging.info("🚀 Запуск Telegram бота")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("custom", custom))
    app.add_handler(CommandHandler("custom_predict", custom_predict))
    app.add_handler(CommandHandler("auto", auto))
    app.add_handler(CommandHandler("stop", stop))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await asyncio.Event().wait()

# === Запуск
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=port), daemon=True).start()
    asyncio.run(run_bot())