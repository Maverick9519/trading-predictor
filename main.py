import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from binance.client import Client
import logging
import io
import time
import threading
import requests
from flask import Flask
import asyncio
import os

# === Telegram Token ===
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

# === Binance API client ===
BINANCE_CLIENT = Client()

# === Logging ===
logging.basicConfig(level=logging.INFO)

# === Flask Server ===
flask_app = Flask(__name__)

@flask_app.route("/")
def home():
    return "✅ Бот працює", 200

def keep_alive():
    def ping():
        while True:
            try:
                requests.get("https://trading-predictor.onrender.com")
                print("🟢 Self-ping successful")
            except Exception as e:
                print("🔴 Self-ping failed:", e)
            time.sleep(300)
    threading.Thread(target=ping, daemon=True).start()

# === Load data ===
def load_crypto_data():
    klines = BINANCE_CLIENT.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1HOUR, limit=500)
    data = pd.DataFrame(klines, columns=[
        'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close time', 'Quote asset volume', 'Number of trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])
    data['Date'] = pd.to_datetime(data['Close time'], unit='ms')
    data['Price'] = data['Close'].astype(float)
    return data[['Date', 'Price']]

def create_dataset(series, look_back=10):
    X, y = [], []
    for i in range(len(series) - look_back):
        X.append(series[i:(i + look_back)])
        y.append(series[i + look_back])
    return np.array(X), np.array(y)

def predict_lstm(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['Price']].values)
    look_back = 10
    X, y = create_dataset(scaled, look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

    last_sequence = scaled[-look_back:].reshape((1, look_back, 1))
    prediction = model.predict(last_sequence)[0][0]
    predicted_price = scaler.inverse_transform([[prediction]])[0][0]
    return predicted_price

def plot_latest_data(df):
    plt.figure(figsize=(10, 4))
    plt.plot(df['Date'], df['Price'], label='Real Price')
    plt.title('Bitcoin Price - Last 500 Hours')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.xticks(rotation=45)
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# === Telegram Bot Handlers ===
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привіт! Я бот прогнозування ціни BTC на основі LSTM з Binance.\n/predict — отримати прогноз.")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Отримання даних з Binance та прогнозування...")
    try:
        df = load_crypto_data()
        predicted_price = predict_lstm(df)
        now_price = df['Price'].iloc[-1]
        change = predicted_price - now_price
        change_pct = (change / now_price) * 100

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

async def run_bot():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await app.updater.idle()

def start_all():
    keep_alive()
    threading.Thread(target=lambda: asyncio.run(run_bot()), daemon=True).start()
    flask_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))

if __name__ == '__main__':
    start_all()
