import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import requests
import telebot
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

TOKEN = os.environ.get('BOT_TOKEN')
bot = telebot.TeleBot(TOKEN)

# Завантаження даних
def load_crypto_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30"
    response = requests.get(url)
    data = response.json()
    try:
        prices = np.array([item[1] for item in data['prices']])
        timestamps = [datetime.fromtimestamp(item[0] / 1000.0) for item in data['prices']]
        return prices.reshape(-1, 1), timestamps
    except KeyError as e:
        raise ValueError(f"Неможливо обробити дані: {e}")

# Підготовка даних для LSTM
def prepare_data(data, seq_length=10):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# Побудова LSTM моделі
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Прогноз та побудова графіка
def get_prediction_text_and_plot(user_id):
    prices, timestamps = load_crypto_data()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    X, y = prepare_data(scaled_prices)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    model = create_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    # Прогноз
    last_sequence = scaled_prices[-10:]
    last_sequence = np.reshape(last_sequence, (1, 10, 1))
    prediction = model.predict(last_sequence)[0][0]
    predicted_price = scaler.inverse_transform([[prediction]])[0][0]

    # Побудова графіка
    predicted_prices = model.predict(X)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    real_prices = scaler.inverse_transform(y.reshape(-1, 1))

    plot_buf = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps[-len(real_prices):], real_prices, label='Real')
    plt.plot(timestamps[-len(predicted_prices):], predicted_prices, label='Predicted')
    plt.title('Crypto Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_buf, format='png')
    plt.close()
    plot_buf.seek(0)

    text = (
        f"Модель: LSTM\n"
        f"Останнє передбачене значення: {predicted_price:.2f} USD\n"
        f"Дата: {timestamps[-1].strftime('%Y-%m-%d %H:%M:%S')}"
    )
    return text, plot_buf

# Обробка команди /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Привіт! Я бот для прогнозування ціни BTC. Введи /predict, щоб отримати прогноз.")

# Обробка команди /predict
@bot.message_handler(commands=['predict'])
def predict(message):
    try:
        start_time = time.time()
        text, plot = get_prediction_text_and_plot(message.chat.id)
        end_time = time.time()
        duration = end_time - start_time

        text += f"\nЧас прогнозування: {duration:.2f} сек."
        bot.send_message(message.chat.id, text)
        bot.send_photo(message.chat.id, photo=plot)
    except Exception as e:
        bot.send_message(message.chat.id, f"Помилка: {e}")

# Запуск
if __name__ == '__main__':
    bot.polling(none_stop=True)
