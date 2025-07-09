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
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from prophet import Prophet

# --- Налаштування
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)

# --- Завантаження історичних даних BTC
def fetch_historical_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 100}
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

# --- Графік
def plot_forecast(df, future_dates, predictions, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(df["ds"], df["y"], label="Історія")
    plt.plot(future_dates, predictions, linestyle='--', marker='o', label="Прогноз")
    plt.xlabel("Дата")
    plt.ylabel("Ціна (USD)")
    plt.title(f"Bitcoin ({model_name} прогноз)")
    plt.legend()
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# --- Фічі
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
def apply_human_factor(price, mood):
    factors = {
        "neutral": 0.0,
        "fear": -0.03,
        "greed": 0.04,
        "panic": -0.07,
        "euphoria": 0.08
    }
    return price * (1 + factors.get(mood.lower(), 0.0))

# --- Прогноз
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        model_type = "prophet"
        mood = "neutral"
        days = 1

        if context.args:
            for arg in context.args:
                if arg.startswith("model="):
                    model_type = arg.split("=")[1].lower()
                elif arg.startswith("mood="):
                    mood = arg.split("=")[1].lower()
                elif arg.startswith("days="):
                    days = int(arg.split("=")[1])

        df = fetch_historical_data()
        now_price = df["y"].iloc[-1]
        predicted_values = []
        future_dates = []

        # --- Prophet
        if model_type == "prophet":
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            predicted_values = forecast.iloc[-days:]["yhat"].values
            future_dates = forecast.iloc[-days:]["ds"].values
            label = "Prophet"

        # --- RandomForest
        elif model_type == "randomforest":
            X, y, features = prepare_features(df)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_scaled, y)

            last_row = df.iloc[-1:].copy()
            for i in range(days):
                new_row = {
                    "day": last_row["ds"].dt.day.values[0],
                    "month": last_row["ds"].dt.month.values[0],
                    "year": last_row["ds"].dt.year.values[0],
                    "dayofweek": last_row["ds"].dt.dayofweek.values[0],
                    "lag1": last_row["y"].values[0],
                    "lag2": df.iloc[-2]["y"]
                }
                X_pred = pd.DataFrame([new_row])[features]
                X_pred_scaled = scaler.transform(X_pred)
                pred = model.predict(X_pred_scaled)[0]
                predicted_values.append(pred)
                next_date = last_row["ds"].values[0] + np.timedelta64(1, 'D')
                future_dates.append(next_date)
                last_row = pd.DataFrame({"ds": [next_date], "y": [pred]})
            label = "RandomForest"

        # --- SVR
        elif model_type == "svr":
            X, y, features = prepare_features(df)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = SVR(kernel='rbf')
            model.fit(X_scaled, y)

            last_row = df.iloc[-1:].copy()
            for i in range(days):
                new_row = {
                    "day": last_row["ds"].dt.day.values[0],
                    "month": last_row["ds"].dt.month.values[0],
                    "year": last_row["ds"].dt.year.values[0],
                    "dayofweek": last_row["ds"].dt.dayofweek.values[0],
                    "lag1": last_row["y"].values[0],
                    "lag2": df.iloc[-2]["y"]
                }
                X_pred = pd.DataFrame([new_row])[features]
                X_pred_scaled = scaler.transform(X_pred)
                pred = model.predict(X_pred_scaled)[0]
                predicted_values.append(pred)
                next_date = last_row["ds"].values[0] + np.timedelta64(1, 'D')
                future_dates.append(next_date)
                last_row = pd.DataFrame({"ds": [next_date], "y": [pred]})
            label = "SVR"

        else:
            await update.message.reply_text("❗️Невідома модель. Використай model=prophet, randomforest або svr")
            return

        # --- Застосовуємо настрій
        final_price = apply_human_factor(predicted_values[-1], mood)
        change = final_price - now_price
        percent = (change / now_price) * 100

        # --- Формуємо графік і відповідь
        plot_buf = plot_forecast(df, future_dates, predicted_values, label)
        text = (
            f"📊 Поточна ціна: ${now_price:.2f}\n"
            f"🔮 Прогноз ({label}) на {days} дн.: ${final_price:.2f}\n"
            f"📈 Зміна: ${change:.2f} ({percent:.2f}%)\n"
            f"🧠 Настрій: {mood}"
        )

        await update.message.reply_photo(photo=plot_buf, caption=text)

    except Exception as e:
        logging.exception("❗️Помилка прогнозу")
        await update.message.reply_text(f"❌ Помилка: {e}")

# --- /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Я крипто-прогнозатор 📈\n"
        "Команда: /predict model=prophet/randomforest/svr days=1..7 mood=greed/fear/euphoria\n"
        "Приклад: /predict model=svr days=3 mood=greed"
    )

# --- Flask keep-alive
flask_app = Flask(__name__)
@flask_app.route('/')
def index():
    return "✅ Бот працює!"

# --- Telegram запуск
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
async def run_bot():
    logging.info("🚀 Бот запускається...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    await asyncio.Event().wait()

# --- Запуск
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    threading.Thread(target=lambda: flask_app.run(host="0.0.0.0", port=port), daemon=True).start()
    asyncio.run(run_bot())
