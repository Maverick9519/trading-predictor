import os
import threading
import logging
import asyncio
import requests
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
from datetime import datetime, timedelta

# --- Налаштування
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
COINMARKETCAP_API_KEY = os.environ.get("COINMARKETCAP_API_KEY")
if not TELEGRAM_TOKEN or not COINMARKETCAP_API_KEY:
    raise RuntimeError("❌ TELEGRAM_TOKEN або COINMARKETCAP_API_KEY не задані!")

# --- Завантаження історичних даних з CoinMarketCap
def fetch_historical_data():
    end = datetime.utcnow()
    start = end - timedelta(days=200)

    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
    params = {
        "symbol": "BTC",
        "convert": "USD",
        "time_start": int(start.timestamp()),
        "time_end": int(end.timestamp()),
    }
    headers = {"X-CMC_PRO_API_KEY": COINMARKETCAP_API_KEY}

    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()

    quotes = data["data"]["quotes"]
    df = pd.DataFrame([
        {"ds": datetime.fromtimestamp(q["time_open"]), "y": q["quote"]["USD"]["close"]}
        for q in quotes
    ])
    return df

# --- Побудова графіку
def plot_forecast(df, future_dates, predictions, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(df["ds"], df["y"], label="Історія", color="#2563eb", linewidth=2)
    plt.plot(future_dates, predictions, label="Прогноз", color="#22c55e", linewidth=2, linestyle="--")
    plt.xlabel("Дата")
    plt.ylabel("Ціна (USD)")
    plt.title(f"Bitcoin ({model_name} прогноз)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# --- Побудова фіч
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

# --- Команда /predict
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        model_type = "prophet"
        days = 3

        if context.args:
            for arg in context.args:
                if arg.startswith("model="):
                    model_type = arg.split("=")[1].lower()
                elif arg.startswith("days="):
                    days = int(arg.split("=")[1])

        df = fetch_historical_data()
        now_price = df["y"].iloc[-1]
        predicted_values = []
        future_dates = []

        if model_type == "prophet":
            model = Prophet()
            model.fit(df)
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            predicted_values = forecast.iloc[-days:]["yhat"].values
            future_dates = forecast.iloc[-days:]["ds"].values
            model_label = "Prophet"

        elif model_type in ["randomforest", "svr"]:
            X, y, features = prepare_features(df)

            scaler = StandardScaler()
            X_scaled = X.copy()
            X_scaled[["day", "month", "year", "dayofweek"]] = scaler.fit_transform(
                X_scaled[["day", "month", "year", "dayofweek"]]
            )

            model = RandomForestRegressor(n_estimators=200, random_state=42) if model_type == "randomforest" else SVR(kernel='rbf')
            model.fit(X_scaled, y)

            last_row = df.iloc[-1:].copy()
            prev2 = df.iloc[-2]["y"]

            for _ in range(days):
                features_input = {
                    "day": last_row["ds"].dt.day.values[0],
                    "month": last_row["ds"].dt.month.values[0],
                    "year": last_row["ds"].dt.year.values[0],
                    "dayofweek": last_row["ds"].dt.dayofweek.values[0],
                    "lag1": last_row["y"].values[0],
                    "lag2": prev2,
                }

                X_pred = pd.DataFrame([features_input])[features]
                X_pred_scaled = X_pred.copy()
                X_pred_scaled[["day", "month", "year", "dayofweek"]] = scaler.transform(
                    X_pred_scaled[["day", "month", "year", "dayofweek"]]
                )

                pred = model.predict(X_pred_scaled)[0]
                predicted_values.append(pred)
                next_date = last_row["ds"].values[0] + pd.Timedelta(days=1)
                future_dates.append(next_date)

                prev2 = last_row["y"].values[0]
                last_row = pd.DataFrame({"ds": [next_date], "y": [pred]})

            model_label = "RandomForest" if model_type == "randomforest" else "SVR"

        else:
            await update.message.reply_text("❗️Модель не розпізнано. Використай model=prophet, svr або randomforest.")
            return

        plot_buf = plot_forecast(df, future_dates, predicted_values, model_label)

        text = f"📊 Поточна ціна: ${now_price:.2f}\n🔮 Прогноз ({model_label}): ${predicted_values[-1]:.2f}"
        await update.message.reply_photo(photo=plot_buf, caption=text)

    except Exception as e:
        logging.exception("❌ Помилка прогнозу")
        await update.message.reply_text(f"❌ Помилка: {e}")

# --- Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Я крипто-прогнозатор 📈\n"
        "Використай /predict model=prophet/randomforest/svr days=1..7\n"
        "Приклад: /predict model=svr days=3"
    )

# --- Flask keep-alive
flask_app = Flask(__name__)
@flask_app.route('/')
def index():
    return "✅ Бот працює!"

# --- Telegram запуск через Application
async def run_bot():
    logging.info("🚀 Бот запускається...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    await app.initialize()
    await app.start()
    await asyncio.Event().wait()  # щоб не закривався

# --- Запуск
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    threading.Thread(target=lambda: flask_app.run(host="0.0.0.0", port=port), daemon=True).start()
    asyncio.run(run_bot())
