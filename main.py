import os
import logging
import asyncio
import requests
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, request
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from prophet import Prophet

# --- Налаштування
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(level=logging.INFO)

# --- Telegram token та webhook URL
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")  # наприклад: https://yourapp.onrender.com

bot = Bot(token=TELEGRAM_TOKEN)

# --- Flask додаток
flask_app = Flask(__name__)

# --- Завантаження історичних даних з Binance
def fetch_historical_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1d", "limit": 200}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    raw_data = resp.json()
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

# --- Команди бота
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Я крипто-прогнозатор 📈\n"
        "Використай /predict model=prophet/randomforest/svr days=1..7\n"
        "Приклад: /predict model=svr days=3"
    )

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

            if model_type == "randomforest":
                model = RandomForestRegressor(n_estimators=200, random_state=42)
            else:
                model = SVR(kernel='rbf')

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

# --- Flask route для webhook
@flask_app.route(f"/webhook/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    data = request.get_json(force=True)
    update = Update.de_json(data, bot)
    asyncio.create_task(application.update_queue.put(update))
    return "ok", 200

# --- Flask route для тесту
@flask_app.route('/')
def index():
    return "✅ Бот працює!"

# --- Telegram Application
application = Application.builder().token(TELEGRAM_TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("predict", predict))

# --- Запуск webhook на Render
if __name__ == '__main__':
    # Встановлюємо webhook
    async def set_webhook():
        await bot.set_webhook(f"{WEBHOOK_URL}/webhook/{TELEGRAM_TOKEN}")
        logging.info("✅ Webhook встановлено")

    asyncio.run(set_webhook())
    port = int(os.environ.get("PORT", 4000))
    flask_app.run(host="0.0.0.0", port=port)
