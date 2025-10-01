# main.py
import os
import logging
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # щоб matplotlib працював на сервері без дисплея
import matplotlib.pyplot as plt
from io import BytesIO
from flask import Flask, request, abort
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, ContextTypes
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import asyncio

# --- Логи
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Environment variables
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")
CMC_KEY = os.environ.get("COINMARKETCAP_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN не знайдено в Environment variables.")
if not WEBHOOK_URL:
    raise RuntimeError("WEBHOOK_URL не знайдено в Environment variables.")
if not CMC_KEY:
    raise RuntimeError("COINMARKETCAP_API_KEY не знайдено в Environment variables.")

# --- Ініціалізація бота і Flask
bot = Bot(token=TELEGRAM_TOKEN)
app = Flask(__name__)

# --- Отримання історичних даних (CoinMarketCap)
def fetch_historical_data():
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"
    params = {
        "symbol": "BTC",
        "convert": "USD",
        "interval": "daily",
        "count": 200
    }
    headers = {"X-CMC_PRO_API_KEY": CMC_KEY}
    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if "data" not in data or "quotes" not in data["data"]:
        raise ValueError("Невірна відповідь від CoinMarketCap API")
    raw = data["data"]["quotes"]
    df = pd.DataFrame([{
        "ds": pd.to_datetime(item["timestamp"]),
        "y": float(item["quote"]["USD"]["close"])
    } for item in raw])
    return df

# --- Малювання графіку
def plot_forecast(df, future_dates, predictions, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(df["ds"], df["y"], label="Історія", linewidth=2, color="#2563eb")
    plt.plot(future_dates, predictions, label="Прогноз", linewidth=2, linestyle="--", color="#22c55e")
    plt.xlabel("Дата")
    plt.ylabel("Ціна (USD)")
    plt.title(f"Bitcoin ({model_name} прогноз)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

# --- Фічі для ML моделей
def prepare_features(df):
    df = df.copy()
    df["day"] = df["ds"].dt.day
    df["month"] = df["ds"].dt.month
    df["year"] = df["ds"].dt.year
    df["dayofweek"] = df["ds"].dt.dayofweek
    df["lag1"] = df["y"].shift(1)
    df["lag2"] = df["y"].shift(2)
    df = df.dropna()
    features = ["day", "month", "year", "dayofweek", "lag1", "lag2"]
    return df[features], df["y"], features

# --- Команди Telegram
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
                    model_type = arg.split("=", 1)[1].lower()
                elif arg.startswith("days="):
                    days = int(arg.split("=", 1)[1])

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

        elif model_type in ("randomforest", "svr"):
            X, y, features = prepare_features(df)
            scaler = StandardScaler()
            X_scaled = X.copy()
            X_scaled[["day", "month", "year", "dayofweek"]] = scaler.fit_transform(
                X_scaled[["day", "month", "year", "dayofweek"]]
            )
            if model_type == "randomforest":
                model = RandomForestRegressor(n_estimators=200, random_state=42)
            else:
                model = SVR(kernel="rbf")
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
                next_date = pd.to_datetime(last_row["ds"].values[0]) + pd.Timedelta(days=1)
                future_dates.append(next_date)
                prev2 = last_row["y"].values[0]
                last_row = pd.DataFrame({"ds": [next_date], "y": [pred]})
            model_label = "RandomForest" if model_type == "randomforest" else "SVR"

        else:
            await update.message.reply_text("❗️Модель не розпізнано. Використай model=prophet, svr або randomforest.")
            return

        plot_buf = plot_forecast(df, future_dates, predicted_values, model_label)
        text = (
            f"📊 Поточна ціна: ${now_price:.2f}\n"
            f"🔮 Прогноз ({model_label}, +{days} дн.): ${predicted_values[-1]:.2f}"
        )
        await update.message.reply_photo(photo=plot_buf, caption=text)

    except Exception as e:
        logger.exception("Помилка під час /predict")
        await update.message.reply_text(f"❌ Помилка: {e}")

# --- Application
application = Application.builder().token(TELEGRAM_TOKEN).build()
application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("predict", predict))

# --- Flask маршрути
@app.route("/")
def index():
    return "✅ Бот працює!"

@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def telegram_webhook():
    if request.method != "POST":
        abort(405)

    try:
        update = Update.de_json(request.get_json(force=True), bot)
    except Exception:
        logger.exception("Не вдалось розпарсити Update")
        return "bad request", 400

    async def process():
        await application.initialize()
        await application.start()
        await application.process_update(update)

    try:
        asyncio.run(process())
    except Exception:
        logger.exception("Помилка при process_update")
        return "error", 500

    return "ok", 200

# --- Головний блок
if __name__ == "__main__":
    logger.info("Встановлюю webhook...")
    try:
        asyncio.run(bot.set_webhook(f"{WEBHOOK_URL}/{TELEGRAM_TOKEN}"))
        logger.info("Webhook встановлено: %s/%s", WEBHOOK_URL, TELEGRAM_TOKEN)
    except Exception:
        logger.exception("Не вдалось встановити webhook")
        raise

    port = int(os.environ.get("PORT", 5000))
    logger.info("Стартую Flask на порту %s", port)
    app.run(host="0.0.0.0", port=port)
