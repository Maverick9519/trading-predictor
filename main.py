# ===== SYSTEM =====
import os
import io
import time
import json
import asyncio
import logging
import datetime
import requests
from dotenv import load_dotenv

# ===== DATA =====
import numpy as np
import pandas as pd

# ===== ML =====
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

# ===== MATPLOTLIB =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===== TELEGRAM =====
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from aiohttp import web

# ================= CONFIG =================
load_dotenv()  # Для локального запуску через .env

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CMC_API_KEY = os.getenv("CMC_API_KEY")
APP_URL = os.getenv("APP_URL")  # https://твій_домен.onrender.com

MODEL_FILE = "user_models.json"
LOG_FILE = "prediction_log.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ================= STORAGE =================
def load_user_models():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "r") as f:
            return json.load(f)
    return {}

def save_user_models(data):
    with open(MODEL_FILE, "w") as f:
        json.dump(data, f)

user_models = load_user_models()

# ================= COINMARKETCAP =================
def load_crypto_data():
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
    params = {"start": "1", "limit": "50", "convert": "USD"}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        data = r.json()
        if "data" not in data:
            raise RuntimeError("CoinMarketCap API error: no 'data' in response")
        btc_data = next((item for item in data["data"] if item["symbol"] == "BTC"), None)
        if not btc_data:
            raise RuntimeError("CoinMarketCap API error: BTC not found")
        df = pd.DataFrame([{"Date": datetime.datetime.now(), "Price": btc_data["quote"]["USD"]["price"]}])
        df = pd.concat([df]*100, ignore_index=True)
        df["MA10"] = df["Price"].rolling(10).mean()
        df["MA30"] = df["Price"].rolling(30).mean()
        df["Volatility"] = df["Price"].pct_change().rolling(10).std()
        df["Target"] = df["Price"].shift(-1)
        return df.dropna()
    except requests.RequestException as e:
        raise RuntimeError(f"CoinMarketCap API request failed: {str(e)}")

# ================= ML =================
def train_model(df):
    features = ["Price", "MA10", "MA30", "Volatility"]
    X = df[features]
    y = df["Target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    model = TransformedTargetRegressor(regressor=LinearRegression(), transformer=StandardScaler())
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, df.iloc[-len(y_test):], y_test, predictions, mse

# ================= PLOT =================
def plot_prediction(df_test, y_test, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(df_test["Date"], y_test.values, label="Real")
    plt.plot(df_test["Date"], predictions, label="Predicted")
    plt.legend()
    plt.title("BTC Price Prediction (CoinMarketCap)")
    plt.xticks(rotation=45)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

# ================= LOG =================
def log_prediction(user_id, mse, total, elapsed):
    row = pd.DataFrame([{
        "time": datetime.datetime.now().isoformat(),
        "user_id": user_id,
        "mse": round(mse, 4),
        "sum_prediction": round(total, 2),
        "elapsed": round(elapsed, 2)
    }])
    if os.path.exists(LOG_FILE):
        row.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(LOG_FILE, index=False)

# ================= CORE =================
def make_prediction(user_id):
    start = time.time()
    df = load_crypto_data()
    _, df_test, y_test, predictions, mse = train_model(df)
    plot = plot_prediction(df_test, y_test, predictions)
    elapsed = time.time() - start
    total = float(np.sum(predictions))
    log_prediction(user_id, mse, total, elapsed)
    text = f"📈 BTC прогноз\nMSE: {mse:.2f}\nСума прогнозу: {total:.2f}\nЧас: {elapsed:.2f} сек"
    return text, plot

# ================= TELEGRAM HANDLERS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🤖 Crypto прогнозатор (CoinMarketCap)\n\n/predict — отримати прогноз")

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    await update.message.reply_text("⏳ Розрахунок прогнозу...")
    try:
        loop = asyncio.get_running_loop()
        text, plot = await loop.run_in_executor(None, make_prediction, user_id)
        await update.message.reply_text(text)
        await update.message.reply_photo(photo=plot)
    except Exception as e:
        logging.exception("Prediction failed")
        await update.message.reply_text(f"❌ Помилка:\n{str(e)}")

# ================= WEBHOOK =================
async def webhook(request):
    logging.info("🔔 Webhook received request")
    data = await request.json()
    update = Update.de_json(data, app.bot)
    await app.update_queue.put(update)
    return web.Response(text="ok")

# ================= MAIN =================
def main():
    global app
    if not TELEGRAM_TOKEN or not CMC_API_KEY or not APP_URL:
        logging.error("❌ Будь ласка, встановіть TELEGRAM_TOKEN, CMC_API_KEY та APP_URL у Environment Variables")
        return

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))

    # Встановлюємо webhook
    webhook_url = f"{APP_URL}/webhook"
    logging.info(f"🔗 Setting webhook to {webhook_url}")
    import requests
    r = requests.get(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/setWebhook?url={webhook_url}")
    logging.info(r.json())

    # Запускаємо aiohttp сервер
    web_app = web.Application()
    web_app.router.add_post("/webhook", webhook)
    web.run_app(web_app, port=int(os.environ.get("PORT", 10000)))

if __name__ == "__main__":
    main()
