import os
import io
import time
import json
import logging
import datetime
import requests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ================= ENV LOADING =================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # on server no dotenv

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CMC_API_KEY = os.getenv("CMC_API_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN не встановлено!")
if not CMC_API_KEY:
    raise RuntimeError("CMC_API_KEY не встановлено!")

# ================= LOGGING =================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

MODEL_FILE = "user_models.json"
LOG_FILE = "prediction_log.csv"

# ================= USER MODELS =================
def load_user_models():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "r") as f:
            return json.load(f)
    return {}

def save_user_models(data):
    with open(MODEL_FILE, "w") as f:
        json.dump(data, f)

user_models = load_user_models()

# ================= LOG PREDICTION =================
def log_prediction(user_id, model_type, mse, predictions, elapsed_time, total_prediction):
    df_log = pd.DataFrame([{
        "timestamp": datetime.datetime.now().isoformat(),
        "user_id": user_id,
        "model_type": model_type,
        "mse": round(mse, 4),
        "prediction_preview": list(np.round(predictions[:3], 2)),
        "prediction_sum": round(total_prediction, 2),
        "elapsed_time": round(elapsed_time, 2)
    }])
    if os.path.exists(LOG_FILE):
        df_log.to_csv(LOG_FILE, mode="a", header=False, index=False)
    else:
        df_log.to_csv(LOG_FILE, mode="w", header=True, index=False)

# ================= LOAD DATA FROM COINMARKETCAP =================
def fetch_cmc_historical():
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
    headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
    end_dt = datetime.datetime.utcnow()
    start_dt = end_dt - datetime.timedelta(days=90)  # 90 days history
    params = {
        "symbol": "BTC",
        "convert": "USD",
        "time_start": start_dt.isoformat(),
        "time_end": end_dt.isoformat(),
    }
    r = requests.get(url, headers=headers, params=params, timeout=20)
    data = r.json()
    if "data" not in data or "quotes" not in data["data"]:
        raise RuntimeError("Не вдалося отримати історію з CoinMarketCap")
    return data["data"]["quotes"]

def load_crypto_data():
    quotes = fetch_cmc_historical()
    rows = []
    for q in quotes:
        rows.append({
            "Date": pd.to_datetime(q["timestamp"]),
            "Price": q["quote"]["USD"]["close"]
        })
    df = pd.DataFrame(rows).sort_values("Date")
    df["Moving_Avg_10"] = df["Price"].rolling(10).mean()
    df["Moving_Avg_50"] = df["Price"].rolling(50).mean()
    df["Volatility"] = df["Price"].pct_change().rolling(10).std()

    delta = df["Price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    df["Target"] = df["Price"].shift(-1)
    return df.dropna()

# ================= TRAIN MODEL =================
def train_model(df, model_type="LinearRegression"):
    features = ["Price", "Moving_Avg_10", "Moving_Avg_50", "Volatility", "RSI"]
    X = df[features]
    y = df["Target"]

    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

    if model_type == "SVR":
        base = SVR()
    elif model_type == "RandomForest":
        base = RandomForestRegressor(n_estimators=100)
    else:
        base = LinearRegression()

    model = TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return model, df.loc[y_test.index], y_test, preds, mse

# ================= PLOT =================
def plot_prediction(df_test, y_test, preds):
    plt.figure(figsize=(10, 5))
    plt.plot(df_test["Date"], y_test.values, label="Real")
    plt.plot(df_test["Date"], preds, label="Predicted")
    plt.legend()
    plt.title("BTC Price Prediction (CMC)")
    plt.xticks(rotation=45)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plt.close()
    return buf

def get_prediction_text_and_plot(model_type="LinearRegression", user_id="anonymous"):
    start = time.time()
    df = load_crypto_data()
    model, df_test, y_test, preds, mse = train_model(df, model_type)
    buf = plot_prediction(df_test, y_test, preds)
    elapsed = time.time() - start
    total_pred = np.sum(preds)
    log_prediction(user_id, model_type, mse, preds, elapsed, total_pred)

    text = (
        f"Модель: {model_type}\n"
        f"MSE: {mse:.2f}\n"
        f"Сума: {total_pred:.2f}\n"
        f"Час: {elapsed:.2f} сек"
    )
    return text, buf

# ================= TELEGRAM HANDLERS =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Бот з CoinMarketCap даними\n"
        "/predict — прогноз\n"
        "/model — вибір моделі\n"
        "/log — історія прогнозів"
    )

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    model_type = user_models.get(user_id, "LinearRegression")
    await update.message.reply_text(f"Прогнозуємо ({model_type})…")
    try:
        text, pic = get_prediction_text_and_plot(model_type, user_id)
        await update.message.reply_text(text)
        await update.message.reply_photo(photo=pic)
    except Exception as e:
        logging.exception("Prediction error")
        await update.message.reply_text("Помилка під час прогнозу.")

async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if context.args:
        m = context.args[0]
        if m in ["LinearRegression", "SVR", "RandomForest"]:
            user_models[user_id] = m
            save_user_models(user_models)
            await update.message.reply_text(f"Модель: {m}")
        else:
            await update.message.reply_text("Доступні: LinearRegression, SVR, RandomForest")

async def show_log(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(LOG_FILE):
        await update.message.reply_text("Немає логів.")
        return
    df = pd.read_csv(LOG_FILE)
    uid = str(update.effective_user.id)
    df_user = df[df["user_id"] == uid].tail(5)
    if df_user.empty:
        await update.message.reply_text("Ще немає прогнозів.")
        return
    msg = "Останні 5 прогнозів:\n"
    for _, r in df_user.iterrows():
        msg += f"{r['timestamp']} | {r['model_type']} | MSE {r['mse']}\n"
    await update.message.reply_text(msg)

# ================= MAIN =================
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("model", set_model))
    app.add_handler(CommandHandler("log", show_log))
    logging.info("Bot started with CoinMarketCap")
    app.run_polling()

if __name__ == "__main__":
    main()
