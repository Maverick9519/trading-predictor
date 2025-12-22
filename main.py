import os
import io
import json
import time
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

from flask import Flask, request
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from dotenv import load_dotenv
load_dotenv()

# ================= Config =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CMC_API_KEY = os.getenv("CMC_API_KEY")

MODEL_FILE = "user_models.json"
LOG_FILE = "prediction_log.csv"

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ================= Flask App =================
app = Flask(__name__)

# ================= User Models =================
def load_user_models():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_models(data):
    with open(MODEL_FILE, 'w') as f:
        json.dump(data, f)

user_models = load_user_models()

# ================= Crypto Data =================
def load_crypto_data():
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical"
    params = {
        "symbol": "BTC",
        "convert": "USD",
        "time_start": (datetime.datetime.now() - datetime.timedelta(days=100)).strftime("%Y-%m-%d"),
        "time_end": datetime.datetime.now().strftime("%Y-%m-%d"),
    }
    headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    data = r.json()
    if "data" not in data or "quotes" not in data["data"]:
        raise RuntimeError("Не вдалося завантажити дані з CoinMarketCap")

    rows = []
    for quote in data["data"]["quotes"]:
        ts = datetime.datetime.fromisoformat(quote["time_open"].replace("Z",""))
        price = quote["quote"]["USD"]["close"]
        rows.append({"Date": ts, "Price": price})

    df = pd.DataFrame(rows)
    df['Moving_Avg_10'] = df['Price'].rolling(10).mean()
    df['Moving_Avg_50'] = df['Price'].rolling(50).mean()
    df['Volatility'] = df['Price'].pct_change().rolling(10).std()
    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Target'] = df['Price'].shift(-1)
    return df.dropna()

def train_model(df, model_type='LinearRegression'):
    features = ['Price','Moving_Avg_10','Moving_Avg_50','Volatility','RSI']
    X = df[features]
    y = df['Target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

    if model_type=='SVR':
        base_model = SVR()
    elif model_type=='RandomForest':
        base_model = RandomForestRegressor(n_estimators=100)
    else:
        base_model = LinearRegression()

    model = TransformedTargetRegressor(regressor=base_model, transformer=StandardScaler())
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return model, df.loc[y_test.index], y_test, predictions, mse

def plot_prediction(df_test, y_test, predictions):
    plt.figure(figsize=(10,5))
    plt.plot(df_test['Date'], y_test.values, label='Real')
    plt.plot(df_test['Date'], predictions, label='Predicted')
    plt.legend()
    plt.title("Crypto Price Prediction (CMC)")
    plt.xticks(rotation=45)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def get_prediction_text_and_plot(model_type='LinearRegression', user_id='anonymous'):
    start_time = time.time()
    df = load_crypto_data()
    model, df_test, y_test, predictions, mse = train_model(df, model_type)
    plot_buf = plot_prediction(df_test, y_test, predictions)
    elapsed_time = time.time()-start_time
    total_prediction = np.sum(predictions)
    # log
    df_log = pd.DataFrame([{
        "timestamp": datetime.datetime.now().isoformat(),
        "user_id": user_id,
        "model_type": model_type,
        "mse": round(mse,4),
        "prediction_preview": list(np.round(predictions[:3],2)),
        "prediction_sum": round(total_prediction,2),
        "elapsed_time": round(elapsed_time,2)
    }])
    if os.path.exists(LOG_FILE):
        df_log.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df_log.to_csv(LOG_FILE, mode='w', header=True, index=False)
    text = (f"Модель: {model_type}\nMSE: {mse:.2f}\nСума прогнозу: {total_prediction:.2f}\nЧас прогнозування: {elapsed_time:.2f} сек.")
    return text, plot_buf

# ================= Telegram Handlers =================
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Я трейдинг-прогнозатор бот.\n/predict — прогноз\n/model — обрати модель\n/log — останні 5 прогнозів"
    )

async def predict_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    model_type = user_models.get(user_id, 'LinearRegression')
    await update.message.reply_text(f"Прогноз за моделлю: {model_type}...")
    text, plot_buf = get_prediction_text_and_plot(model_type, user_id)
    await update.message.reply_text(text)
    await update.message.reply_photo(photo=plot_buf)

async def model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if context.args:
        model = context.args[0]
        if model in ['LinearRegression','SVR','RandomForest']:
            user_models[user_id]=model
            save_user_models(user_models)
            await update.message.reply_text(f"Модель встановлено: {model}")
        else:
            await update.message.reply_text("Доступні моделі: LinearRegression, SVR, RandomForest")
    else:
        await update.message.reply_text("Використання: /model LinearRegression")

async def log_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not os.path.exists(LOG_FILE):
        await update.message.reply_text("Журнал прогнозів порожній.")
        return
    df = pd.read_csv(LOG_FILE)
    df_user = df[df['user_id']==user_id].tail(5)
    if df_user.empty:
        await update.message.reply_text("Для вас ще не збережено прогнозів.")
        return
    log_text="Останні 5 прогнозів:\n"
    for _, row in df_user.iterrows():
        log_text += f"- {row['timestamp'][:19]}\n  Модель: {row['model_type']}, MSE: {row['mse']}, Сума: {row.get('prediction_sum','N/A')}, Час: {row.get('elapsed_time','N/A')} сек, Прогноз: {row['prediction_preview']}\n"
    await update.message.reply_text(log_text)

# ================= Application =================
app_bot = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
app_bot.add_handler(CommandHandler("start", start_cmd))
app_bot.add_handler(CommandHandler("predict", predict_cmd))
app_bot.add_handler(CommandHandler("model", model_cmd))
app_bot.add_handler(CommandHandler("log", log_cmd))

# ================= Webhook =================
@app.route(f"/{TELEGRAM_TOKEN}", methods=["POST"])
def webhook():
    update = Update.de_json(request.get_json(force=True), app_bot.bot)
    app_bot.update_queue.put(update)
    return "OK", 200

@app.route("/")
def index():
    return "Бот працює!", 200

if __name__=="__main__":
    # Webhook для Render
    url = f"https://<YOUR_RENDER_DOMAIN>.onrender.com/{TELEGRAM_TOKEN}"
    app_bot.bot.set_webhook(url=url)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
