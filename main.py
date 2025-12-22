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
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

from dotenv import load_dotenv
load_dotenv()

# ================= CONFIG =================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CMC_API_KEY = os.getenv("CMC_API_KEY")

if not TELEGRAM_TOKEN or not CMC_API_KEY:
    raise RuntimeError("TELEGRAM_TOKEN або CMC_API_KEY не встановлено у .env")

# ================= FILES =================
MODEL_FILE = "user_models.json"
LOG_FILE = "prediction_log.csv"

# ================= LOGGING =================
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ================= USER MODELS =================
def load_user_models():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_models(data):
    with open(MODEL_FILE, 'w') as f:
        json.dump(data, f)

user_models = load_user_models()

# ================= LOGGING PREDICTIONS =================
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
        df_log.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df_log.to_csv(LOG_FILE, mode='w', header=True, index=False)

# ================= LOAD CRYPTO DATA =================
def load_crypto_data():
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/historical"
    params = {
        "symbol": "BTC",
        "time_start": int(time.time()) - 86400*100,  # 100 днів назад
        "interval": "daily"
    }
    headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}

    r = requests.get(url, headers=headers, params=params)
    if r.status_code != 200:
        raise RuntimeError(f"CMC API error: {r.status_code} {r.text}")

    data = r.json()
    try:
        quotes = data["data"]["quotes"]
        prices = np.array([float(q['quote']['USD']['close']) for q in quotes])
        timestamps = [datetime.datetime.fromisoformat(q['time_open']) for q in quotes]
    except Exception as e:
        raise RuntimeError(f"Помилка обробки даних CMC: {e}")

    df = pd.DataFrame({'Date': timestamps, 'Price': prices})

    df['Moving_Avg_10'] = df['Price'].rolling(window=10).mean()
    df['Moving_Avg_50'] = df['Price'].rolling(window=50).mean()
    df['Volatility'] = df['Price'].pct_change().rolling(window=10).std()

    delta = df['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Target'] = df['Price'].shift(-1)
    return df.dropna()

# ================= TRAIN MODEL =================
def train_model(df, model_type='LinearRegression'):
    features = ['Price', 'Moving_Avg_10', 'Moving_Avg_50', 'Volatility', 'RSI']
    X = df[features]
    y = df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, shuffle=False, test_size=0.2)

    if model_type == 'SVR':
        base_model = SVR()
    elif model_type == 'RandomForest':
        base_model = RandomForestRegressor(n_estimators=100)
    else:
        base_model = LinearRegression()

    model = TransformedTargetRegressor(regressor=base_model, transformer=StandardScaler())
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    return model, df.loc[y_test.index], y_test, predictions, mse

# ================= PLOT =================
def plot_prediction(df_test, y_test, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(df_test['Date'], y_test.values, label='Real')
    plt.plot(df_test['Date'], predictions, label='Predicted')
    plt.legend()
    plt.title("BTC Price Prediction (CMC)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# ================= PREDICTION =================
def get_prediction_text_and_plot(model_type='LinearRegression', user_id='anonymous'):
    start_time = time.time()
    df = load_crypto_data()
    model, df_test, y_test, predictions, mse = train_model(df, model_type)
    plot_buf = plot_prediction(df_test, y_test, predictions)
    elapsed_time = time.time() - start_time
    total_prediction = np.sum(predictions)
    log_prediction(user_id, model_type, mse, predictions, elapsed_time, total_prediction)

    text = (
        f"Модель: {model_type}\n"
        f"Mean Squared Error: {mse:.2f}\n"
        f"Сума прогнозу: {total_prediction:.2f}\n"
        f"Час прогнозування: {elapsed_time:.2f} сек."
    )
    return text, plot_buf

# ================= TELEGRAM HANDLERS =================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Я трейдинг-прогнозатор бот.\n"
        "Команди:\n"
        "/predict — отримати прогноз\n"
        "/model [LinearRegression|SVR|RandomForest] — обрати модель\n"
        "/log — останні 5 прогнозів"
    )

async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    model_type = user_models.get(user_id, 'LinearRegression')
    await update.message.reply_text(f"Прогноз за моделлю: {model_type}...")
    try:
        text, plot_buf = get_prediction_text_and_plot(model_type, user_id)
        await update.message.reply_text(text)
        await update.message.reply_photo(photo=plot_buf)
    except Exception as e:
        await update.message.reply_text(f"Виникла помилка під час прогнозування: {e}")
        logging.exception("Error during prediction")

async def set_model_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if context.args:
        model = context.args[0]
        if model in ['LinearRegression', 'SVR', 'RandomForest']:
            user_models[user_id] = model
            save_user_models(user_models)
            await update.message.reply_text(f"Модель встановлено: {model}")
        else:
            await update.message.reply_text("Доступні моделі: LinearRegression, SVR, RandomForest")
    else:
        await update.message.reply_text("Використання: /model LinearRegression")

async def show_log_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    if not os.path.exists(LOG_FILE):
        await update.message.reply_text("Журнал прогнозів порожній.")
        return
    df = pd.read_csv(LOG_FILE)
    df_user = df[df['user_id'] == user_id].tail(5)
    if df_user.empty:
        await update.message.reply_text("Для вас ще не збережено прогнозів.")
        return
    log_text = "Останні 5 прогнозів:\n"
    for _, row in df_user.iterrows():
        log_text += (
            f"- {row['timestamp'][:19]}\n"
            f"  Модель: {row['model_type']}, MSE: {row['mse']}, "
            f"Сума: {row.get('prediction_sum', 'N/A')}, "
            f"Час: {row.get('elapsed_time', 'N/A')} сек, "
            f"Прогноз: {row['prediction_preview']}\n"
        )
    await update.message.reply_text(log_text)

# ================= MAIN =================
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("predict", predict_command))
    app.add_handler(CommandHandler("model", set_model_command))
    app.add_handler(CommandHandler("log", show_log_command))
    print("Бот запущено...")
    app.run_polling()

if __name__ == '__main__':
    main()
