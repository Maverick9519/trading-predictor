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
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
)
import logging
import io
import json
import os
import datetime
import time
import requests

# === Telegram Token ===
TELEGRAM_TOKEN = '7632093001:AAGojU_FXYAWGfKTZAk3w7fuOhLxKoXdi6Y'

# === Файли ===
MODEL_FILE = "user_models.json"
LOG_FILE = "prediction_log.csv"

# === Логування ===
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# === Завантаження моделей ===
def load_user_models():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_user_models(data):
    with open(MODEL_FILE, 'w') as f:
        json.dump(data, f)

user_models = load_user_models()

def log_prediction(user_id, model_type, mse, predictions, elapsed_time, last_prediction):
    df_log = pd.DataFrame([{
        "timestamp": datetime.datetime.now().isoformat(),
        "user_id": user_id,
        "model_type": model_type,
        "mse": round(mse, 4),
        "prediction_preview": list(np.round(predictions[:4], 2)),
        "last_prediction": round(last_prediction, 4),
        "elapsed_time": round(elapsed_time, 2)
    }])
    if os.path.exists(LOG_FILE):
        df_log.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df_log.to_csv(LOG_FILE, mode='w', header=True, index=False)

def load_crypto_data():
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=100"
    response = requests.get(url)
    data = response.json()

    prices = np.array([item[1] for item in data['prices']])
    timestamps = [datetime.datetime.fromtimestamp(item[0] / 1000) for item in data['prices']]
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

def plot_prediction(df_test, y_test, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(df_test['Date'], y_test.values, label='Real')
    plt.plot(df_test['Date'], predictions, label='Predicted')
    plt.legend()
    plt.title("Crypto Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Price")
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
    elapsed_time = time.time() - start_time
    last_prediction = predictions[-1]
    log_prediction(user_id, model_type, mse, predictions, elapsed_time, last_prediction)

    prediction_formatted = "{:.2f}".format(last_prediction)

    text = (
        f"Модель: {model_type}\n"
        f"Mean Squared Error: {mse:.2f}\n"
        f"Останнє передбачене значення: {prediction_formatted} USD\n"
        f"Час прогнозування: {elapsed_time:.2f} сек."
    )
    return text, plot_buf

# === Telegram команди ===

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привіт! Я трейдинг-прогнозатор бот.\n"
        "Команди:\n"
        "/predict — отримати прогноз\n"
        "/model [LinearRegression|SVR|RandomForest] — обрати модель\n"
        "/log — останні 5 прогнозів"
    )

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_user.id)
    model_type = user_models.get(user_id, 'LinearRegression')
    await update.message.reply_text(f"Прогноз за моделлю: {model_type}...")
    try:
        text, plot_buf = get_prediction_text_and_plot(model_type, user_id)
        await update.message.reply_text(text)
        await update.message.reply_photo(photo=plot_buf)
    except Exception as e:
        await update.message.reply_text(f"Помилка: {e}")
        logging.exception("Error during prediction")

async def set_model(update: Update, context: ContextTypes.DEFAULT_TYPE):
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

async def show_log(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
            f"Прогноз: {row.get('last_prediction', 'N/A')} USD, "
            f"Час: {row.get('elapsed_time', 'N/A')} сек, "
            f"Перші значення: {row['prediction_preview']}\n"
        )
    await update.message.reply_text(log_text)

# === Головна функція ===

def run_bot_and_server():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("predict", predict))
    application.add_handler(CommandHandler("model", set_model))
    application.add_handler(CommandHandler("log", show_log))
    application.run_polling()

if __name__ == '__main__':
    run_bot_and_server()
