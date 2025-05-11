import os
import time
import io
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Збір даних з CoinGecko
COINGECKO_API_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привіт! Я бот для прогнозування цін на криптовалюту. Напиши /predict для отримання прогнозу.")

def load_crypto_data():
    params = {
        'vs_currency': 'usd',
        'days': '100',
        'interval': 'daily'
    }
    response = requests.get(COINGECKO_API_URL, params=params)
    data = response.json()
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'Price'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['Price'] = df['Price'].astype(float)
    df = df.drop(columns=['timestamp'])
    df['Moving_Avg_10'] = df['Price'].rolling(window=10).mean()
    df['Moving_Avg_50'] = df['Price'].rolling(window=50).mean()
    df['Volatility'] = df['Price'].rolling(window=10).std()
    delta = df['Price'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Target'] = df['Price'].shift(-1)
    return df

def train_model_full(df, model_type='LinearRegression'):
    features = ['Price', 'Moving_Avg_10', 'Moving_Avg_50', 'Volatility', 'RSI']
    df = df.dropna()
    X = df[features]
    y = df['Target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_type == 'SVR':
        base_model = SVR()
    elif model_type == 'RandomForest':
        base_model = RandomForestRegressor(n_estimators=100)
    else:
        base_model = LinearRegression()

    model = TransformedTargetRegressor(regressor=base_model, transformer=StandardScaler())
    model.fit(X_scaled, y)
    return model, scaler, df

def forecast_future_prices(model, df, scaler, days_ahead=5):
    features = ['Price', 'Moving_Avg_10', 'Moving_Avg_50', 'Volatility', 'RSI']
    latest_data = df[features].iloc[-1:].copy()

    predictions = []
    current_date = df['Date'].iloc[-1]

    for _ in range(days_ahead):
        x_input = scaler.transform(latest_data)
        next_price = model.predict(x_input)[0]
        predictions.append((current_date + datetime.timedelta(days=1), next_price))

        new_row = {
            'Price': next_price,
            'Moving_Avg_10': latest_data['Moving_Avg_10'].values[0],
            'Moving_Avg_50': latest_data['Moving_Avg_50'].values[0],
            'Volatility': latest_data['Volatility'].values[0],
            'RSI': latest_data['RSI'].values[0],
        }
        latest_data = pd.DataFrame([new_row])
        current_date += datetime.timedelta(days=1)

    return predictions

def plot_future_predictions(predictions):
    dates = [d for d, _ in predictions]
    values = [p for _, p in predictions]
    plt.figure(figsize=(10, 5))
    plt.plot(dates, values, marker='o', label="Forecast")
    plt.title("Forecasted Crypto Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def get_prediction_text_and_plot(model_type='LinearRegression', user_id='anonymous'):
    start_time = time.time()
    df = load_crypto_data()
    model, scaler, df = train_model_full(df, model_type)
    future_predictions = forecast_future_prices(model, df, scaler, days_ahead=5)
    plot_buf = plot_future_predictions(future_predictions)
    elapsed_time = time.time() - start_time
    total_prediction = sum([p for _, p in future_predictions])

    text = (
        f"Модель: {model_type}\n"
        f"Сума прогнозу на 5 днів: {total_prediction:.2f}\n"
        f"Час прогнозування: {elapsed_time:.2f} сек.\n"
        f"Наступна ціна: {future_predictions[0][1]:.2f}"
    )
    return text, plot_buf

async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model_type = 'LinearRegression'
    user_id = update.effective_user.id
    try:
        text, plot_buf = get_prediction_text_and_plot(model_type=model_type, user_id=user_id)
        await update.message.reply_text(f"Прогноз за моделлю: {model_type}...")
        await update.message.reply_text(text)
        await update.message.reply_photo(photo=plot_buf)
    except Exception as e:
        await update.message.reply_text(f"Помилка: {str(e)}")

if __name__ == '__main__':
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("predict", predict))
    application.run_polling()
