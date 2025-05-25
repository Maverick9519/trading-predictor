import os
import threading
import asyncio
import logging
import datetime
import random
import matplotlib.pyplot as plt
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

auto_tasks = {}

# === Функція для автоматичного прогнозу з графіком ===
async def auto_predict(context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_id = context.job.chat_id

        # Симульовані значення — заміни на свої реальні
        current_price = 107628.76
        forecast_price = 113010.20
        change = forecast_price - current_price
        change_percent = (change / current_price) * 100

        # Побудова графіку
        today = datetime.date.today()
        dates = [today + datetime.timedelta(days=i*30) for i in range(6)]
        prices = [current_price + (change * i / 5) + random.uniform(-100, 100) for i in range(6)]

        plt.figure(figsize=(8, 4))
        plt.plot(dates, prices, marker='o')
        plt.title("Bitcoin (поточна ціна)")
        plt.xlabel("Дата")
        plt.ylabel("Ціна (USD)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("chart.png")
        plt.close()

        # Відправлення повідомлення і графіку
        await context.bot.send_message(
            chat_id=chat_id,
            text=f"📊 Поточна ціна: ${current_price}\n🔮 Прогноз: ${forecast_price}\n📈 Зміна: ${round(change, 2)} ({round(change_percent, 2)}%)"
        )
        await context.bot.send_photo(
            chat_id=chat_id,
            photo=open("chart.png", "rb")
        )

    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Помилка при автопрогнозі: {e}")


# === Команда /stop
async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in auto_tasks:
        auto_tasks[chat_id].schedule_removal()
        del auto_tasks[chat_id]
        await update.message.reply_text("⛔️ Авто-прогноз зупинено.")
    else:
        await update.message.reply_text("❗️ Авто-прогноз не запущено.")


# === Команда /auto (оновлена з графіком)
async def auto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        minutes = int(context.args[0]) if context.args else 1
        chat_id = update.effective_chat.id

        if chat_id in auto_tasks:
            auto_tasks[chat_id].schedule_removal()

        job = context.job_queue.run_repeating(
            auto_predict, interval=minutes * 60, first=0, chat_id=chat_id
        )
        auto_tasks[chat_id] = job
        await update.message.reply_text(f"✅ Авто-прогноз кожні {minutes} хвилин.")
    except Exception as e:
        await update.message.reply_text(f"❌ Помилка: {e}")


# === Запуск Telegram бота
async def run_bot():
    logging.info("🚀 Запуск Telegram бота")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("predict", predict))
    app.add_handler(CommandHandler("custom", custom))
    app.add_handler(CommandHandler("auto", auto))
    app.add_handler(CommandHandler("stop", stop))
    await app.initialize()
    await app.start()
    await app.updater.start_polling()


# === Головний запуск
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 4000))
    threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=port), daemon=True).start()
    loop = asyncio.get_event_loop()
    loop.create_task(run_bot())
    loop.run_forever()
