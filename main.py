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

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in auto_tasks:
        auto_tasks[chat_id].schedule_removal()
        del auto_tasks[chat_id]
        await update.message.reply_text("⛔️ Авто-прогноз зупинено.")
    else:
        await update.message.reply_text("❗️ Авто-прогноз не запущено.")

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
if name == '__main__':
    port = int(os.environ.get("PORT", 4000))
    threading.Thread(target=lambda: flask_app.run(host='0.0.0.0', port=port), daemon=True).start()
    loop = asyncio.get_event_loop()
    loop.create_task(run_bot())
    loop.run_forever()
