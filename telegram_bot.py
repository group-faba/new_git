import os, zipfile
import gdown

MODEL_DIR = "dialogpt-small"
MODEL_ZIP_PATH = "dialogpt-small.zip"
GOOGLE_DRIVE_ID = "ТВОЙ_ID_ИЗ_ССЫЛКИ"

if not os.path.exists(MODEL_DIR):
    print("📦 Загружаю модель с Google Drive...")
    url = f"https://drive.google.com/uc?id=1vaokB9Ozqil3rrVn0GchaQ8klJvYV4HS"
    gdown.download(url, MODEL_ZIP_PATH, quiet=False)
    with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(MODEL_DIR)
    print("✅ Модель распакована.")

from transformers import AutoTokenizer, AutoModelForCausalLM
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
)
import os, torch, logging

# Инициализация
tokenizer = AutoTokenizer.from_pretrained("./dialogpt-small")
model = AutoModelForCausalLM.from_pretrained("./dialogpt-small")
chat_histories = {}

logging.basicConfig(level=logging.INFO)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я DialoGPT-бот.")

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    msg = update.message.text

    input_ids = tokenizer.encode(msg + tokenizer.eos_token, return_tensors="pt")
    if chat_id not in chat_histories or chat_histories[chat_id].shape[-1] > 256:
        bot_input = input_ids
    else:
        bot_input = torch.cat([chat_histories[chat_id], input_ids], dim=-1)

    history = model.generate(bot_input, max_length=200, pad_token_id=tokenizer.eos_token_id)
    chat_histories[chat_id] = history
    response = tokenizer.decode(history[:, bot_input.shape[-1]:][0], skip_special_tokens=True)
    await update.message.reply_text(response)

async def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    await app.run_polling()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
