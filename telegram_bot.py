import os
import zipfile
import logging
import torch
import gdown
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# Отключаем FlexAttention
os.environ["TRANSFORMERS_NO_FLEX_ATTENTION"] = "1"

# Заглушки для совместимости
if not hasattr(torch, "compiler"):
    class DummyCompiler:
        @staticmethod
        def disable(recursive=False):
            def decorator(func): return func
            return decorator
    torch.compiler = DummyCompiler()

if not hasattr(torch, "float8_e4m3fn"):
    torch.float8_e4m3fn = torch.float32

# Пути
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "dialogpt-small"
ZIP_PATH = BASE_DIR / "dialogpt-small.zip"
TOKENIZER_JSON = MODEL_DIR / "tokenizer.json"

# ✅ Скачиваем и распаковываем модель
if not MODEL_DIR.exists():
    print("\U0001F4E6 Загружаю модель с Google Drive...")
    url = "https://drive.google.com/uc?id=1J_uFKwD5ktNwES6SZJSdXnH5LQFxKBVH"
    gdown.download(url, str(ZIP_PATH), quiet=False)

    print("\U0001F4C2 Распаковываю архив...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(BASE_DIR)

    print("\U0001F4C1 Перемещаю файлы в dialogpt-small/")
    extracted_folder = BASE_DIR / "dialogpt-small-main"
    if extracted_folder.exists():
        for item in extracted_folder.iterdir():
            item.rename(MODEL_DIR / item.name)
        extracted_folder.rmdir()

    print("✅ Модель распакована.")

if not TOKENIZER_JSON.exists():
    raise FileNotFoundError(f"❌ Файл {TOKENIZER_JSON} не найден.")

# Загружаем токенизатор и модель
print("🤖 Загружаю модель...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True).to("cpu")

# История сообщений
chat_histories = {}

# Логгирование
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Привет! Я бот на DialoGPT. Напиши мне что-нибудь!")

# Ответ на сообщение
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_message = update.message.text

    new_input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors="pt")

    if chat_id not in chat_histories or chat_histories[chat_id].shape[-1] > 256:
        bot_input_ids = new_input_ids
    else:
        bot_input_ids = torch.cat([chat_histories[chat_id], new_input_ids], dim=-1)

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=200,
        pad_token_id=tokenizer.eos_token_id
    )

    chat_histories[chat_id] = chat_history_ids
    response_ids = chat_history_ids[:, bot_input_ids.shape[-1]:]
    bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True).strip()

    if bot_response:
        await update.message.reply_text(bot_response)
    else:
        await update.message.reply_text("🤔 Я не понял. Попробуй переформулировать.")

# Запуск
async def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ Переменная TELEGRAM_BOT_TOKEN не найдена.")
        return

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Бот запущен.")
    await app.run_polling()

if __name__ == "__main__":
    import nest_asyncio
    import asyncio

    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())
