import os
import zipfile
import logging
import torch
import gdown
import shutil
import asyncio
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

# Настройка логирования
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

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
MODEL_NAME = "dialogpt-small"
MODEL_DIR = Path(f"./{MODEL_NAME}").resolve()
ZIP_PATH = f"{MODEL_NAME}.zip"
TOKENIZER_JSON = MODEL_DIR / "tokenizer.json"

# ✅ Скачиваем и распаковываем модель, если нужно
if not TOKENIZER_JSON.exists():
    print("📦 Загружаю модель с Google Drive...")
    url = "https://drive.google.com/uc?id=1J_uFKwD5ktNwES6SZJSdXnH5LQFxKBVH"
    gdown.download(url, ZIP_PATH, quiet=False)

    print("📂 Распаковываю архив...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall("tmp_extract")

    print("📁 Перемещаю файлы в dialogpt-small/")
    extracted_dir = Path("tmp_extract") / MODEL_NAME
    if extracted_dir.exists():
        if not MODEL_DIR.exists():
            shutil.move(str(extracted_dir), str(MODEL_DIR))
        shutil.rmtree("tmp_extract")
    else:
        raise FileNotFoundError("❌ Не найдена папка с моделью внутри архива.")

    if not TOKENIZER_JSON.exists():
        raise FileNotFoundError(f"❌ Файл {TOKENIZER_JSON} не найден.")
    print("✅ Модель распакована.")

# Загружаем модель и токенизатор
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True).to("cpu")

# Истории сообщений
chat_histories = {}

# Обработка команд
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Привет! Я бот на DialoGPT. Напиши мне что-нибудь.")

# Обработка сообщений
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
    bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    await update.message.reply_text(bot_response)

# Главная функция запуска
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

# ✅ Запуск (без asyncio.run)
if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())
