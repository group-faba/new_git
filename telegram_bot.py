import os
import zipfile
import logging
import torch
import gdown
import shutil
from pathlib import Path
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM
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
MODEL_DIR = Path("dialogpt-small").resolve()
ZIP_PATH = "dialogpt-small.zip"

# ✅ Скачиваем и распаковываем модель
if not MODEL_DIR.exists():
    print("📦 Загружаю модель с Google Drive...")
    file_id = "1J_uFKwD5ktNwES6SZJSdXnH5LQFxKBVH"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, ZIP_PATH, quiet=False)

    print("📂 Распаковываю архив...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall("temp_extracted")

    print("📁 Перемещаю файлы в dialogpt-small/")
    MODEL_DIR.mkdir(exist_ok=True)
    for file in os.listdir("temp_extracted"):
        src = os.path.join("temp_extracted", file)
        if os.path.isfile(src):
            shutil.move(src, MODEL_DIR / file)
    shutil.rmtree("temp_extracted")
    print("✅ Модель распакована.")

# Проверка наличия нужного файла
TOKENIZER_JSON = MODEL_DIR / "tokenizer.json"
if not TOKENIZER_JSON.exists():
    raise FileNotFoundError(f"❌ Файл {TOKENIZER_JSON} не найден.")

# Загрузка токенизатора и модели
tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(TOKENIZER_JSON))
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True).to("cpu")

# Истории диалогов
chat_histories = {}

# Логгирование
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# Команда /start
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
    bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    await update.message.reply_text(bot_response)

# Запуск
async def main():
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("❌ Переменная TELEGRAM_BOT_TOKEN не найдена.")
        return

    application = ApplicationBuilder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Бот запущен.")
    await application.run_polling()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
