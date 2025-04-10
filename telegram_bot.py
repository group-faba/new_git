import os
import zipfile
import shutil
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

# Пути к модели
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "dialogpt-small"
ZIP_PATH = BASE_DIR / "dialogpt-small.zip"
TOKENIZER_JSON = MODEL_DIR / "tokenizer.json"

# ✅ Скачиваем и распаковываем модель
if not MODEL_DIR.exists():
    print("\U0001F4E6 Загружаю модель с Google Drive...")
    file_id = "1J_uFKwD5ktNwES6SZJSdXnH5LQFxKBVH"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(ZIP_PATH), quiet=False)

    print("\U0001F4C2 Распаковываю архив...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(BASE_DIR)

    print("\U0001F4C1 Перемещаю файлы в dialogpt-small/")
    extracted_path = BASE_DIR / "dialogpt-small"
    if not extracted_path.exists():
        os.mkdir(extracted_path)
    for file in BASE_DIR.glob("*.json"):
        shutil.move(str(file), str(MODEL_DIR / file.name))
    for file in BASE_DIR.glob("*.txt"):
        shutil.move(str(file), str(MODEL_DIR / file.name))
    for file in BASE_DIR.glob("*.safetensors"):
        shutil.move(str(file), str(MODEL_DIR / file.name))
    print("✅ Модель распакована.")

# Проверяем наличие tokenizer.json
if not TOKENIZER_JSON.exists():
    raise FileNotFoundError(f"❌ Файл {TOKENIZER_JSON} не найден.")

# Загружаем токенизатор и модель
print("🤖 Загружаю модель...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True).to("cpu")

# Истории диалогов
chat_histories = {}

# Логгирование
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# Команда /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Привет! Я бот на DialoGPT. Напиши мне что-нибудь!")

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
        attention_mask=torch.ones_like(bot_input_ids),  # Добавлено для устранения предупреждения
        max_length=200,
        pad_token_id=tokenizer.eos_token_id
    )

    chat_histories[chat_id] = chat_history_ids
    response_ids = chat_history_ids[:, bot_input_ids.shape[-1]:]
    bot_response = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    if bot_response.strip():
        await update.message.reply_text(bot_response)
    else:
        await update.message.reply_text("(Пустой ответ, попробуй ещё раз)")

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
