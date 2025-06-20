import telebot
import os
from ocr import recognize_full_text  # импорт функции распознования изображения
from detect.inference import load_craft_model, detect_text_region # импорт функции определения поля текста
from uuid import uuid4

# Токен бота
TOKEN = "8008457466:AAEXYRD-aSmHcwv0FEEbU7OdGEkKMpwF8uI"
bot = telebot.TeleBot(TOKEN)

# Установка меню команд
commands = [
    telebot.types.BotCommand("start", "Что умеет бот"),
    telebot.types.BotCommand("recognize", "Распознать текст на фото")
]
bot.set_my_commands(commands)

# Обработчик команды /start
@bot.message_handler(commands=['start'])
def send_massage(message):
    welcome_text = "👋 Привет! Я AI Vision Bot, созданный для распознавания текста на изображениях."
    bot.send_message(message.chat.id, welcome_text)

# Обработчик команды /recognize
@bot.message_handler(commands=['recognize'])
def request_photo(message):
    bot.send_message(message.chat.id, "📸 Отправьте фото с текстом, и я его обработаю!")

# Обработчик изображений
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    bot.send_message(message.chat.id, "🔄 Обрабатываю изображение...")

    # Скачиваем изображение
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    temp_filename = f"temp_{uuid4().hex}.png"
    with open(temp_filename, 'wb') as new_file:
        new_file.write(downloaded_file)
    craft_model = load_craft_model()
    cropped_path = detect_text_region(temp_filename, craft_model)


    try:
        # Распознаём текст через модель
        result = recognize_full_text(cropped_path)
 
        if result:
            bot.send_message(message.chat.id, "📜 Распознанный текст:")
            bot.send_message(message.chat.id, result)
        else:
            bot.send_message(message.chat.id, "❌ Не удалось распознать текст.")
    except Exception as e:
        bot.send_message(message.chat.id, f"⚠️ Ошибка при обработке изображения: {e}")
    finally:
        os.remove(temp_filename)
        os.remove(cropped_path)


# Запуск бота
bot.polling(none_stop=True)
