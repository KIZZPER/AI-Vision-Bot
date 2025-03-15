import telebot



#Токен бота
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
    welcome_text = ("👋 Привет! Я AI Vision Bot, созданный для распознавания текста на изображениях.")
    bot.send_message(message.chat.id, welcome_text)

# Обработчик команды /recognize
@bot.message_handler(commands=['recognize'])
def request_photo(message):
    bot.send_message(message.chat.id, "📸 Отправьте фото с текстом, и я его обработаю!")

# Обработчик изображений
@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    bot.send_message(message.chat.id, "🔄 Обрабатываю изображение... (пока заглушка)")
    # Здесь позже добавится код обработки изображения
    bot.send_message(message.chat.id, "✅ Готово! Отправьте ещё одно фото, если хотите.")









# Запуск бота
bot.polling(none_stop=True)