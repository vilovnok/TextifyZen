import json
import time
import logging
from pathlib import Path
from aiogram import Bot, Dispatcher, executor, types
from simplifier import Simplifier

logging.basicConfig(level=logging.INFO)

config_path = Path(__file__).parent / "config.json"
with open(config_path, "r") as file:
    config = json.load(file)

answers_path = Path(__file__).parent / "bot_answers.json"
with open(answers_path, "r", encoding='utf-8') as file:
    answers = json.load(file)

TOKEN = config['bot_token']
bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot)
logging.info(f'Model loaded successfully! {time.asctime()}')

simplifier = Simplifier()
@dp.message_handler(commands=["start"])
async def start_handler(message: types.Message):
    user_id = message.from_user.id
    user_name = message.from_user.first_name
    user_full_name = message.from_user.full_name    
    # logging.info(f'{user_id} {user_full_name} {time.asctime()}')
    await message.answer(answers['start'])

@dp.message_handler(content_types=['text'])
async def simplify(message: types.Message):
    text = message.text
    user_id = message.from_user.id
    # logging.info(f'User {user_id} sent text: "{text}" {time.asctime()}')
    reply = await message.reply(answers['loading'])
    simplified_text = simplifier.simplify(text)
    # logging.info(f'Text simplified: "{simplified_text}" {time.asctime()}')
    await reply.edit_text(simplified_text)

if __name__ == "__main__":
    executor.start_polling(dp)