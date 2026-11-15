"""
Telegram front-end for budget.py.

Run this instead of budget.py to chat with the budget assistant through Telegram
instead of the terminal. It reuses the exact same logic (classify, /balance, /report, ...).

Install:
  pip install python-telegram-bot

Setup:
  1. Put your bot token from @BotFather in .env as TELEGRAM_BOT_TOKEN.
  2. Run this script, then send /start to your bot — it replies with your chat ID.
  3. Put that chat ID in .env as TELEGRAM_CHAT_ID and restart. From then on, the bot
     only responds to that chat, so no one else who finds the bot can use it.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from budget import app as budget_app

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
ALLOWED_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def is_allowed(chat_id: int) -> bool:
    return not ALLOWED_CHAT_ID or str(chat_id) == ALLOWED_CHAT_ID


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not is_allowed(chat_id):
        return
    msg = f"👋 Hi! Your chat ID is {chat_id}."
    if not ALLOWED_CHAT_ID:
        msg += f"\nAdd TELEGRAM_CHAT_ID={chat_id} to .env to restrict this bot to just you."
    await update.message.reply_text(msg)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not is_allowed(chat_id):
        return
    result = budget_app.invoke({"user_input": update.message.text.strip()})
    await update.message.reply_text(result["reply"])


def main() -> None:
    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT, handle_message))
    print("Telegram bot is running. Press Ctrl+C to stop.")
    application.run_polling()


if __name__ == "__main__":
    main()
