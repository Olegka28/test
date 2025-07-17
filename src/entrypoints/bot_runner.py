import asyncio
import logging
import sys
import os
from pathlib import Path

# Добавляем корневую директорию в путь
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))

from src.integrations.telegram_bot import TelegramBot
from src.config.settings import TELEGRAM_BOT_TOKEN

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Главная функция для запуска бота"""
    try:
        logger.info("Starting Crypto Signal Bot...")
        
        # Проверяем токен
        if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == 'your_bot_token_here':
            logger.error("❌ Telegram bot token not configured!")
            logger.error("Please set TELEGRAM_BOT_TOKEN environment variable or update settings.py")
            return
        
        # Создаем и запускаем бота
        bot = TelegramBot()
        await bot.start()
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Error running bot: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())