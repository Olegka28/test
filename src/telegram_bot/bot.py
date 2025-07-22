import os, logging
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, ConversationHandler
)
from .handlers import (
    start, choose_pair, choose_tf, refresh, back, settings_menu, settings_callback, main_menu, main_menu_callback, SETTINGS_MENU
)
from .fsm import SELECT_PAIR, SELECT_TIMEFRAME, SHOW_SIGNAL
from src.config import settings, get_logger
import asyncio

logger = get_logger("crypto-bot")
autoscan_task = None

def run_bot():
    token = settings.telegram.api_token
    if not token:
        logger.error("TELEGRAM__API_TOKEN is not set!")
        raise RuntimeError("Set TELEGRAM__API_TOKEN in .env file or configure it in settings")

    app = Application.builder().token(token).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start), CallbackQueryHandler(main_menu_callback, pattern="^main_")],
        states={
            SELECT_PAIR:      [CallbackQueryHandler(choose_pair, pattern="^pair_.*$|^main_menu$")],
            SELECT_TIMEFRAME: [CallbackQueryHandler(choose_tf,   pattern="^tf_.*$|^back$|^main_menu$")],
            SHOW_SIGNAL:      [CallbackQueryHandler(refresh,   pattern="^refresh_.*$|^back$|^main_menu$")],
            SETTINGS_MENU:    [CallbackQueryHandler(settings_callback, pattern="^toggle_autoscan$|^back$|^main_menu$")],
        },
        fallbacks=[CallbackQueryHandler(main_menu_callback, pattern="^main_menu$")],
    )
    app.add_handler(conv)

    logger.info("Bot started. Waiting for events...")

    async def autoscan_loop(application):
        from src.strategies.base_strategy import BaseStrategy
        from src.data.exchange_client import ExchangeClient
        from src.data.data_repository import DataRepository
        import traceback
        from src.telegram_bot.utils import main_menu_keyboard
        while True:
            await asyncio.sleep(5)
            if not application.bot_data.get('autoscan_enabled', False):
                await asyncio.sleep(30)
                continue
            try:
                pairs = list(settings.trading.pairs)
                tfs = list(settings.trading.timeframes.keys())
                chat_ids = settings.telegram.chat_ids
                exchange_client = ExchangeClient(settings.exchange)
                data_repo = DataRepository(exchange_client)
                strategy = BaseStrategy()
                logger.info(f"[AUTOSCAN] Start scan for pairs: {pairs} tfs: {tfs}")
                for pair in pairs:
                    for tf in tfs:
                        logger.info(f"[AUTOSCAN] Analyzing {pair} {tf}")
                        df = await data_repo.get_historical_data(pair, tf, limit=300)
                        if df is not None and not df.empty:
                            signal = strategy.analyze_model(pair, tf, df)
                            if signal:
                                signal.is_autoscan = True
                                msg = signal.to_telegram_message()
                                for chat_id in chat_ids:
                                    try:
                                        await application.bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown", reply_markup=main_menu_keyboard())
                                        logger.info(f"[AUTOSCAN] Signal sent to {chat_id} for {pair} {tf}")
                                    except Exception as e:
                                        logger.error(f"Ошибка отправки сигнала: {e}")
                await data_repo.close_connections()
                logger.info("[AUTOSCAN] Scan finished.")
            except Exception as e:
                logger.error(f"[AUTOSCAN ERROR] {e}\n{traceback.format_exc()}")
            await asyncio.sleep(1800)

    async def on_startup(application):
        global autoscan_task
        if autoscan_task is None:
            logger.info("Starting autoscan background task...")
            autoscan_task = asyncio.create_task(autoscan_loop(application))

    app.post_init = on_startup
    app.run_polling()

if __name__ == "__main__":
    run_bot() 