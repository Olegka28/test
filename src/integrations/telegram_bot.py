import asyncio
import logging
from typing import Dict, Optional
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes
from src.services.signal_service import SignalService
from src.config.settings import TELEGRAM_BOT_TOKEN, SUPPORTED_PAIRS, SUPPORTED_TIMEFRAMES, BUTTONS_PER_ROW

logger = logging.getLogger(__name__)

class TelegramBot:
    """Telegram бот для торговых сигналов"""
    
    def __init__(self):
        self.signal_service = SignalService()
        self.application = None
        
    async def start(self):
        """Запускает бота"""
        if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == 'your_bot_token_here':
            logger.error("Telegram bot token not configured!")
            return
            
        self.application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
        
        # Добавляем обработчики
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("overview", self.overview_command))
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        
        logger.info("Starting Telegram bot...")
        await self.application.run_polling()
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /start"""
        welcome_message = (
            "🚀 **Добро пожаловать в Crypto Signal Bot!**\n\n"
            "Я помогу вам получить торговые сигналы на основе ML моделей.\n\n"
            "**Доступные команды:**\n"
            "/start - Начать работу с ботом\n"
            "/help - Показать справку\n"
            "/overview - Обзор всех пар\n\n"
            "**Выберите пару для торговли:**"
        )
        
        keyboard = self._create_pair_keyboard()
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /help"""
        help_message = (
            "📚 **Справка по использованию бота**\n\n"
            "**Как использовать:**\n"
            "1. Выберите торговую пару\n"
            "2. Получите обзор по всем таймфреймам\n"
            "3. Выберите конкретный таймфрейм для детального сигнала\n\n"
            "**Поддерживаемые пары:**\n"
            "• SOL/USDT (Solana)\n"
            "• BTC/USDT (Bitcoin)\n"
            "• ETH/USDT (Ethereum)\n"
            "• HYPE/USDT (Hype)\n\n"
            "**Таймфреймы:**\n"
            "• 15m - 15 минут\n"
            "• 1h - 1 час\n"
            "• 4h - 4 часа\n"
            "• 1d - 1 день\n\n"
            "**Сигналы:**\n"
            "🟢 LONG - Рекомендуется покупка\n"
            "🔴 SHORT - Рекомендуется продажа\n"
            "⚪ NONE - Нет четкого сигнала\n\n"
            "**Важно:** Это не финансовый совет. Всегда проводите собственный анализ!"
        )
        
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
    async def overview_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик команды /overview"""
        await update.message.reply_text("📊 Выберите пару для получения обзора:")
        
        keyboard = self._create_pair_keyboard(action="overview")
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            "Выберите пару:",
            reply_markup=reply_markup
        )
    
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик нажатий на кнопки"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        user_id = query.from_user.id
        
        try:
            if data.startswith("pair_"):
                # Выбор пары
                symbol = data.replace("pair_", "")
                await self._handle_pair_selection(query, symbol)
                
            elif data.startswith("overview_"):
                # Обзор пары
                symbol = data.replace("overview_", "")
                await self._handle_overview_request(query, symbol)
                
            elif data.startswith("timeframe_"):
                # Выбор таймфрейма
                parts = data.replace("timeframe_", "").split("_")
                symbol = parts[0]
                timeframe = parts[1]
                await self._handle_timeframe_selection(query, symbol, timeframe)
                
            elif data == "back_to_pairs":
                # Возврат к выбору пары
                await self._show_pair_selection(query)
                
        except Exception as e:
            logger.error(f"Error handling button callback: {e}")
            await query.edit_message_text("❌ Произошла ошибка. Попробуйте еще раз.")
    
    async def _handle_pair_selection(self, query, symbol: str):
        """Обработка выбора пары"""
        # Получаем обзор для выбранной пары
        overview = self.signal_service.get_overview_for_pair(symbol)
        
        if overview.get('success', False):
            message = self.signal_service.format_overview_message(overview)
            
            # Создаем кнопки для выбора таймфрейма
            keyboard = self._create_timeframe_keyboard(symbol)
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        else:
            error_msg = f"❌ Ошибка получения данных для {symbol}"
            if 'error' in overview:
                error_msg += f"\n{overview['error']}"
            
            keyboard = [[InlineKeyboardButton("🔙 Назад к парам", callback_data="back_to_pairs")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                error_msg,
                reply_markup=reply_markup
            )
    
    async def _handle_overview_request(self, query, symbol: str):
        """Обработка запроса обзора"""
        await self._handle_pair_selection(query, symbol)
    
    async def _handle_timeframe_selection(self, query, symbol: str, timeframe: str):
        """Обработка выбора таймфрейма"""
        # Получаем сигнал для конкретного таймфрейма
        signal = self.signal_service.get_signal_for_pair(symbol, timeframe)
        
        if signal.get('success', False):
            message = self.signal_service.format_signal_message(signal)
            
            # Добавляем кнопки навигации
            keyboard = [
                [InlineKeyboardButton("🔙 Назад к обзору", callback_data=f"pair_{symbol}")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_pairs")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                message,
                reply_markup=reply_markup,
                parse_mode='Markdown'
            )
        else:
            error_msg = f"❌ Ошибка получения сигнала для {symbol} {timeframe}"
            if 'error' in signal:
                error_msg += f"\n{signal['error']}"
            
            keyboard = [
                [InlineKeyboardButton("🔙 Назад к обзору", callback_data=f"pair_{symbol}")],
                [InlineKeyboardButton("🏠 Главное меню", callback_data="back_to_pairs")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                error_msg,
                reply_markup=reply_markup
            )
    
    async def _show_pair_selection(self, query):
        """Показывает выбор пары"""
        welcome_message = (
            "🚀 **Crypto Signal Bot**\n\n"
            "Выберите пару для торговли:"
        )
        
        keyboard = self._create_pair_keyboard()
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            welcome_message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    
    def _create_pair_keyboard(self, action: str = "pair") -> list:
        """Создает клавиатуру для выбора пары"""
        keyboard = []
        row = []
        
        for i, pair in enumerate(SUPPORTED_PAIRS):
            # Эмодзи для пар
            pair_emoji = {
                'SOL/USDT': '🟣',
                'BTC/USDT': '🟡',
                'ETH/USDT': '🔵',
                'HYPE/USDT': '🟢'
            }
            
            emoji = pair_emoji.get(pair, '💰')
            button_text = f"{emoji} {pair}"
            callback_data = f"{action}_{pair}"
            
            row.append(InlineKeyboardButton(button_text, callback_data=callback_data))
            
            if len(row) == BUTTONS_PER_ROW:
                keyboard.append(row)
                row = []
        
        if row:  # Добавляем оставшиеся кнопки
            keyboard.append(row)
        
        return keyboard
    
    def _create_timeframe_keyboard(self, symbol: str) -> list:
        """Создает клавиатуру для выбора таймфрейма"""
        keyboard = []
        row = []
        
        for i, timeframe in enumerate(SUPPORTED_TIMEFRAMES):
            # Эмодзи для таймфреймов
            timeframe_emoji = {
                '15m': '⏱️',
                '1h': '🕐',
                '4h': '🕓',
                '1d': '📅'
            }
            
            emoji = timeframe_emoji.get(timeframe, '⏰')
            button_text = f"{emoji} {timeframe}"
            callback_data = f"timeframe_{symbol}_{timeframe}"
            
            row.append(InlineKeyboardButton(button_text, callback_data=callback_data))
            
            if len(row) == 2:  # 2 кнопки в ряду для таймфреймов
                keyboard.append(row)
                row = []
        
        if row:  # Добавляем оставшиеся кнопки
            keyboard.append(row)
        
        # Добавляем кнопку возврата
        keyboard.append([InlineKeyboardButton("🔙 Назад к парам", callback_data="back_to_pairs")])
        
        return keyboard 