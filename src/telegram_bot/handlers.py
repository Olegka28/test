from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ConversationHandler
from .fsm import SELECT_PAIR, SHOW_ANALYSIS, SELECT_TIMEFRAME, SHOW_SIGNAL
from .utils import pairs_keyboard, tf_keyboard, signal_keyboard, main_menu_keyboard, settings_keyboard, about_keyboard
from .access import ensure_user, block_message
from src.config import settings

# Импортируем необходимые классы для стратегии
from src.strategies.base_strategy import BaseStrategy
from src.data.exchange_client import ExchangeClient
from src.data.data_repository import DataRepository
import asyncio

PAIRS = list(settings.trading.pairs)
TFS = list(settings.trading.timeframes.keys())
ADMIN_NICK = "@Olegka28" 
ADMIN_IDS = set(settings.telegram.chat_ids)  # Можно расширить список админов

# --- Новый раздел: Настройки ---
SETTINGS_MENU, = range(100, 101)

# --- Главное меню ---
async def main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text("🏠 *Головне меню*", parse_mode="Markdown", reply_markup=main_menu_keyboard())
    elif update.callback_query:
        await update.callback_query.edit_message_text("🏠 *Головне меню*", parse_mode="Markdown", reply_markup=main_menu_keyboard())
    return ConversationHandler.END

# --- Обработка кнопок главного меню ---
async def main_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data = query.data
    if data == "main_get_signal":
        await query.edit_message_text("📌 Виберіть криптовалюту:", reply_markup=pairs_keyboard(PAIRS))
        return SELECT_PAIR
    elif data == "main_settings":
        return await settings_menu(update, context)
    elif data == "main_about":
        await query.edit_message_text(
            "ℹ️ *Crypto Signal Bot*\n\nБот для генерации торговых сигналов по криптовалюте.\n\nРозробка: @your_admin",
            parse_mode="Markdown",
            reply_markup=about_keyboard()
        )
        return ConversationHandler.END
    elif data == "main_menu":
        return await main_menu(update, context)
    return ConversationHandler.END

# --- Остальные обработчики (start, choose_pair, choose_tf, refresh, back, settings_menu, settings_callback) ---
# Везде добавляем обработку кнопки main_menu и возврат в главное меню

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    uname = update.effective_user.username
    status, trial_end = ensure_user(uid, uname)

    if status == "blocked":
        await update.message.reply_text(block_message())
        return ConversationHandler.END

    if status == "trial":
        await update.message.reply_text(
            f"🎁 Вітаємо! Тестовий доступ до {trial_end.strftime('%Y-%m-%d %H:%M UTC')}."
        )

    return await main_menu(update, context)

async def choose_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "main_menu":
        return await main_menu(update, context)
    pair = query.data.split("_")[1]
    context.user_data["pair"] = pair

    # Реальный анализ тренда по всем таймфреймам
    exchange_client = ExchangeClient(settings.exchange)
    data_repo = DataRepository(exchange_client)
    trend_summary = []
    try:
        for tf in TFS:
            df = await data_repo.get_historical_data(pair, tf, limit=100)
            trend_emoji = "⚪️"
            trend_text = "Нейтрально"
            if df is not None and not df.empty:
                # Простой тренд: close > ema_50 — вверх, < ema_50 — вниз
                if "ema_50" in df.columns:
                    ema = df["ema_50"].iloc[-1]
                else:
                    import pandas_ta as ta
                    ema = ta.ema(df["close"], length=50).iloc[-1]
                close = df["close"].iloc[-1]
                if close > ema:
                    trend_emoji = "🟢"
                    trend_text = "Вгору"
                elif close < ema:
                    trend_emoji = "🔴"
                    trend_text = "Вниз"
            trend_summary.append(f"{tf}: {trend_emoji} {trend_text}")
        await data_repo.close_connections()
    except Exception as e:
        trend_summary.append(f"⚠️ Ошибка анализа: {e}")

    text = f"📊 *Аналіз {pair}*\n" + "\n".join(trend_summary) + "\n\n🕒 Оберіть таймфрейм:"
    await query.edit_message_text(text, parse_mode="Markdown", reply_markup=tf_keyboard(TFS))
    return SELECT_TIMEFRAME

async def choose_tf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "main_menu":
        return await main_menu(update, context)
    if query.data == "back":
        pair = context.user_data.get("pair")
        await query.edit_message_text("📌 Виберіть криптовалюту:", reply_markup=pairs_keyboard(PAIRS))
        return SELECT_PAIR
    tf = query.data.split("_")[1]
    pair = context.user_data["pair"]

    # --- Реальная интеграция стратегии ---
    msg = "⚠️ Не удалось получить сигнал."
    try:
        # 1. Инициализация клиентов
        exchange_client = ExchangeClient(settings.exchange)
        data_repo = DataRepository(exchange_client)
        strategy = BaseStrategy()
        # 2. Получаем исторические данные (например, 300 свечей)
        df = await data_repo.get_historical_data(pair, tf, limit=300)
        if df is not None and not df.empty:
            # 3. Анализируем сигнал
            signal = strategy.analyze_model(pair, tf, df)
            if signal:
                msg = signal.to_telegram_message()
            else:
                msg = "❌ Нет сигнала для этой пары и таймфрейма."
        else:
            msg = "❌ Не удалось получить данные с биржи."
        await data_repo.close_connections()
    except Exception as e:
        msg = f"⚠️ Ошибка при генерации сигнала: {e}"

    await query.edit_message_text(msg, parse_mode="Markdown", reply_markup=signal_keyboard(pair, tf))
    return SHOW_SIGNAL

async def refresh(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "main_menu":
        return await main_menu(update, context)
    _, pair, tf = query.data.split("_")
    # Просто вызываем choose_tf заново
    context.user_data["pair"] = pair
    fake_update = Update(update.update_id, callback_query=query)
    await choose_tf(fake_update, context)
    return SHOW_SIGNAL

async def back(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    if query.data == "main_menu":
        return await main_menu(update, context)
    await query.edit_message_text("📌 Виберіть криптовалюту:", reply_markup=pairs_keyboard(PAIRS))
    return SELECT_PAIR

# --- Настройки и about ---
async def settings_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id if update.message else update.callback_query.from_user.id
    if uid not in ADMIN_IDS:
        if update.message:
            await update.message.reply_text("⛔️ Тільки адміністратор може змінювати налаштування.")
        else:
            await update.callback_query.edit_message_text("⛔️ Тільки адміністратор може змінювати налаштування.")
        return ConversationHandler.END
    autoscan_enabled = context.application.bot_data.get('autoscan_enabled', False)
    if update.message:
        await update.message.reply_text(
            "⚙️ *Налаштування*", parse_mode="Markdown",
            reply_markup=settings_keyboard(autoscan_enabled)
        )
    else:
        await update.callback_query.edit_message_text(
            "⚙️ *Налаштування*", parse_mode="Markdown",
            reply_markup=settings_keyboard(autoscan_enabled)
        )
    return SETTINGS_MENU

async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    uid = query.from_user.id
    if uid not in ADMIN_IDS:
        await query.edit_message_text("⛔️ Тільки адміністратор може змінювати налаштування.")
        return ConversationHandler.END
    data = query.data
    if data == "toggle_autoscan":
        autoscan_enabled = context.application.bot_data.get('autoscan_enabled', False)
        context.application.bot_data['autoscan_enabled'] = not autoscan_enabled
        autoscan_enabled = not autoscan_enabled
        await query.edit_message_text(
            f"⚙️ *Налаштування*\nАвтосканування {'🟢 ВКЛ' if autoscan_enabled else '🔴 ВЫКЛ'}",
            parse_mode="Markdown",
            reply_markup=settings_keyboard(autoscan_enabled)
        )
        return SETTINGS_MENU
    elif data == "back":
        await query.edit_message_text("📌 Виберіть криптовалюту:", reply_markup=pairs_keyboard(PAIRS))
        return SELECT_PAIR
    elif data == "main_menu":
        return await main_menu(update, context)
    return SETTINGS_MENU 