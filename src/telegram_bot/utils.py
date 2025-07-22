from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from typing import List


def main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("📈 Получить сигнал", callback_data="main_get_signal")],
        [InlineKeyboardButton("⚙️ Настройки", callback_data="main_settings")],
        [InlineKeyboardButton("ℹ️ О боте", callback_data="main_about")],
    ])


def pairs_keyboard(pairs: List[str]) -> InlineKeyboardMarkup:
    rows, row = [], []
    for i, pair in enumerate(pairs, 1):
        row.append(InlineKeyboardButton(pair, callback_data=f"pair_{pair}"))
        if i % 4 == 0:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    # Добавляем кнопку "🏠 Главное меню"
    rows.append([InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")])
    return InlineKeyboardMarkup(rows)


def tf_keyboard(tfs: List[str]) -> InlineKeyboardMarkup:
    rows = [[InlineKeyboardButton(f"🕒 {tf}", callback_data=f"tf_{tf}")] for tf in tfs]
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="back"), InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")])
    return InlineKeyboardMarkup(rows)


def signal_keyboard(pair: str, tf: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🔄 Оновити", callback_data=f"refresh_{pair}_{tf}")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back"), InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ])


def settings_keyboard(autoscan_enabled: bool) -> InlineKeyboardMarkup:
    status = "🟢 ВКЛ" if autoscan_enabled else "🔴 ВЫКЛ"
    return InlineKeyboardMarkup([
        [InlineKeyboardButton(f"Автосканирование: {status}", callback_data="toggle_autoscan")],
        [InlineKeyboardButton("⬅️ Назад", callback_data="back"), InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ])


def about_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("🏠 Главное меню", callback_data="main_menu")]
    ]) 