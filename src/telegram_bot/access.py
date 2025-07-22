from datetime import datetime, timedelta
from typing import Union
from src.database.session import SessionLocal
from src.database.repository import UserRepository, UserStatus

ADMIN_NICK = "@your_admin"  # TODO: замените на ник администратора


def ensure_user(telegram_id: int, username: Union[str, None]) -> tuple[str, Union[datetime, None]]:
    """
    Проверяет пользователя в базе.
    Возвращает tuple (status, trial_end).
    status: "active" | "trial" | "blocked".
    """
    db = SessionLocal()
    repo = UserRepository(db)
    user = repo.get_user_by_telegram_id(telegram_id)

    if user is None:
        # Новый пользователь — создаём и даём день trial
        trial_end = datetime.utcnow() + timedelta(days=1)
        repo.create_user(telegram_id, username or "unknown")
        db.close()
        return "trial", trial_end

    # Существующий пользователь — просто возвращаем его статус и дату окончания trial
    if user.status == UserStatus.BLOCKED:
        db.close()
        return "blocked", None

    if user.status == UserStatus.TRIAL:
        db.close()
        return "trial", user.trial_end_date

    db.close()
    return "active", None


def block_message() -> str:
    return (
        "🚫 Доступ обмежено.\n"
        f"Щоб отримати доступ — зверніться до адміністратора: {ADMIN_NICK}"
    ) 