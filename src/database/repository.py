from datetime import datetime
from typing import Optional

from sqlalchemy.orm import Session
from src.database.models import User, UserStatus


class UserRepository:
    """
    Handles database operations for User model.
    """

    def __init__(self, db_session: Session):
        self.db = db_session

    def get_user_by_telegram_id(self, telegram_id: int) -> Optional[User]:
        """
        Retrieves a user by their Telegram ID.
        """
        return self.db.query(User).filter(User.telegram_id == telegram_id).first()

    def create_user(self, telegram_id: int, username: str) -> User:
        """
        Creates a new user with a 1-day trial.
        """
        db_user = User(
            telegram_id=telegram_id,
            username=username,
            status=UserStatus.TRIAL
        )
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)
        return db_user

    def add_or_update_user(self, telegram_id: int, username: str, status: UserStatus = UserStatus.TRIAL) -> User:
        """
        Creates a new user or updates an existing one with the given status.
        """
        user = self.get_user_by_telegram_id(telegram_id)
        if user:
            user.username = username
            user.status = status
        else:
            user = User(
                telegram_id=telegram_id,
                username=username,
                status=status
            )
            self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def has_active_access(self, telegram_id: int) -> bool:
        """
        Checks if a user has active access (active status or within trial period).
        """
        user = self.get_user_by_telegram_id(telegram_id)
        if not user:
            return False

        if user.status == UserStatus.ACTIVE:
            return True

        if user.status == UserStatus.TRIAL and user.trial_end_date > datetime.utcnow():
            return True

        return False

    def deactivate_expired_trials(self):
        """
        Deactivates all trial users whose trial period has expired.
        """
        expired_users = self.db.query(User).filter(
            User.status == UserStatus.TRIAL,
            User.trial_end_date <= datetime.utcnow()
        ).all()

        for user in expired_users:
            user.status = UserStatus.INACTIVE

        self.db.commit()
        return len(expired_users) 