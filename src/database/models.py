import datetime
from sqlalchemy import Column, Integer, String, DateTime, Enum
from sqlalchemy.orm import declarative_base
import enum

Base = declarative_base()


class UserStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRIAL = "trial"
    BLOCKED = "blocked"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(Integer, unique=True, index=True, nullable=False)
    username = Column(String, unique=True)
    status = Column(Enum(UserStatus), default=UserStatus.TRIAL, nullable=False)
    trial_start_date = Column(DateTime, default=datetime.datetime.utcnow)
    trial_end_date = Column(DateTime, default=lambda: datetime.datetime.utcnow() + datetime.timedelta(days=1))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

    def __repr__(self):
        return (f"<User(id={self.id}, telegram_id={self.telegram_id}, "
                f"username='{self.username}', status='{self.status}', "
                f"trial_end_date={self.trial_end_date})>") 