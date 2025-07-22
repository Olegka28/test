from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config import settings

# The engine is the entry point to the database.
# It's configured with the database URL from settings.
engine = create_engine(
    settings.database.url,
    connect_args={"check_same_thread": False},  # Required for SQLite
    echo=settings.database.echo
)

# The SessionLocal class is a factory for new Session objects.
# A session is used to interact with the database.
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


def get_db():
    """
    Dependency function to get a database session.
    It ensures that the database session is always closed after the request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 