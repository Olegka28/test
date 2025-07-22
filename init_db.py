from src.database.models import Base
from src.database.session import engine


def init_database():
    print("Initializing database...")
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully.")


if __name__ == "__main__":
    init_database() 