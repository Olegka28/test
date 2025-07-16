import os


class Settings:
    """
    Simple settings loader using environment variables without external dependencies.
    """
    # Exchange API credentials
    BINANCE_API_KEY: str = os.getenv('BINANCE_API_KEY', '')
    BINANCE_API_SECRET: str = os.getenv('BINANCE_API_SECRET', '')
    BYBIT_API_KEY: str = os.getenv('BYBIT_API_KEY', '')
    BYBIT_API_SECRET: str = os.getenv('BYBIT_API_SECRET', '')

    # Exchange selection
    EXCHANGE: str = os.getenv('EXCHANGE', 'binance')

    # Repository configuration
    DATA_DIR: str = os.getenv('DATA_DIR', 'data/ohlcv')

    # Telegram bot settings
    TELEGRAM_TOKEN: str = os.getenv('TELEGRAM_TOKEN', '')
    TELEGRAM_WEBHOOK_URL: str = os.getenv('TELEGRAM_WEBHOOK_URL', '')

    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL: str = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

    # ML model paths
    MODEL_DIR: str = os.getenv('MODEL_DIR', 'models')

    # Signal logic thresholds
    CONFIDENCE_THRESHOLD: float = float(os.getenv('CONFIDENCE_THRESHOLD', '0.65'))
    ADX_THRESHOLD: float = float(os.getenv('ADX_THRESHOLD', '20.0'))


# Instantiate settings
settings = Settings()