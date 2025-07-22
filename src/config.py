import logging
from typing import List, Dict, Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Новый универсальный логгер ---
def get_logger(name: str = "crypto-bot") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


class TelegramSettings(BaseModel):
    """Telegram bot settings"""
    api_token: str = Field("", description="Telegram Bot API token")
    chat_ids: List[int] = Field(default=[], description="List of recipient chat IDs")


class ExchangeSettings(BaseModel):
    """Exchange API settings"""
    name: str = Field("binance", description="Default exchange name")
    api_key: str = Field("", description="Exchange API key")
    api_secret: str = Field("", description="Exchange API secret")
    default_type: str = Field("future", description="Default market type (e.g., spot, future)")
    sandbox_mode: bool = Field(False, description="Enable sandbox mode for paper trading")


class TimeframeConfig(BaseModel):
    """Configuration for a single timeframe"""
    prediction_horizon_hours: int = Field(..., description="Prediction horizon in hours")
    data_points: int = Field(..., description="Number of data points to use for feature calculation")
    price_change_threshold: float = Field(..., description="Price change threshold to define a target")
    feature_config: Dict[str, Any] = Field(..., description="Configuration for feature generation")

    @property
    def prediction_candles(self) -> int:
        """Calculates prediction horizon in candles based on timeframe string"""
        if "m" in self.timeframe:
            minutes = int(self.timeframe.replace("m", ""))
            return int((self.prediction_horizon_hours * 60) / minutes)
        if "h" in self.timeframe:
            hours = int(self.timeframe.replace("h", ""))
            return int(self.prediction_horizon_hours / hours)
        if "d" in self.timeframe:
            days = int(self.timeframe.replace("d", ""))
            return int((self.prediction_horizon_hours / 24) / days)
        return 0 # Should not happen with valid timeframes

    # This field will be populated dynamically
    timeframe: str = "15m"


class TradingSettings(BaseModel):
    """Trading strategy settings"""
    pairs: List[str] = Field(
        default=["BTC/USDT", "ETH/USDT", "ADA/USDT", "SOL/USDT", "XRP/USDT"],
        description="List of trading pairs"
    )
    timeframes: Dict[str, TimeframeConfig] = Field(
        default={
            "15m": TimeframeConfig(
                prediction_horizon_hours=6, data_points=300, price_change_threshold=0.01,
                feature_config={
                    "base": ["ema_5", "ema_9", "ema_21", "rsi_7", "macd", "obv", "volume_change", "cdl_all"],
                    "context": {"1h": ["ema_50_trend", "rsi_14", "adx_14", "supertrend_7_3"]}
                }
            ),
            "1h": TimeframeConfig(
                prediction_horizon_hours=12, data_points=300, price_change_threshold=0.015,
                feature_config={
                    "base": ["ema_21", "ema_50", "rsi_14", "atr_14", "volume_spike"],
                    "context": {"4h": ["ema_200_trend"], "1d": ["bbands_20_2_width"]}
                }
            ),
            "4h": TimeframeConfig(
                prediction_horizon_hours=48, data_points=300, price_change_threshold=0.03,
                feature_config={
                    "base": ["supertrend_10_3", "macd", "adx_14", "momentum_10", "time_features"],
                    "context": {"1d": ["ema_100_trend", "market_regime"]}
                }
            ),
            "1d": TimeframeConfig(
                prediction_horizon_hours=168, data_points=300, price_change_threshold=0.05,
                feature_config={
                    "base": ["ema_50", "ema_200", "macd_hist", "volatility_30", "seasonality", "cdl_all"],
                    "context": {} # No context for daily
                }
            ),
        },
        description="Timeframe specific configurations"
    )
    # Add timeframe to each config object
    def __init__(self, **data: Any):
        super().__init__(**data)
        for tf, config in self.timeframes.items():
            config.timeframe = tf


class ModelSettings(BaseModel):
    """Machine learning model settings"""
    directory: str = Field("models/", description="Directory to save/load models")
    target_column: str = Field("target", description="Name of the target variable for prediction")


class DatabaseSettings(BaseModel):
    """Database connection settings"""
    url: str = Field("sqlite:///./crypto_bot.db", description="Database connection URL")
    echo: bool = Field(False, description="Enable SQLAlchemy echo logs")


class AppSettings(BaseModel):
    """General application settings"""
    log_level: str = Field("INFO", description="Logging level (e.g., DEBUG, INFO, WARNING)")
    log_path: str = Field("logs/app.log", description="Path to log file")


class Settings(BaseSettings):
    """Main settings object, loading from .env file"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields from .env file
    )

    app: AppSettings = AppSettings()
    telegram: TelegramSettings = TelegramSettings()
    exchange: ExchangeSettings = ExchangeSettings()
    trading: TradingSettings = TradingSettings()
    models: ModelSettings = ModelSettings()
    database: DatabaseSettings = DatabaseSettings()


def setup_logging(settings: AppSettings):
    """Configures the logging for the application."""
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logging.basicConfig(level=settings.log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    # You might want to use Loguru here as it's in requirements.txt
    # from loguru import logger
    # logger.add(settings.log_path, rotation="10 MB", level=settings.log_level, format=log_format)


# Instantiate settings
settings = Settings()

# Setup logging based on the loaded settings
setup_logging(settings.app)

# Example of how to access settings:
# from src.config import settings
# print(settings.telegram.api_token)
# print(settings.trading.pairs) 