import os
from typing import Dict, List

# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7884030672:AAFy_msJRlSms3Dm4fkVbJFrjjgHKMx_r5w')

# Trading Configuration
SUPPORTED_PAIRS = ['SOL/USDT', 'BTC/USDT', 'ETH/USDT']
SUPPORTED_TIMEFRAMES = ['15m', '1h', '4h', '1d']

# Bybit API Configuration
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
BYBIT_SECRET_KEY = os.getenv('BYBIT_SECRET_KEY', '')
BYBIT_TESTNET = os.getenv('BYBIT_TESTNET', 'false').lower() == 'true'

# Model Configuration
MODEL_DIR = 'models'
MODEL_PREFIX = 'xgb'

# Feature Configuration
FEATURE_WINDOW = 20
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2

# Signal Configuration
SIGNAL_THRESHOLD = 0.6
TP_RATIO = 0.02  # 2% take profit
SL_RATIO = 0.01  # 1% stop loss

# Data Configuration
DATA_DIR = 'data/ohlcv'
CANDLE_LIMIT = 500

# Button Configuration
BUTTONS_PER_ROW = 2 