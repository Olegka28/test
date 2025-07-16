import os
import pandas as pd
from typing import Optional

from src.data.connectors.binance_connector import BinanceConnector
from src.data.connectors.bybit_connector import BybitConnector


class OHLCVRepository:
    """
    Repository for OHLCV data with local CSV caching.
    """
    def __init__(self, data_dir: str, exchange: str = 'binance', api_key: Optional[str] = None, secret: Optional[str] = None):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        # Initialize appropriate connector
        if exchange.lower() == 'binance':
            self.connector = BinanceConnector(api_key, secret)
        elif exchange.lower() == 'bybit':
            self.connector = BybitConnector(api_key, secret)
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")

    def _cache_path(self, symbol: str, timeframe: str) -> str:
        filename = f"{symbol.replace('/', '_')}_{timeframe}.csv"
        return os.path.join(self.data_dir, filename)

    def load(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        """
        Load OHLCV DataFrame from cache or fetch and cache.

        Args:
            symbol (str): Trading pair symbol, e.g. 'BTC/USDT'
            timeframe (str): e.g. '15m'
            limit (int): Number of candles to fetch if not cached
        Returns:
            pd.DataFrame with columns ['open','high','low','close','volume'], indexed by datetime
        """
        path = self._cache_path(symbol, timeframe)
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['datetime'], index_col='datetime')
        else:
            raw = self.connector.fetch_ohlcv(symbol, timeframe, limit)
            df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            df.to_csv(path)
        return df
