import ccxt

class BybitConnector:
    """
    Connector for Bybit exchange using ccxt to fetch OHLCV data.
    """
    def __init__(self, api_key: str = None, secret: str = None, rate_limit: bool = True):
        params = {}
        if api_key and secret:
            params['apiKey'] = api_key
            params['secret'] = secret
        params['enableRateLimit'] = rate_limit
        self.exchange = ccxt.bybit(params)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500):
        """
        Fetches OHLCV data from Bybit.

        Args:
            symbol (str): Trading pair symbol, e.g. 'BTC/USDT'.
            timeframe (str): Timeframe string understood by ccxt, e.g. '15m', '1h'.
            limit (int): Number of candles to fetch.

        Returns:
            List of lists: [timestamp, open, high, low, close, volume].
        """
        try:
            # Bybit requires unified symbols like 'BTC/USDT'
            return self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        except Exception:
            raise
