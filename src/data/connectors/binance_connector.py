import ccxt

class BinanceConnector:
    """
    Connector for Binance exchange using ccxt to fetch OHLCV data.
    """
    def __init__(self, api_key: str = None, secret: str = None, rate_limit: bool = True):
        params = {}
        if api_key and secret:
            params['apiKey'] = api_key
            params['secret'] = secret
        params['enableRateLimit'] = rate_limit
        self.exchange = ccxt.binance(params)

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500):
        """
        Fetches OHLCV data from Binance.

        Args:
            symbol (str): Trading pair symbol, e.g. 'BTC/USDT'.
            timeframe (str): Timeframe string understood by ccxt, e.g. '15m', '1h'.
            limit (int): Number of candles to fetch.

        Returns:
            List of lists: [timestamp, open, high, low, close, volume].
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            # handle rate limits or connectivity issues
            raise
