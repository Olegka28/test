import ccxt.async_support as ccxt
import pandas as pd
from typing import Optional
from datetime import datetime
import asyncio

from src.config import ExchangeSettings


class ExchangeClient:
    """
    A client to connect to crypto exchanges and fetch data using CCXT.
    """

    def __init__(self, settings: ExchangeSettings):
        self.settings = settings
        self.exchange = self._init_exchange()

    def _init_exchange(self):
        """Initializes the CCXT exchange instance."""
        exchange_class = getattr(ccxt, self.settings.name)
        config = {
            "apiKey": self.settings.api_key,
            "secret": self.settings.api_secret,
            "options": {
                "defaultType": self.settings.default_type,
            },
        }
        if self.settings.sandbox_mode:
            # Some exchanges might need this to be set differently
            self.exchange.set_sandbox_mode(True)
        
        return exchange_class(config)

    async def get_latest_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> Optional[pd.DataFrame]:
        """
        Fetches the latest OHLCV data for a given symbol and timeframe.
        """
        try:
            await self.exchange.load_markets()
            if symbol not in self.exchange.markets:
                print(f"Symbol {symbol} not found on {self.settings.name}")
                return None

            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            return df

        except ccxt.NetworkError as e:
            print(f"Network error while fetching {symbol}: {e}")
        except ccxt.ExchangeError as e:
            print(f"Exchange error while fetching {symbol}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while fetching {symbol}: {e}")
        
        return None

    async def get_historical_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[pd.DataFrame]:
        """
        Fetches historical OHLCV data within a date range using pagination.
        """
        await self.exchange.load_markets()
        if symbol not in self.exchange.markets:
            print(f"Symbol {symbol} not found on {self.settings.name}")
            return None

        since_timestamp = int(start_date.timestamp() * 1000)
        to_timestamp = int(end_date.timestamp() * 1000)
        
        all_ohlcv = []
        limit = 1000  # Use a safe limit

        while since_timestamp < to_timestamp:
            try:
                ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=limit)
                if not ohlcv:
                    break
                
                last_candle_ts = ohlcv[-1][0]
                all_ohlcv.extend(ohlcv)
                
                if last_candle_ts >= since_timestamp:
                    since_timestamp = last_candle_ts + 1
                else: # Safeguard against exchanges returning same last candle
                    timeframe_duration_ms = self.exchange.parse_timeframe(timeframe) * 1000
                    since_timestamp += timeframe_duration_ms

                if since_timestamp > to_timestamp:
                    break
                    
                await asyncio.sleep(self.exchange.rateLimit / 1000)

            except Exception as e:
                print(f"An error occurred while fetching historical data for {symbol}: {e}")
                return None
        
        if not all_ohlcv:
            return pd.DataFrame()
            
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        df.drop_duplicates(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)
        return df

    async def close(self):
        """Closes the exchange connection."""
        if self.exchange:
            await self.exchange.close()

# Example usage (can be run for testing)
# if __name__ == '__main__':
#     import asyncio
#     from src.config import settings

#     async def main():
#         client = ExchangeClient(settings.exchange)
#         btc_df = await client.get_ohlcv('BTC/USDT', '1h', 100)
#         if btc_df is not None:
#             print("Successfully fetched BTC/USDT data:")
#             print(btc_df.head())
#         await client.close()

#     asyncio.run(main()) 