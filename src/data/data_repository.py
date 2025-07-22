from typing import Optional
import pandas as pd
from datetime import datetime

from src.config import settings
from src.data.exchange_client import ExchangeClient


class DataRepository:
    """
    Acts as a source of truth for market data.
    It can fetch data from an exchange and could be extended
    to fetch from a local cache (e.g., a database or files).
    """

    def __init__(self, exchange_client: ExchangeClient):
        self.exchange_client = exchange_client

    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        limit: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetches historical OHLCV data either by last N candles (limit)
        or by a date range.
        """
        if limit and not (start_date or end_date):
            print(f"Fetching last {limit} candles for {symbol} on {timeframe}...")
            data = await self.exchange_client.get_latest_ohlcv(
                symbol=symbol, timeframe=timeframe, limit=limit
            )
        elif start_date and end_date and not limit:
            print(f"Fetching data for {symbol} on {timeframe} from {start_date} to {end_date}...")
            data = await self.exchange_client.get_historical_ohlcv(
                symbol=symbol, timeframe=timeframe, start_date=start_date, end_date=end_date
            )
        else:
            raise ValueError("Provide either 'limit' or both 'start_date' and 'end_date'.")

        if data is not None and not data.empty:
            print(f"Successfully fetched {len(data)} data points for {symbol}.")
        else:
            print(f"Failed to fetch data for {symbol} or no data in range.")

        return data

    async def close_connections(self):
        """Closes underlying connections, like the exchange client."""
        await self.exchange_client.close()


# Example of how it might be used:
# if __name__ == '__main__':
#     import asyncio

#     async def main():
#         client = ExchangeClient(settings.exchange)
#         repo = DataRepository(client)
#         btc_data = await repo.get_historical_data("BTC/USDT", "1h", 200)
#         if btc_data is not None:
#             print("\nBTC Data from Repository:")
#             print(btc_data.head())
#         await repo.close_connections()

#     asyncio.run(main()) 