import pandas as pd
import requests
import time
from typing import List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class BybitConnector:
    def __init__(self, api_key: str = '', secret_key: str = '', testnet: bool = False):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = "https://api-testnet.bybit.com" if testnet else "https://api.bybit.com"
        
    def get_klines(self, symbol: str, interval: str, limit: int = 500, start_time: Optional[int] = None) -> pd.DataFrame:
        """
        Получает исторические данные OHLCV с Bybit
        
        Args:
            symbol: Торговая пара (например, 'SOLUSDT')
            interval: Таймфрейм ('15', '60', '240', 'D')
            limit: Количество свечей
            start_time: Время начала в миллисекундах
            
        Returns:
            DataFrame с колонками: timestamp, open, high, low, close, volume
        """
        # Конвертируем символ в формат Bybit
        symbol = symbol.replace('/', '').replace('USDT', 'USDT')
        
        # Конвертируем таймфрейм в формат Bybit
        interval_map = {
            '15m': '15',
            '1h': '60', 
            '4h': '240',
            '1d': 'D'
        }
        bybit_interval = interval_map.get(interval, interval)
        
        url = f"{self.base_url}/v5/market/kline"
        params = {
            'category': 'spot',
            'symbol': symbol,
            'interval': bybit_interval,
            'limit': limit
        }
        
        if start_time:
            params['start'] = start_time
            
        logger.info(f"Requesting Bybit API: {url} with params: {params}")
        
        try:
            response = requests.get(url, params=params)
            logger.info(f"Response status: {response.status_code}")
            logger.info(f"Response content: {response.text[:500]}...")
            
            if response.status_code != 200:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return pd.DataFrame()
                
            data = response.json()
            
            if data['retCode'] != 0:
                logger.error(f"Bybit API error: {data}")
                return pd.DataFrame()
                
            result = data['result']
            klines = result.get('list', [])
            
            if not klines:
                logger.warning("No klines returned from Bybit API")
                return pd.DataFrame()
                
            # Конвертируем в DataFrame
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'extra'])
            df = df.drop('extra', axis=1)
            
            # Конвертируем типы данных
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            logger.info(f"Successfully fetched {len(df)} candles for {symbol} {interval}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Bybit: {e}")
            return pd.DataFrame()
    
    def get_historical_data(self, symbol: str, interval: str, days: int = 30) -> pd.DataFrame:
        """
        Получает исторические данные за указанное количество дней
        """
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            df = self.get_klines(symbol, interval, limit=1000, start_time=current_start)
            if df.empty:
                break
                
            all_data.append(df)
            
            # Обновляем время начала для следующего запроса
            if len(df) > 0:
                last_timestamp = df['timestamp'].iloc[-1]
                current_start = int(last_timestamp.timestamp() * 1000) + 1
            else:
                break
                
            time.sleep(0.1)  # Небольшая задержка между запросами
            
        if all_data:
            return pd.concat(all_data, ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        else:
            return pd.DataFrame()
