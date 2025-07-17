import pandas as pd
import numpy as np
from typing import Dict, Any
from src.config.settings import RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BB_PERIOD, BB_STD

class TechnicalIndicators:
    """Класс для расчета технических индикаторов"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        """Рассчитывает RSI (Relative Strength Index)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi 
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL) -> Dict[str, pd.Series]:
        """Рассчитывает MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = BB_PERIOD, std_dev: float = BB_STD) -> Dict[str, pd.Series]:
        """Рассчитывает Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band,
            'bb_width': (upper_band - lower_band) / sma,
            'bb_position': (prices - lower_band) / (upper_band - lower_band)
        }
    
    @staticmethod
    def calculate_moving_averages(prices: pd.Series) -> Dict[str, pd.Series]:
        """Рассчитывает различные скользящие средние"""
        return {
            'sma_5': prices.rolling(window=5).mean(),
            'sma_10': prices.rolling(window=10).mean(),
            'sma_20': prices.rolling(window=20).mean(),
            'sma_50': prices.rolling(window=50).mean(),
            'ema_12': prices.ewm(span=12).mean(),
            'ema_26': prices.ewm(span=26).mean()
        }
    
    @staticmethod
    def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Рассчитывает Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Рассчитывает Average True Range (ATR)"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_volume_indicators(close: pd.Series, volume: pd.Series) -> Dict[str, pd.Series]:
        """Рассчитывает индикаторы объема"""
        # Volume Weighted Average Price (VWAP)
        typical_price = close  # Для простоты используем close вместо (high + low + close) / 3
        vwap = (typical_price * volume).rolling(window=20).sum() / volume.rolling(window=20).sum()
        
        # Volume Rate of Change
        volume_roc = volume.pct_change(periods=10) * 100
        
        # On Balance Volume (OBV)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        
        return {
            'vwap': vwap,
            'volume_roc': volume_roc,
            'obv': obv
        }
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Рассчитывает все технические индикаторы для DataFrame"""
        result_df = df.copy()
        
        # RSI
        result_df['rsi'] = TechnicalIndicators.calculate_rsi(result_df['close'])
        
        # MACD
        macd_data = TechnicalIndicators.calculate_macd(result_df['close'])
        result_df['macd'] = macd_data['macd']
        result_df['macd_signal'] = macd_data['signal']
        result_df['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = TechnicalIndicators.calculate_bollinger_bands(result_df['close'])
        result_df['bb_upper'] = bb_data['upper']
        result_df['bb_middle'] = bb_data['middle']
        result_df['bb_lower'] = bb_data['lower']
        result_df['bb_width'] = bb_data['bb_width']
        result_df['bb_position'] = bb_data['bb_position']
        
        # Moving Averages
        ma_data = TechnicalIndicators.calculate_moving_averages(result_df['close'])
        for name, values in ma_data.items():
            result_df[name] = values
        
        # Stochastic
        stoch_data = TechnicalIndicators.calculate_stochastic(result_df['high'], result_df['low'], result_df['close'])
        result_df['stoch_k'] = stoch_data['k']
        result_df['stoch_d'] = stoch_data['d']
        
        # ATR
        result_df['atr'] = TechnicalIndicators.calculate_atr(result_df['high'], result_df['low'], result_df['close'])
        
        # Volume Indicators
        volume_data = TechnicalIndicators.calculate_volume_indicators(result_df['close'], result_df['volume'])
        result_df['vwap'] = volume_data['vwap']
        result_df['volume_roc'] = volume_data['volume_roc']
        result_df['obv'] = volume_data['obv']
        
        # Дополнительные индикаторы
        result_df['price_change'] = result_df['close'].pct_change()
        result_df['price_change_5'] = result_df['close'].pct_change(periods=5)
        result_df['price_change_10'] = result_df['close'].pct_change(periods=10)
        
        result_df['volatility'] = result_df['close'].rolling(window=20).std()
        result_df['volatility_ratio'] = result_df['volatility'] / result_df['close']
        
        return result_df 