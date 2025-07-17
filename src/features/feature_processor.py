import pandas as pd
import numpy as np
from typing import List, Optional
from src.features.technical_indicators import TechnicalIndicators
from src.features.pattern_features import PatternFeatures
from src.config.settings import FEATURE_WINDOW

class FeatureProcessor:
    """Класс для обработки и создания всех признаков"""
    
    def __init__(self):
        self.feature_columns = []
        
    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обрабатывает DataFrame и добавляет все технические индикаторы и паттерны
        
        Args:
            df: DataFrame с колонками timestamp, open, high, low, close, volume
            
        Returns:
            DataFrame с добавленными признаками
        """
        if df.empty:
            return df
            
        # Создаем копию для работы
        result_df = df.copy()
        
        # Добавляем технические индикаторы
        result_df = TechnicalIndicators.calculate_all_indicators(result_df)
        
        # Добавляем паттерны свечей
        result_df = PatternFeatures.calculate_candlestick_patterns(result_df)
        
        # Добавляем лаговые признаки
        result_df = self._add_lag_features(result_df)
        
        # Добавляем скользящие статистики
        result_df = self._add_rolling_features(result_df)
        
        # Удаляем NaN значения
        result_df = result_df.dropna()
        
        # Сохраняем список колонок с признаками
        self.feature_columns = self._get_feature_columns(result_df)
        
        return result_df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет лаговые признаки"""
        result_df = df.copy()
        
        # Лаги для цены
        for lag in [1, 2, 3, 5, 10]:
            result_df[f'close_lag_{lag}'] = result_df['close'].shift(lag)
            result_df[f'volume_lag_{lag}'] = result_df['volume'].shift(lag)
            result_df[f'high_lag_{lag}'] = result_df['high'].shift(lag)
            result_df[f'low_lag_{lag}'] = result_df['low'].shift(lag)
        
        # Лаги для индикаторов
        for lag in [1, 2, 3]:
            result_df[f'rsi_lag_{lag}'] = result_df['rsi'].shift(lag)
            result_df[f'macd_lag_{lag}'] = result_df['macd'].shift(lag)
            result_df[f'bb_position_lag_{lag}'] = result_df['bb_position'].shift(lag)
        
        return result_df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет скользящие статистики"""
        result_df = df.copy()
        
        # Скользящие средние для объема
        for window in [5, 10, 20]:
            result_df[f'volume_sma_{window}'] = result_df['volume'].rolling(window=window).mean()
            result_df[f'volume_std_{window}'] = result_df['volume'].rolling(window=window).std()
        
        # Скользящие статистики для волатильности
        for window in [5, 10, 20]:
            result_df[f'volatility_sma_{window}'] = result_df['volatility'].rolling(window=window).mean()
            result_df[f'volatility_std_{window}'] = result_df['volatility'].rolling(window=window).std()
        
        # Скользящие корреляции
        result_df['price_volume_corr'] = result_df['close'].rolling(window=20).corr(result_df['volume'])
        
        return result_df
    
    def _get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Получает список колонок с признаками (исключая базовые OHLCV и timestamp)"""
        base_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in df.columns if col not in base_columns]
        return feature_columns
    
    def get_feature_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Получает матрицу признаков для ML модели
        
        Args:
            df: DataFrame с обработанными признаками
            
        Returns:
            numpy array с признаками
        """
        if self.feature_columns:
            return df[self.feature_columns].values
        else:
            # Если список колонок не определен, используем все кроме базовых
            base_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in df.columns if col not in base_columns]
            return df[feature_columns].values
    
    def get_latest_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Получает признаки для последней свечи (для предсказания)
        
        Args:
            df: DataFrame с обработанными признаками
            
        Returns:
            numpy array с признаками последней свечи
        """
        feature_matrix = self.get_feature_matrix(df)
        if len(feature_matrix) > 0:
            return feature_matrix[-1:].reshape(1, -1)
        else:
            return np.array([])
    
    def get_feature_names(self) -> List[str]:
        """Возвращает список названий признаков"""
        return self.feature_columns.copy() 