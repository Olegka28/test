import pandas as pd
import numpy as np
from typing import Dict, List

class PatternFeatures:
    """Класс для расчета паттернов свечей"""
    
    @staticmethod
    def calculate_candlestick_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Рассчитывает паттерны свечей"""
        result_df = df.copy()
        
        # Базовые характеристики свечей
        result_df['body_size'] = abs(result_df['close'] - result_df['open'])
        result_df['upper_shadow'] = result_df['high'] - result_df[['open', 'close']].max(axis=1)
        result_df['lower_shadow'] = result_df[['open', 'close']].min(axis=1) - result_df['low']
        result_df['total_range'] = result_df['high'] - result_df['low']
        
        # Тип свечи (бычья/медвежья)
        result_df['is_bullish'] = (result_df['close'] > result_df['open']).astype(int)
        result_df['is_bearish'] = (result_df['close'] < result_df['open']).astype(int)
        result_df['is_doji'] = (abs(result_df['close'] - result_df['open']) <= result_df['total_range'] * 0.1).astype(int)
        
        # Размеры относительно предыдущих свечей
        result_df['body_size_ratio'] = result_df['body_size'] / result_df['body_size'].rolling(window=20).mean()
        result_df['range_ratio'] = result_df['total_range'] / result_df['total_range'].rolling(window=20).mean()
        
        # Паттерны свечей
        result_df['hammer'] = PatternFeatures._is_hammer(result_df)
        result_df['shooting_star'] = PatternFeatures._is_shooting_star(result_df)
        result_df['engulfing_bullish'] = PatternFeatures._is_engulfing_bullish(result_df)
        result_df['engulfing_bearish'] = PatternFeatures._is_engulfing_bearish(result_df)
        result_df['doji'] = result_df['is_doji']
        
        # Паттерны из нескольких свечей
        result_df['three_white_soldiers'] = PatternFeatures._is_three_white_soldiers(result_df)
        result_df['three_black_crows'] = PatternFeatures._is_three_black_crows(result_df)
        result_df['morning_star'] = PatternFeatures._is_morning_star(result_df)
        result_df['evening_star'] = PatternFeatures._is_evening_star(result_df)
        
        return result_df
    
    @staticmethod
    def _is_hammer(df: pd.DataFrame) -> pd.Series:
        """Определяет паттерн молот (Hammer)"""
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        return (
            (lower_shadow > body_size * 2) &  # Нижняя тень в 2 раза больше тела
            (upper_shadow < body_size * 0.5) &  # Верхняя тень меньше половины тела
            (body_size > 0)  # Тело не является доджи
        ).astype(int)
    
    @staticmethod
    def _is_shooting_star(df: pd.DataFrame) -> pd.Series:
        """Определяет паттерн падающая звезда (Shooting Star)"""
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        return (
            (upper_shadow > body_size * 2) &  # Верхняя тень в 2 раза больше тела
            (lower_shadow < body_size * 0.5) &  # Нижняя тень меньше половины тела
            (body_size > 0)  # Тело не является доджи
        ).astype(int)
    
    @staticmethod
    def _is_engulfing_bullish(df: pd.DataFrame) -> pd.Series:
        """Определяет бычий паттерн поглощения"""
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        curr_open = df['open']
        curr_close = df['close']
        
        return (
            (prev_close < prev_open) &  # Предыдущая свеча медвежья
            (curr_close > curr_open) &  # Текущая свеча бычья
            (curr_open < prev_close) &  # Открытие текущей ниже закрытия предыдущей
            (curr_close > prev_open)  # Закрытие текущей выше открытия предыдущей
        ).astype(int)
    
    @staticmethod
    def _is_engulfing_bearish(df: pd.DataFrame) -> pd.Series:
        """Определяет медвежий паттерн поглощения"""
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        curr_open = df['open']
        curr_close = df['close']
        
        return (
            (prev_close > prev_open) &  # Предыдущая свеча бычья
            (curr_close < curr_open) &  # Текущая свеча медвежья
            (curr_open > prev_close) &  # Открытие текущей выше закрытия предыдущей
            (curr_close < prev_open)  # Закрытие текущей ниже открытия предыдущей
        ).astype(int)
    
    @staticmethod
    def _is_three_white_soldiers(df: pd.DataFrame) -> pd.Series:
        """Определяет паттерн три белых солдата"""
        return (
            (df['is_bullish'].shift(2) == 1) &  # Три бычьих свечи подряд
            (df['is_bullish'].shift(1) == 1) &
            (df['is_bullish'] == 1) &
            (df['close'] > df['close'].shift(1)) &  # Каждая свеча закрывается выше предыдущей
            (df['close'].shift(1) > df['close'].shift(2))
        ).astype(int)
    
    @staticmethod
    def _is_three_black_crows(df: pd.DataFrame) -> pd.Series:
        """Определяет паттерн три черных ворона"""
        return (
            (df['is_bearish'].shift(2) == 1) &  # Три медвежьих свечи подряд
            (df['is_bearish'].shift(1) == 1) &
            (df['is_bearish'] == 1) &
            (df['close'] < df['close'].shift(1)) &  # Каждая свеча закрывается ниже предыдущей
            (df['close'].shift(1) < df['close'].shift(2))
        ).astype(int)
    
    @staticmethod
    def _is_morning_star(df: pd.DataFrame) -> pd.Series:
        """Определяет паттерн утренняя звезда"""
        return (
            (df['is_bearish'].shift(2) == 1) &  # Первая свеча медвежья
            (df['is_doji'].shift(1) == 1) &  # Вторая свеча доджи
            (df['is_bullish'] == 1) &  # Третья свеча бычья
            (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2)  # Третья свеча закрывается выше середины первой
        ).astype(int)
    
    @staticmethod
    def _is_evening_star(df: pd.DataFrame) -> pd.Series:
        """Определяет паттерн вечерняя звезда"""
        return (
            (df['is_bullish'].shift(2) == 1) &  # Первая свеча бычья
            (df['is_doji'].shift(1) == 1) &  # Вторая свеча доджи
            (df['is_bearish'] == 1) &  # Третья свеча медвежья
            (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2)  # Третья свеча закрывается ниже середины первой
        ).astype(int)
