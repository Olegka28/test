import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from src.data.connectors.bybit_connector import BybitConnector
from src.features.feature_processor import FeatureProcessor
from src.ml.model_manager import ModelManager
from src.config.settings import SUPPORTED_PAIRS, SUPPORTED_TIMEFRAMES, CANDLE_LIMIT, TP_RATIO, SL_RATIO

logger = logging.getLogger(__name__)

class SignalService:
    """Сервис для генерации торговых сигналов"""
    
    def __init__(self):
        self.connector = BybitConnector()
        self.feature_processor = FeatureProcessor()
        self.model_manager = ModelManager()
        
    def get_signal_for_pair(self, symbol: str, timeframe: str) -> Dict[str, any]:
        """
        Получает сигнал для конкретной пары и таймфрейма
        
        Args:
            symbol: Торговая пара (например, 'SOL/USDT')
            timeframe: Таймфрейм ('15m', '1h', '4h', '1d')
            
        Returns:
            Словарь с информацией о сигнале
        """
        try:
            # Получаем последние ~100 свечей по запросу пользователя (3 дня для всех таймфреймов)
            df = self.connector.get_historical_data(symbol, timeframe, days=3)
            if df is None:
                df = pd.DataFrame()
            if df.empty:
                logger.warning(f"No data received for {symbol} {timeframe}")
                return self._create_empty_signal(symbol, timeframe)
            
            # Обрабатываем признаки
            df_with_features = self.feature_processor.process_features(df)
            
            if df_with_features.empty:
                logger.warning(f"No features generated for {symbol} {timeframe}")
                return self._create_empty_signal(symbol, timeframe)
            
            # Получаем признаки для предсказания
            features = self.feature_processor.get_latest_features(df_with_features)
            
            if features.size == 0:
                logger.warning(f"No features available for prediction {symbol} {timeframe}")
                return self._create_empty_signal(symbol, timeframe)
            
            # Делаем предсказание
            signal, confidence, probabilities = self.model_manager.predict(symbol, timeframe, features)
            
            # Получаем текущую цену
            current_price = df['close'].iloc[-1]
            
            # Рассчитываем уровни TP/SL
            tp_price, sl_price = self._calculate_tp_sl(current_price, signal)
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal': signal,
                'confidence': confidence,
                'probabilities': probabilities,
                'current_price': current_price,
                'tp_price': tp_price,
                'sl_price': sl_price,
                'timestamp': df['timestamp'].iloc[-1] if 'timestamp' in df.columns else None,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error getting signal for {symbol} {timeframe}: {e}")
            return self._create_empty_signal(symbol, timeframe, error=str(e))
    
    def get_overview_for_pair(self, symbol: str) -> Dict[str, any]:
        """
        Получает обзор сигналов для пары по всем таймфреймам
        
        Args:
            symbol: Торговая пара
            
        Returns:
            Словарь с обзором сигналов
        """
        try:
            timeframe_signals = {}
            features_dict = {}
            
            # Получаем данные и сигналы для всех таймфреймов
            for timeframe in SUPPORTED_TIMEFRAMES:
                # Получаем данные
                df = self.connector.get_klines(symbol, timeframe, limit=CANDLE_LIMIT)
                
                if not df.empty:
                    # Обрабатываем признаки
                    df_with_features = self.feature_processor.process_features(df)
                    
                    if not df_with_features.empty:
                        # Получаем признаки для предсказания
                        features = self.feature_processor.get_latest_features(df_with_features)
                        
                        if features.size > 0:
                            features_dict[timeframe] = features
                            
                            # Делаем предсказание
                            signal, confidence, probabilities = self.model_manager.predict(symbol, timeframe, features)
                            
                            timeframe_signals[timeframe] = {
                                'signal': signal,
                                'confidence': confidence,
                                'probabilities': probabilities,
                                'current_price': df['close'].iloc[-1],
                                'timestamp': df['timestamp'].iloc[-1]
                            }
            
            # Определяем общий сигнал
            overall_signal, avg_confidence = self.model_manager.get_overall_signal(timeframe_signals)
            
            return {
                'symbol': symbol,
                'overall_signal': overall_signal,
                'overall_confidence': avg_confidence,
                'timeframe_signals': timeframe_signals,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error getting overview for {symbol}: {e}")
            return {
                'symbol': symbol,
                'overall_signal': 'none',
                'overall_confidence': 0.0,
                'timeframe_signals': {},
                'success': False,
                'error': str(e)
            }
    
    def get_all_signals(self) -> Dict[str, Dict[str, any]]:
        """
        Получает сигналы для всех поддерживаемых пар
        
        Returns:
            Словарь с сигналами для всех пар
        """
        all_signals = {}
        
        for symbol in SUPPORTED_PAIRS:
            overview = self.get_overview_for_pair(symbol)
            all_signals[symbol] = overview
        
        return all_signals
    
    def _calculate_tp_sl(self, current_price: float, signal: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Рассчитывает уровни Take Profit и Stop Loss
        
        Args:
            current_price: Текущая цена
            signal: Сигнал ('long', 'short', 'none')
            
        Returns:
            Tuple: (tp_price, sl_price)
        """
        if signal == 'long':
            tp_price = current_price * (1 + TP_RATIO)
            sl_price = current_price * (1 - SL_RATIO)
        elif signal == 'short':
            tp_price = current_price * (1 - TP_RATIO)
            sl_price = current_price * (1 + SL_RATIO)
        else:
            tp_price = None
            sl_price = None
        
        return tp_price, sl_price
    
    def _create_empty_signal(self, symbol: str, timeframe: str, error: str = None) -> Dict[str, any]:
        """Создает пустой сигнал при ошибке"""
        signal_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'signal': 'none',
            'confidence': 0.0,
            'probabilities': {'long': 0.0, 'short': 0.0, 'none': 1.0},
            'current_price': 0.0,
            'tp_price': None,
            'sl_price': None,
            'timestamp': None,
            'success': False
        }
        
        if error:
            signal_data['error'] = error
        
        return signal_data
    
    def format_signal_message(self, signal_data: Dict[str, any]) -> str:
        """
        Форматирует сигнал в читаемое сообщение
        
        Args:
            signal_data: Данные сигнала
            
        Returns:
            Отформатированное сообщение
        """
        if not signal_data.get('success', False):
            return f"❌ Ошибка получения сигнала для {signal_data['symbol']} {signal_data['timeframe']}"
        
        symbol = signal_data['symbol']
        timeframe = signal_data['timeframe']
        signal = signal_data['signal']
        confidence = signal_data['confidence']
        current_price = signal_data['current_price']
        tp_price = signal_data['tp_price']
        sl_price = signal_data['sl_price']
        
        # Эмодзи для сигналов
        signal_emoji = {
            'long': '🟢',
            'short': '🔴',
            'none': '⚪'
        }
        
        # Эмодзи для таймфреймов
        timeframe_emoji = {
            '15m': '⏱️',
            '1h': '🕐',
            '4h': '🕓',
            '1d': '📅'
        }
        
        message = f"{signal_emoji.get(signal, '⚪')} **{symbol} {timeframe_emoji.get(timeframe, '')} {timeframe}**\n"
        message += f"💰 Цена: ${current_price:.4f}\n"
        message += f"📊 Сигнал: {signal.upper()}\n"
        message += f"🎯 Уверенность: {confidence:.1%}\n"
        
        if tp_price and sl_price:
            message += f"🎯 TP: ${tp_price:.4f}\n"
            message += f"🛑 SL: ${sl_price:.4f}\n"
        
        return message
    
    def format_overview_message(self, overview_data: Dict[str, any]) -> str:
        """
        Форматирует обзор сигналов в читаемое сообщение
        
        Args:
            overview_data: Данные обзора
            
        Returns:
            Отформатированное сообщение
        """
        if not overview_data.get('success', False):
            return f"❌ Ошибка получения обзора для {overview_data['symbol']}"
        
        symbol = overview_data['symbol']
        overall_signal = overview_data['overall_signal']
        overall_confidence = overview_data['overall_confidence']
        timeframe_signals = overview_data['timeframe_signals']
        
        # Эмодзи для сигналов
        signal_emoji = {
            'long': '🟢',
            'short': '🔴',
            'none': '⚪'
        }
        
        # Эмодзи для таймфреймов
        timeframe_emoji = {
            '15m': '⏱️',
            '1h': '🕐',
            '4h': '🕓',
            '1d': '📅'
        }
        
        message = f"📈 **{symbol} - Общий обзор**\n"
        message += f"{signal_emoji.get(overall_signal, '⚪')} Общий сигнал: {overall_signal.upper()}\n"
        message += f"🎯 Общая уверенность: {overall_confidence:.1%}\n\n"
        
        message += "📊 **По таймфреймам:**\n"
        
        for timeframe in SUPPORTED_TIMEFRAMES:
            if timeframe in timeframe_signals:
                tf_data = timeframe_signals[timeframe]
                signal = tf_data['signal']
                confidence = tf_data['confidence']
                price = tf_data['current_price']
                
                message += f"{timeframe_emoji.get(timeframe, '')} {timeframe}: "
                message += f"{signal_emoji.get(signal, '⚪')} {signal.upper()} "
                message += f"({confidence:.1%}) ${price:.4f}\n"
            else:
                message += f"{timeframe_emoji.get(timeframe, '')} {timeframe}: ❌ Нет данных\n"
        
        return message 