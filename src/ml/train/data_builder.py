import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from src.data.connectors.bybit_connector import BybitConnector
from src.config.settings import SUPPORTED_PAIRS, SUPPORTED_TIMEFRAMES, CANDLE_LIMIT

logger = logging.getLogger(__name__)

class DataBuilder:
    """Класс для подготовки данных для обучения моделей"""
    
    def __init__(self):
        self.connector = BybitConnector()
    
    def download_training_data(
        self,
        pairs: List[str] = [],
        timeframes: List[str] = [],
        days: int = 365
    ) -> Dict[str, pd.DataFrame]:
        """
        Загружает исторические данные для обучения
        
        Args:
            pairs: Список торговых пар (если None, используются все поддерживаемые)
            timeframes: Список таймфреймов (если None, используются все поддерживаемые)
            days: Количество дней исторических данных
            
        Returns:
            Словарь с данными для каждой пары и таймфрейма
        """
        if not pairs:
            pairs = SUPPORTED_PAIRS
        if not timeframes:
            timeframes = SUPPORTED_TIMEFRAMES
        
        data_dict = {}
        
        for pair in pairs:
            for timeframe in timeframes:
                try:
                    logger.info(f"Downloading data for {pair} {timeframe}...")
                    
                    # Загружаем исторические данные
                    df = self.connector.get_historical_data(pair, timeframe, days=days)
                    
                    if not df.empty:
                        # Нормализуем ключ
                        key = f"{pair.replace('/', '')}_{timeframe}"
                        data_dict[key] = df
                        
                        logger.info(f"Downloaded {len(df)} candles for {pair} {timeframe}")
                    else:
                        logger.warning(f"No data received for {pair} {timeframe}")
                        
                except Exception as e:
                    logger.error(f"Error downloading data for {pair} {timeframe}: {e}")
                    continue
        
        return data_dict
    
    def create_labels(
        self,
        df: pd.DataFrame,
        future_periods: int = 5,
        tp_threshold: float = 0.02,
        sl_threshold: float = 0.01
    ) -> pd.Series:
        """
        Создает метки для обучения на основе будущего движения цены
        
        Args:
            df: DataFrame с OHLCV данными
            future_periods: Количество периодов в будущем для анализа
            tp_threshold: Порог для Take Profit (2%)
            sl_threshold: Порог для Stop Loss (1%)
            
        Returns:
            Series с метками ('long', 'short', 'none')
        """
        if df.empty or len(df) < future_periods:
            return pd.Series(dtype='object')
        
        labels = []
        
        for i in range(len(df) - future_periods):
            current_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+future_periods+1]
            
            # Рассчитываем максимальный рост и падение
            max_gain = (future_prices.max() - current_price) / current_price
            max_loss = (current_price - future_prices.min()) / current_price
            
            # Определяем сигнал
            if max_gain >= tp_threshold and max_gain > max_loss:
                labels.append('long')
            elif max_loss >= sl_threshold and max_loss > max_gain:
                labels.append('short')
            else:
                labels.append('none')
        
        # Добавляем NaN для последних периодов
        labels.extend([np.nan] * future_periods)
        
        return pd.Series(labels, index=df.index)
    
    def prepare_training_dataset(
        self,
        data_dict: Dict[str, pd.DataFrame],
        future_periods: int = 5
    ) -> Dict[str, Tuple[np.ndarray, pd.Series]]:
        """
        Подготавливает датасеты для обучения всех моделей
        
        Args:
            data_dict: Словарь с данными для каждой пары и таймфрейма
            future_periods: Количество периодов в будущем для создания меток
            
        Returns:
            Словарь с подготовленными данными (X, y) для каждой модели
        """
        from src.features.feature_processor import FeatureProcessor
        
        feature_processor = FeatureProcessor()
        training_datasets = {}
        
        for key, df in data_dict.items():
            try:
                logger.info(f"Preparing dataset for {key}...")
                
                # Создаем метки
                labels = self.create_labels(df, future_periods)
                
                # Обрабатываем признаки
                df_with_features = feature_processor.process_features(df)
                
                if df_with_features.empty:
                    logger.warning(f"No features generated for {key}")
                    continue
                
                # Получаем матрицу признаков
                X = feature_processor.get_feature_matrix(df_with_features)
                
                if X.size == 0:
                    logger.warning(f"No features available for {key}")
                    continue
                
                # --- Синхронизируем индексы меток и признаков ---
                labels = labels.loc[df_with_features.index]
                valid_indices = ~labels.isna()
                X = X[valid_indices]
                y = labels[valid_indices]
                
                if len(X) > 0 and len(y) > 0:
                    training_datasets[key] = (X, y)
                    logger.info(f"Prepared dataset for {key}: {len(X)} samples")
                else:
                    logger.warning(f"No valid samples for {key}")
                    
            except Exception as e:
                logger.error(f"Error preparing dataset for {key}: {e}")
                continue
        
        return training_datasets
    
    def save_training_data(
        self,
        data_dict: Dict[str, pd.DataFrame],
        output_dir: str = "data/training"
    ) -> None:
        """
        Сохраняет данные для обучения в CSV файлы
        
        Args:
            data_dict: Словарь с данными
            output_dir: Директория для сохранения
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for key, df in data_dict.items():
            try:
                filename = f"{key}.csv"
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {filename}")
                
            except Exception as e:
                logger.error(f"Error saving {key}: {e}")
    
    def load_training_data(
        self,
        input_dir: str = "data/training"
    ) -> Dict[str, pd.DataFrame]:
        """
        Загружает данные для обучения из CSV файлов
        
        Args:
            input_dir: Директория с данными
            
        Returns:
            Словарь с загруженными данными
        """
        import os
        data_dict = {}
        
        if not os.path.exists(input_dir):
            logger.warning(f"Directory {input_dir} does not exist")
            return data_dict
        
        for filename in os.listdir(input_dir):
            if filename.endswith('.csv'):
                try:
                    key = filename.replace('.csv', '')
                    filepath = os.path.join(input_dir, filename)
                    df = pd.read_csv(filepath)
                    data_dict[key] = df
                    logger.info(f"Loaded {filename}")
                    
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        
        return data_dict
    
    def get_data_summary(self, data_dict: Dict[str, pd.DataFrame]) -> str:
        """
        Возвращает сводку по загруженным данным
        
        Args:
            data_dict: Словарь с данными
            
        Returns:
            Строка со сводкой
        """
        if not data_dict:
            return "❌ Нет данных"
        
        summary = "📊 **Сводка по данным:**\n\n"
        
        for key, df in data_dict.items():
            if not df.empty:
                import pandas as pd
                start_date = pd.to_datetime(df['timestamp'].iloc[0])
                end_date = pd.to_datetime(df['timestamp'].iloc[-1])
                count = len(df)
                summary += f"**{key}:**\n"
                summary += f"• Период: {start_date} - {end_date}\n"
                summary += f"• Свечей: {count:,}\n"
                summary += f"• Дней: {(end_date - start_date).days}\n\n"
        
        return summary
