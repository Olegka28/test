import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
import logging
from src.features.feature_processor import FeatureProcessor
from src.config.settings import MODEL_DIR, MODEL_PREFIX, FEATURE_WINDOW
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# --- Добавить функцию для сериализации numpy типов ---
def to_serializable(val):
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {k: to_serializable(v) for k, v in val.items()}
    if isinstance(val, list):
        return [to_serializable(v) for v in val]
    return val

class ModelTrainer:
    """Класс для обучения ML моделей"""
    
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.feature_processor = FeatureProcessor()
        
        # Создаем директорию для моделей если её нет
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_dataset(self, df: pd.DataFrame, window_size: int = FEATURE_WINDOW) -> Tuple[np.ndarray, pd.Series]:
        """
        Подготавливает датасет для обучения
        
        Args:
            df: DataFrame с OHLCV данными
            window_size: Размер окна для создания признаков
            
        Returns:
            Tuple: (признаки, метки)
        """
        if df.empty:
            return np.empty((0, 0)), pd.Series(dtype='object')
        
        # Обрабатываем признаки
        df_with_features = self.feature_processor.process_features(df)
        
        if df_with_features.empty:
            return np.empty((0, 0)), pd.Series(dtype='object')
        
        # Получаем матрицу признаков
        X = self.feature_processor.get_feature_matrix(df_with_features)
        
        if X.size == 0:
            return np.empty((0, 0)), pd.Series(dtype='object')
        
        # Создаем метки (здесь нужно будет добавить логику создания меток)
        y = self._create_labels(df_with_features)
        
        # Удаляем строки где нет меток
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        return X, y
    
    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Создает метки для обучения (заглушка - нужно реализовать логику)
        
        Args:
            df: DataFrame с признаками
            
        Returns:
            Series с метками
        """
        # Здесь должна быть логика создания меток
        # Например, на основе будущего движения цены
        
        # Временная заглушка - случайные метки
        np.random.seed(42)
        labels = np.random.choice(['long', 'short', 'none'], size=len(df), p=[0.3, 0.3, 0.4])
        
        return pd.Series(labels, index=df.index, dtype='object')
    
    def train_model(
        self,
        symbol: str,
        timeframe: str,
        X: np.ndarray,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, any]:
        """
        Обучает модель для указанной пары и таймфрейма
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            X: Матрица признаков
            y: Вектор меток
            test_size: Доля тестовых данных
            random_state: Seed для воспроизводимости
            
        Returns:
            Словарь с результатами обучения
        """
        try:
            # --- Кодируем метки в числа ---
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            # Разделяем данные
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
            )
            
            # Вычисляем веса классов для балансировки
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train),
                y=y_train
            )
            weight_dict = dict(zip(np.unique(y_train), class_weights))
            
            # Создаем модель
            model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                scale_pos_weight=1.0,
                class_weight=weight_dict
            )
            
            # Обучаем модель
            logger.info(f"Training model for {symbol} {timeframe}...")
            model.fit(X_train, y_train)
            
            # Предсказания
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # --- Декодируем обратно для метрик ---
            y_test_decoded = le.inverse_transform(y_test)
            y_pred_decoded = le.inverse_transform(y_pred)
            
            # Метрики
            report = classification_report(y_test_decoded, y_pred_decoded, output_dict=True)
            conf_matrix = confusion_matrix(y_test_decoded, y_pred_decoded)
            
            # Сохраняем модель
            model_name = f"{MODEL_PREFIX}_{symbol.replace('/', '')}_{timeframe}"
            model_path = os.path.join(self.model_dir, f"{model_name}.json")
            model.save_model(model_path)
            
            # Сохраняем информацию о модели
            model_info = {
                'symbol': symbol,
                'timeframe': timeframe,
                'feature_count': X.shape[1],
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'classes': list(le.classes_),
                'feature_importance': list(model.feature_importances_),
                'feature_names': self.feature_processor.get_feature_names(),
                'metrics': {
                    'accuracy': report['accuracy'] if 'accuracy' in report else None,
                    'precision': report['weighted avg']['precision'] if 'weighted avg' in report else None,
                    'recall': report['weighted avg']['recall'] if 'weighted avg' in report else None,
                    'f1_score': report['weighted avg']['f1-score'] if 'weighted avg' in report else None
                },
                'class_metrics': report
            }
            
            info_path = os.path.join(self.model_dir, f"{model_name}_info.json")
            with open(info_path, 'w') as f:
                json.dump(to_serializable(model_info), f, indent=2)
            
            logger.info(f"Model saved: {model_path}")
            logger.info(f"Accuracy: {report['accuracy']:.4f}")
            
            return {
                'model': model,
                'model_path': model_path,
                'info': model_info,
                'X_test': X_test,
                'y_test': y_test_decoded,
                'y_pred': y_pred_decoded,
                'y_pred_proba': y_pred_proba
            }
            
        except Exception as e:
            logger.error(f"Error training model for {symbol} {timeframe}: {e}")
            raise
    
    def train_all_models(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, any]]:
        """
        Обучает модели для всех пар и таймфреймов
        
        Args:
            data_dict: Словарь с данными для каждой пары и таймфрейма
            Формат: {'SOLUSDT_15m': df, 'SOLUSDT_1h': df, ...}
            
        Returns:
            Словарь с результатами обучения для всех моделей
        """
        results = {}
        
        for key, df in data_dict.items():
            try:
                # Парсим ключ
                parts = key.split('_')
                if len(parts) >= 2:
                    symbol = '_'.join(parts[:-1])  # Все кроме последней части
                    timeframe = parts[-1]
                    
                    # Подготавливаем данные
                    X, y = self.prepare_dataset(df)
                    
                    if X.size > 0 and len(y) > 0:
                        # Обучаем модель
                        result = self.train_model(symbol, timeframe, X, y)
                        results[key] = result
                    else:
                        logger.warning(f"No valid data for {key}")
                        
            except Exception as e:
                logger.error(f"Error training model for {key}: {e}")
                continue
        
        return results
    
    def evaluate_model(self, model_result: Dict[str, any]) -> str:
        """
        Оценивает качество модели и возвращает отчет
        
        Args:
            model_result: Результат обучения модели
            
        Returns:
            Строка с отчетом
        """
        info = model_result['info']
        metrics = info['metrics']
        
        report = f"📊 **Результаты обучения модели {info['symbol']} {info['timeframe']}**\n\n"
        report += f"📈 **Метрики:**\n"
        report += f"• Точность: {metrics['accuracy']:.4f}\n"
        report += f"• Precision: {metrics['precision']:.4f}\n"
        report += f"• Recall: {metrics['recall']:.4f}\n"
        report += f"• F1-Score: {metrics['f1_score']:.4f}\n\n"
        
        report += f"📋 **Данные:**\n"
        report += f"• Признаков: {info['feature_count']}\n"
        report += f"• Обучающих примеров: {info['train_samples']}\n"
        report += f"• Тестовых примеров: {info['test_samples']}\n"
        report += f"• Классов: {len(info['classes'])}\n\n"
        
        # Топ-10 важных признаков
        feature_importance = list(zip(info['feature_names'], info['feature_importance']))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        report += f"🎯 **Топ-10 важных признаков:**\n"
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            report += f"{i+1}. {feature}: {importance:.4f}\n"
        
        return report
