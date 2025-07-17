import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any
from xgboost import XGBClassifier
import logging
from src.config.settings import MODEL_DIR, MODEL_PREFIX, SIGNAL_THRESHOLD

logger = logging.getLogger(__name__)

class ModelManager:
    """Класс для управления ML моделями"""
    
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.models = {}
        self.model_info = {}
        self._load_models()
    
    def _load_models(self):
        """Загружает все доступные модели"""
        if not os.path.exists(self.model_dir):
            logger.warning(f"Model directory {self.model_dir} does not exist")
            return
            
        for filename in os.listdir(self.model_dir):
            if filename.endswith('.json') and filename.startswith(MODEL_PREFIX) and not filename.endswith('_info.json'):
                model_name = filename.replace('.json', '')
                model_path = os.path.join(self.model_dir, filename)
                
                try:
                    model = XGBClassifier()
                    model.load_model(model_path)
                    self.models[model_name] = model
                    
                    # Загружаем информацию о модели
                    info_path = model_path.replace('.json', '_info.json')
                    if os.path.exists(info_path):
                        with open(info_path, 'r') as f:
                            self.model_info[model_name] = json.load(f)
                    
                    logger.info(f"Loaded model: {model_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load model {model_name}: {e}")
    
    def get_model(self, symbol: str, timeframe: str) -> Optional[XGBClassifier]:
        """
        Получает модель для указанной пары и таймфрейма
        
        Args:
            symbol: Торговая пара (например, 'SOLUSDT')
            timeframe: Таймфрейм ('15m', '1h', '4h', '1d')
            
        Returns:
            XGBClassifier или None если модель не найдена
        """
        # Нормализуем символ
        symbol = symbol.replace('/', '').replace('USDT', 'USDT')
        
        # Формируем имя модели
        model_name = f"{MODEL_PREFIX}_{symbol}_{timeframe}"
        
        return self.models.get(model_name)
    
    def predict(self, symbol: str, timeframe: str, features: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Делает предсказание для указанной пары и таймфрейма
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            features: Матрица признаков
            
        Returns:
            Tuple: (сигнал, уверенность, вероятности классов)
        """
        model = self.get_model(symbol, timeframe)
        
        if model is None:
            logger.warning(f"No model found for {symbol} {timeframe}")
            return "none", 0.0, {"long": 0.0, "short": 0.0, "none": 1.0}
        
        try:
            # Получаем вероятности классов
            probabilities = model.predict_proba(features)[0]
            
            # Определяем классы
            classes = model.classes_
            class_probs = {class_name: prob for class_name, prob in zip(classes, probabilities)}
            
            # Находим класс с максимальной вероятностью
            max_prob_idx = np.argmax(probabilities)
            predicted_class = classes[max_prob_idx]
            confidence = probabilities[max_prob_idx]
            
            # Если уверенность ниже порога, считаем сигнал "none"
            if confidence < SIGNAL_THRESHOLD:
                predicted_class = "none"
                confidence = class_probs.get("none", 0.0)
            
            return predicted_class, confidence, class_probs
            
        except Exception as e:
            logger.error(f"Error making prediction for {symbol} {timeframe}: {e}")
            return "none", 0.0, {"long": 0.0, "short": 0.0, "none": 1.0}
    
    def get_all_predictions(self, symbol: str, features_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        Получает предсказания для всех таймфреймов
        
        Args:
            symbol: Торговая пара
            features_dict: Словарь с признаками для каждого таймфрейма
            
        Returns:
            Словарь с предсказаниями для каждого таймфрейма
        """
        predictions = {}
        
        for timeframe, features in features_dict.items():
            signal, confidence, probabilities = self.predict(symbol, timeframe, features)
            
            predictions[timeframe] = {
                'signal': signal,
                'confidence': confidence,
                'probabilities': probabilities
            }
        
        return predictions
    
    def get_overall_signal(self, predictions: Dict[str, Dict[str, Any]]) -> Tuple[str, float]:
        """
        Определяет общий сигнал на основе предсказаний всех таймфреймов
        
        Args:
            predictions: Словарь с предсказаниями для каждого таймфрейма
            
        Returns:
            Tuple: (общий сигнал, средняя уверенность)
        """
        if not predictions:
            return "none", 0.0
        
        # Подсчитываем количество сигналов каждого типа
        signal_counts = {"long": 0, "short": 0, "none": 0}
        total_confidence = 0.0
        valid_predictions = 0
        
        for timeframe_pred in predictions.values():
            signal = timeframe_pred['signal']
            confidence = timeframe_pred['confidence']
            
            if signal in signal_counts:
                signal_counts[signal] += 1
                total_confidence += confidence
                valid_predictions += 1
        
        if valid_predictions == 0:
            return "none", 0.0
        
        # Определяем общий сигнал
        max_count = max(signal_counts.values())
        max_signals = [signal for signal, count in signal_counts.items() if count == max_count]
        
        # Если есть несколько сигналов с одинаковым количеством, выбираем по приоритету
        if len(max_signals) > 1:
            if "long" in max_signals:
                overall_signal = "long"
            elif "short" in max_signals:
                overall_signal = "short"
            else:
                overall_signal = "none"
        else:
            overall_signal = max_signals[0]
        
        avg_confidence = total_confidence / valid_predictions
        
        return overall_signal, avg_confidence
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает информацию о доступных моделях"""
        available = {}
        
        for model_name, model in self.models.items():
            info = self.model_info.get(model_name, {})
            available[model_name] = {
                'model': model,
                'info': info,
                'feature_count': model.n_features_in_ if hasattr(model, 'n_features_in_') else 0,
                'class_count': len(model.classes_) if hasattr(model, 'classes_') else 0
            }
        
        return available 