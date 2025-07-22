from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime

from src.ml.model_manager import ModelManager
from src.config import settings
from .signal_object import SignalObject, SignalType, ConfidenceLevel


class BaseStrategy(ABC):
    """
    Универсальная базовая стратегия для всех моделей, пар и таймфреймов.
    Адаптируется под конкретную модель и возвращает структурированный сигнал.
    """

    def __init__(self, 
                 min_proba_threshold: float = 0.6,
                 risk_per_trade: float = 0.01,
                 commission: float = 0.0004,
                 slippage: float = 0.0001,
                 account_balance: float = 100):
        """
        Args:
            min_proba_threshold: Минимальная вероятность для сигнала
            risk_per_trade: Максимальный риск на сделку (0.02 = 2%)
            commission: Комиссия биржи
            slippage: Проскальзывание цены
            account_balance: Баланс аккаунта для расчета размера позиции
        """
        self.min_proba_threshold = min_proba_threshold
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.slippage = slippage
        self.account_balance = account_balance
        self.model_manager = ModelManager(settings.models)

    def analyze_model(self, 
                     symbol: str, 
                     timeframe: str, 
                     df: pd.DataFrame) -> Optional[SignalObject]:
        """
        Основной метод анализа модели и генерации сигнала.
        
        Args:
            symbol: Торговая пара (например, 'BTC/USDT')
            timeframe: Таймфрейм (например, '15m', '1h')
            df: DataFrame с данными и фичами
            
        Returns:
            SignalObject или None если сигнал не найден
        """
        # 1. Загружаем модель и фичи
        model = self.model_manager.load_model(symbol, timeframe)
        features_ref = self.model_manager.load_features(symbol, timeframe)
        
        if model is None or features_ref is None:
            print(f"No model found for {symbol} {timeframe}")
            return None
        
        # 2. Подготавливаем данные для модели
        X = self._prepare_features(df, features_ref)
        if X is None or X.empty:
            return None
        
        # 3. Получаем предсказание модели
        signal, probability = self._get_model_prediction(model, X)
        if signal is None:
            return None
        
        # 4. Анализируем рынок
        market_analysis = self._analyze_market(df, symbol, timeframe)
        
        # 5. Проверяем условия для входа
        if not self._should_enter_trade(signal, probability, market_analysis):
            return None
        
        # 6. Рассчитываем точки входа/выхода
        entry_info = self._calculate_entry_exit_points(
            signal, probability, df.iloc[-1], market_analysis
        )
        if entry_info is None:
            return None
        
        # 7. Создаем сигнал
        return self._create_signal_object(
            symbol, timeframe, signal, probability, entry_info, market_analysis
        )

    def _prepare_features(self, df: pd.DataFrame, features_ref: List[str]) -> Optional[pd.DataFrame]:
        """Подготавливает фичи для модели."""
        # Проверяем наличие необходимых колонок
        missing_features = [f for f in features_ref if f not in df.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
            # Добавляем недостающие фичи как 0
            for feature in missing_features:
                df[feature] = 0
        
        # Убираем лишние колонки
        extra_features = [col for col in df.columns if col not in features_ref and col not in ['open', 'high', 'low', 'close', 'volume']]
        if extra_features:
            df = df.drop(columns=extra_features)
        
        # Берем только нужные фичи в правильном порядке
        feature_cols = [col for col in features_ref if col in df.columns]
        if not feature_cols:
            return None
        
        X = df[feature_cols].copy()
        
        # Убираем NaN
        X = X.dropna()
        if X.empty:
            return None
        
        return X

    def _get_model_prediction(self, model, X: pd.DataFrame) -> tuple:
        """Получает предсказание от модели."""
        try:
            # Получаем вероятности
            proba = model.predict_proba(X.iloc[-1:])
            prediction = model.predict(X.iloc[-1:])
            
            # Берем вероятность для предсказанного класса
            pred_class = prediction[0]
            pred_proba = proba[0][prediction[0]]
            
            # Маппинг: 0 -> SHORT, 1 -> LONG
            signal_type = SignalType.LONG if pred_class == 1 else SignalType.SHORT
            
            return signal_type, pred_proba
            
        except Exception as e:
            print(f"Error getting model prediction: {e}")
            return None, None

    def _analyze_market(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict:
        """Анализирует рыночные условия."""
        analysis = {}
        
        # Получаем конфигурацию таймфрейма
        tf_config = settings.trading.timeframes.get(timeframe)
        
        # ATR для волатильности
        if 'atr_14' in df.columns:
            atr = df['atr_14'].iloc[-1]
            atr_percent = atr / df['close'].iloc[-1]
        else:
            # Рассчитываем ATR если его нет
            atr = ta.atr(df['high'], df['low'], df['close'], length=14)
            atr = atr.iloc[-1]
            atr_percent = atr / df['close'].iloc[-1]
        
        analysis['atr'] = atr
        analysis['atr_percent'] = atr_percent
        analysis['volatility_level'] = self._classify_volatility(atr_percent)
        
        # Тренд
        analysis['trend'] = self._analyze_trend(df)
        analysis['trend_strength'] = self._analyze_trend_strength(df)
        
        # RSI
        analysis['rsi'] = self._get_rsi(df)
        
        # Объем
        analysis['volume_signal'] = self._analyze_volume(df)
        
        # Рыночные условия
        analysis['market_condition'] = self._assess_market_condition(analysis)
        
        # Контекст с других таймфреймов
        analysis['context'] = self._get_multi_timeframe_context(df, symbol, timeframe)
        
        return analysis

    def _should_enter_trade(self, signal: SignalType, probability: float, market_analysis: Dict) -> bool:
        """Проверяет, стоит ли входить в сделку."""
        # Минимальная вероятность
        if probability < self.min_proba_threshold:
            return False
        
        # Рыночные условия
        if market_analysis.get('market_condition') == 'unfavorable':
            return False
        
        # Соответствие сигнала и тренда
        if not self._validate_signal_trend_alignment(signal, market_analysis):
            return False
        
        return True

    def _calculate_entry_exit_points(self, 
                                   signal: SignalType, 
                                   probability: float, 
                                   current_bar: pd.Series,
                                   market_analysis: Dict) -> Optional[Dict]:
        """Рассчитывает точки входа и выхода."""
        # Цена входа с учетом slippage
        entry_price = self._apply_slippage(current_bar['close'], signal)
        
        # Рассчитываем TP/SL на основе ATR
        atr = market_analysis.get('atr', 0)
        tp_multiplier = 2.0  # 2x ATR для TP
        sl_multiplier = 1.0  # 1x ATR для SL
        
        if signal == SignalType.LONG:
            take_profit = entry_price + (atr * tp_multiplier)
            stop_loss = entry_price - (atr * sl_multiplier)
        else:
            take_profit = entry_price - (atr * tp_multiplier)
            stop_loss = entry_price + (atr * sl_multiplier)
        
        # Размер позиции
        position_size = self._calculate_position_size(entry_price, stop_loss)
        
        # Комиссии
        commission_cost = self._calculate_commission_cost(entry_price, position_size)
        
        return {
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'position_size': position_size,
            'commission_cost': commission_cost
        }

    def _create_signal_object(self, 
                            symbol: str, 
                            timeframe: str, 
                            signal: SignalType, 
                            probability: float, 
                            entry_info: Dict, 
                            market_analysis: Dict) -> SignalObject:
        """Создает объект сигнала."""
        # Определяем уровень уверенности
        confidence = self._determine_confidence(probability, market_analysis)
        
        # Получаем точность модели
        model_accuracy = self._get_model_accuracy(symbol, timeframe)
        
        return SignalObject(
            symbol=symbol,
            timeframe=timeframe,
            signal_type=signal,
            timestamp=datetime.utcnow(),
            entry_price=entry_info['entry_price'],
            take_profit=entry_info['take_profit'],
            stop_loss=entry_info['stop_loss'],
            probability=probability,
            confidence=confidence,
            model_accuracy=model_accuracy,
            analysis=market_analysis,
            risk_reward_ratio=0.0,  # Будет рассчитано в __post_init__
            position_size=entry_info['position_size'],
            commission_cost=entry_info['commission_cost'],
            market_condition=market_analysis.get('market_condition'),
            volatility_level=market_analysis.get('volatility_level'),
            trend_alignment=market_analysis.get('trend')
        )

    # Вспомогательные методы
    def _classify_volatility(self, atr_percent: float) -> str:
        if atr_percent < 0.01:
            return 'low'
        elif atr_percent < 0.025:
            return 'normal'
        elif atr_percent < 0.05:
            return 'high'
        else:
            return 'extreme'

    def _analyze_trend(self, df: pd.DataFrame) -> str:
        if 'ema_50' in df.columns:
            return 'bullish' if df['close'].iloc[-1] > df['ema_50'].iloc[-1] else 'bearish'
        return 'neutral'

    def _analyze_trend_strength(self, df: pd.DataFrame) -> str:
        if len(df) < 20:
            return 'weak'
        
        # Простой анализ на основе EMA
        if 'ema_21' in df.columns and 'ema_50' in df.columns:
            ema_diff = (df['ema_21'].iloc[-1] - df['ema_50'].iloc[-1]) / df['ema_50'].iloc[-1]
            if abs(ema_diff) < 0.01:
                return 'weak'
            elif abs(ema_diff) < 0.03:
                return 'moderate'
            else:
                return 'strong'
        return 'weak'

    def _get_rsi(self, df: pd.DataFrame) -> float:
        if 'rsi_7' in df.columns:
            return df['rsi_7'].iloc[-1]
        elif 'rsi_14' in df.columns:
            return df['rsi_14'].iloc[-1]
        return 50.0

    def _analyze_volume(self, df: pd.DataFrame) -> str:
        if 'volume' in df.columns:
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            ratio = current_volume / avg_volume if avg_volume > 0 else 1
            return 'high' if ratio > 1.5 else 'normal'
        return 'normal'

    def _assess_market_condition(self, analysis: Dict) -> str:
        volatility = analysis.get('volatility_level', 'normal')
        trend_strength = analysis.get('trend_strength', 'weak')
        
        if volatility == 'extreme':
            return 'unfavorable'
        elif volatility in ['normal', 'high'] and trend_strength in ['moderate', 'strong']:
            return 'favorable'
        return 'neutral'

    def _get_multi_timeframe_context(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """Получает контекст с других таймфреймов."""
        context_parts = []
        
        # Анализируем доступные контекстные фичи
        context_features = [col for col in df.columns if any(tf in col for tf in ['1h', '4h', '1d'])]
        
        for feature in context_features:
            if 'trend' in feature and feature in df.columns:
                value = df[feature].iloc[-1]
                tf = feature.split('_')[-1] if '_' in feature else 'unknown'
                context_parts.append(f"{tf}: {'bullish' if value > 0.5 else 'bearish'}")
        
        return '; '.join(context_parts) if context_parts else 'No context'

    def _validate_signal_trend_alignment(self, signal: SignalType, market_analysis: Dict) -> bool:
        trend = market_analysis.get('trend', 'neutral')
        trend_strength = market_analysis.get('trend_strength', 'weak')
        
        if trend_strength == 'weak':
            return True
        
        if trend == 'bullish' and signal == SignalType.LONG:
            return True
        if trend == 'bearish' and signal == SignalType.SHORT:
            return True
        
        return False

    def _apply_slippage(self, price: float, signal: SignalType) -> float:
        if signal == SignalType.LONG:
            return price * (1 + self.slippage)
        else:
            return price * (1 - self.slippage)

    def _calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        risk_amount = self.account_balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        if price_risk == 0:
            return 0
        
        return risk_amount / price_risk

    def _calculate_commission_cost(self, price: float, position_size: float) -> float:
        return price * position_size * self.commission

    def _determine_confidence(self, probability: float, market_analysis: Dict) -> ConfidenceLevel:
        if probability >= 0.8 and market_analysis.get('market_condition') == 'favorable':
            return ConfidenceLevel.HIGH
        elif probability >= 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _get_model_accuracy(self, symbol: str, timeframe: str) -> float:
        """Получает точность модели из отчетов."""
        try:
            import os
            report_path = f"reports/report_{symbol.replace('/', '_')}_{timeframe}.csv"
            if os.path.exists(report_path):
                report_df = pd.read_csv(report_path, index_col=0)
                if 'accuracy' in report_df.index:
                    return report_df.loc['accuracy', 'precision']
        except:
            pass
        return 0.7  # Значение по умолчанию 