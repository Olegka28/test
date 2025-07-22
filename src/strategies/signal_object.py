from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime
from enum import Enum


class SignalType(Enum):
    LONG = "long"
    SHORT = "short"


class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class SignalObject:
    """
    Структурированный объект торгового сигнала для отправки в Telegram.
    """
    
    # Основная информация
    symbol: str
    timeframe: str
    signal_type: SignalType
    timestamp: datetime
    
    # Цены
    entry_price: float
    take_profit: float
    stop_loss: float
    
    # Модель и уверенность
    probability: float
    confidence: ConfidenceLevel
    model_accuracy: float
    
    # Анализ рынка
    analysis: Dict
    
    # Риск/прибыль
    risk_reward_ratio: float
    
    # Дополнительная информация
    position_size: Optional[float] = None
    commission_cost: Optional[float] = None
    market_condition: Optional[str] = None
    volatility_level: Optional[str] = None
    trend_alignment: Optional[str] = None
    is_autoscan: bool = False  # Новый флаг
    
    def __post_init__(self):
        """Валидация и расчет дополнительных полей."""
        if self.probability < 0 or self.probability > 1:
            raise ValueError("Probability must be between 0 and 1")
        
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
        
        # Рассчитываем risk/reward ratio
        if self.signal_type == SignalType.LONG:
            profit = self.take_profit - self.entry_price
            risk = self.entry_price - self.stop_loss
        else:  # SHORT
            profit = self.entry_price - self.take_profit
            risk = self.stop_loss - self.entry_price
        
        if risk > 0:
            self.risk_reward_ratio = profit / risk
        else:
            self.risk_reward_ratio = 0.0
    
    def to_dict(self) -> Dict:
        """Конвертирует объект в словарь для JSON сериализации."""
        d = {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'signal_type': self.signal_type.value,
            'timestamp': self.timestamp.isoformat(),
            'entry_price': self.entry_price,
            'take_profit': self.take_profit,
            'stop_loss': self.stop_loss,
            'probability': self.probability,
            'confidence': self.confidence.value,
            'model_accuracy': self.model_accuracy,
            'analysis': self.analysis,
            'risk_reward_ratio': self.risk_reward_ratio,
            'position_size': self.position_size,
            'commission_cost': self.commission_cost,
            'market_condition': self.market_condition,
            'volatility_level': self.volatility_level,
            'trend_alignment': self.trend_alignment,
            'is_autoscan': self.is_autoscan,
        }
        return d
    
    def to_telegram_message(self, postfix: Optional[str] = None) -> str:
        signal_emoji = "🟢" if self.signal_type == SignalType.LONG else "🔴"
        confidence_emoji = {
            ConfidenceLevel.LOW: "🟡",
            ConfidenceLevel.MEDIUM: "🟠",
            ConfidenceLevel.HIGH: "🟢"
        }[self.confidence]
        autoscan_note = "\n🤖 _Сигнал с автосканирования_" if self.is_autoscan else ""
        message = (
            f"{signal_emoji} **{self.symbol} {self.timeframe} {self.signal_type.value.upper()}**\n\n"
            f"💰 **Entry:** ${self.entry_price:,.2f}\n"
            f"🎯 **Take Profit:** ${self.take_profit:,.2f}\n"
            f"🛑 **Stop Loss:** ${self.stop_loss:,.2f}\n\n"
            f"📊 **Probability:** {self.probability:.1%}\n"
            f"{confidence_emoji} **Confidence:** {self.confidence.value.upper()}\n"
            f"⚖️ **Risk/Reward:** {self.risk_reward_ratio:.2f}\n\n"
            f"📈 **Analysis:**\n"
            f"• Trend: {self.analysis.get('trend', 'N/A')}\n"
            f"• Volatility: {self.analysis.get('volatility_level', 'N/A')}\n"
            f"• Volume: {self.analysis.get('volume_signal', 'N/A')}\n"
            f"• Market: {self.analysis.get('market_condition', 'N/A')}\n\n"
            f"🎯 **Model Accuracy:** {self.model_accuracy:.1%}\n"
            f"⏰ **Time:** {self.timestamp.strftime('%Y-%m-%d %H:%M UTC')}"
            f"{autoscan_note}"
        )
        if postfix:
            message += f"\n{postfix}"
        return message.strip()
    
    def is_valid(self) -> bool:
        """Проверяет валидность сигнала."""
        return (
            self.probability >= 0.6 and  # Минимальная вероятность
            self.risk_reward_ratio >= 1.5 and  # Минимальный R/R
            self.confidence != ConfidenceLevel.LOW and  # Не низкая уверенность
            self.market_condition != 'unfavorable'  # Не неблагоприятные условия
        )
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SignalObject':
        """Создает объект из словаря."""
        return cls(
            symbol=data['symbol'],
            timeframe=data['timeframe'],
            signal_type=SignalType(data['signal_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            entry_price=data['entry_price'],
            take_profit=data['take_profit'],
            stop_loss=data['stop_loss'],
            probability=data['probability'],
            confidence=ConfidenceLevel(data['confidence']),
            model_accuracy=data['model_accuracy'],
            analysis=data['analysis'],
            risk_reward_ratio=data['risk_reward_ratio'],
            position_size=data.get('position_size'),
            commission_cost=data.get('commission_cost'),
            market_condition=data.get('market_condition'),
            volatility_level=data.get('volatility_level'),
            trend_alignment=data.get('trend_alignment'),
            is_autoscan=data.get('is_autoscan', False)
        ) 