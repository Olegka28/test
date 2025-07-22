import asyncio
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Optional, Dict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings
from src.data.exchange_client import ExchangeClient
from src.data.data_repository import DataRepository
from src.ml.data_builder import DataBuilder
from src.strategies.base_strategy import BaseStrategy
from src.strategies.signal_object import SignalObject, SignalType

class BacktestWithStrategy:
    """
    Бэктест с использованием универсальной стратегии.
    """

    def __init__(self, strategy: BaseStrategy):
        self.strategy = strategy
        self.exchange_client = ExchangeClient(settings.exchange)
        self.data_repo = DataRepository(self.exchange_client)
        self.data_builder = DataBuilder(self.data_repo)

    async def run_backtest(self, 
                          symbol: str, 
                          timeframe: str, 
                          days: int = 180) -> pd.DataFrame:
        """
        Запускает бэктест для конкретной модели.
        
        Args:
            symbol: Торговая пара
            timeframe: Таймфрейм
            days: Количество дней для бэктеста
            
        Returns:
            DataFrame с результатами сделок
        """
        print(f"\n--- Backtesting {symbol} {timeframe} with strategy ---")
        
        # 1. Получаем данные
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        tf_config = settings.trading.timeframes[timeframe]
        dataset = await self.data_builder.build(
            symbol=symbol,
            timeframe=timeframe,
            tf_config=tf_config,
            start_date=start_date,
            end_date=end_date
        )
        
        if not dataset:
            print(f"Could not build dataset for {symbol} {timeframe}")
            await self.exchange_client.close()
            return pd.DataFrame()
        
        X, y = dataset
        
        # 2. Получаем OHLCV данные для симуляции сделок
        ohlc_data = await self.data_repo.get_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if ohlc_data is None or ohlc_data.empty:
            print("No OHLCV data for backtest")
            await self.exchange_client.close()
            return pd.DataFrame()
        
        # 3. Совмещаем данные
        df = X.copy()
        df = df.join(ohlc_data[['close', 'high', 'low']], how='left')
        df = df.dropna(subset=['close', 'high', 'low'])
        
        # 4. Симулируем торговлю
        trades = []
        for i in range(len(df) - 1):  # -1 чтобы не брать последний бар
            current_data = df.iloc[:i+1]  # Данные до текущего бара
            
            # Анализируем модель
            signal_obj = self.strategy.analyze_model(symbol, timeframe, current_data)
            
            if signal_obj is None or not signal_obj.is_valid():
                continue
            
            # Симулируем сделку
            trade_result = self._simulate_trade(signal_obj, df.iloc[i+1:], tf_config.prediction_candles)
            if trade_result:
                trades.append(trade_result)
        
        await self.exchange_client.close()
        
        # 5. Анализируем результаты
        if trades:
            trades_df = pd.DataFrame(trades)
            self._print_backtest_results(symbol, timeframe, trades_df)
            return trades_df
        else:
            print("No valid trades found")
            return pd.DataFrame()

    async def close(self):
        await self.exchange_client.close()

    def _simulate_trade(self, 
                       signal_obj: SignalObject, 
                       future_data: pd.DataFrame, 
                       max_bars: int) -> Optional[Dict]:
        """
        Симулирует одну сделку.
        """
        entry_price = signal_obj.entry_price
        take_profit = signal_obj.take_profit
        stop_loss = signal_obj.stop_loss
        direction = signal_obj.signal_type
        
        # Ограничиваем количество баров для анализа
        future_data = future_data.head(max_bars)
        
        exit_price = entry_price
        result = 0
        exit_type = 'close'
        exit_bar = None
        
        for idx, bar in future_data.iterrows():
            if direction == SignalType.LONG:
                if bar['high'] >= take_profit:
                    exit_price = take_profit
                    result = (take_profit - entry_price) / entry_price
                    exit_type = 'tp'
                    exit_bar = idx
                    break
                if bar['low'] <= stop_loss:
                    exit_price = stop_loss
                    result = (stop_loss - entry_price) / entry_price
                    exit_type = 'sl'
                    exit_bar = idx
                    break
            else:  # SHORT
                if bar['low'] <= take_profit:
                    exit_price = take_profit
                    result = (entry_price - take_profit) / entry_price
                    exit_type = 'tp'
                    exit_bar = idx
                    break
                if bar['high'] >= stop_loss:
                    exit_price = stop_loss
                    result = (entry_price - stop_loss) / entry_price
                    exit_type = 'sl'
                    exit_bar = idx
                    break
        else:
            # Закрываем по последней цене
            last_bar = future_data.iloc[-1]
            exit_price = last_bar['close']
            if direction == SignalType.LONG:
                result = (exit_price - entry_price) / entry_price
            else:
                result = (entry_price - exit_price) / entry_price
            exit_bar = future_data.index[-1]
        
        # Учитываем комиссии
        commission_cost = signal_obj.commission_cost or 0
        net_result = result - commission_cost
        
        return {
            'open_time': signal_obj.timestamp,
            'close_time': exit_bar,
            'direction': signal_obj.signal_type.value,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'result': net_result,
            'exit_type': exit_type,
            'probability': signal_obj.probability,
            'confidence': signal_obj.confidence.value,
            'risk_reward_ratio': signal_obj.risk_reward_ratio,
            'market_condition': signal_obj.market_condition,
            'volatility_level': signal_obj.volatility_level,
            'trend_alignment': signal_obj.trend_alignment,
            'position_size': signal_obj.position_size,
            'commission_cost': commission_cost
        }

    def _print_backtest_results(self, symbol: str, timeframe: str, trades_df: pd.DataFrame):
        """Выводит результаты бэктеста."""
        total_trades = len(trades_df)
        if total_trades == 0:
            print("No trades executed")
            return
        
        # Общая статистика
        total_profit = trades_df['result'].sum()
        winrate = (trades_df['result'] > 0).mean()
        avg_result = trades_df['result'].mean()
        max_drawdown = self._calculate_max_drawdown(trades_df['result'])
        
        print(f"\nBacktest Results for {symbol} {timeframe}:")
        print(f"Total trades: {total_trades}")
        print(f"Winrate: {winrate:.2%}")
        print(f"Total profit: {total_profit:.2%}")
        print(f"Average result: {avg_result:.2%}")
        print(f"Max drawdown: {max_drawdown:.2%}")
        
        # Статистика по типам позиций
        long_trades = trades_df[trades_df['direction'] == 'long']
        short_trades = trades_df[trades_df['direction'] == 'short']
        
        print(f"\nLong positions:")
        print(f"  Count: {len(long_trades)}")
        if len(long_trades) > 0:
            print(f"  Winrate: {(long_trades['result'] > 0).mean():.2%}")
            print(f"  Profit: {long_trades['result'].sum():.2%}")
            print(f"  Avg result: {long_trades['result'].mean():.2%}")
        
        print(f"\nShort positions:")
        print(f"  Count: {len(short_trades)}")
        if len(short_trades) > 0:
            print(f"  Winrate: {(short_trades['result'] > 0).mean():.2%}")
            print(f"  Profit: {short_trades['result'].sum():.2%}")
            print(f"  Avg result: {short_trades['result'].mean():.2%}")
        
        # Статистика по уверенности
        print(f"\nBy confidence level:")
        for confidence in ['low', 'medium', 'high']:
            conf_trades = trades_df[trades_df['confidence'] == confidence]
            if len(conf_trades) > 0:
                print(f"  {confidence.upper()}: {len(conf_trades)} trades, "
                      f"winrate: {(conf_trades['result'] > 0).mean():.2%}, "
                      f"profit: {conf_trades['result'].sum():.2%}")
        
        # Статистика по рыночным условиям
        print(f"\nBy market condition:")
        for condition in ['favorable', 'neutral', 'unfavorable']:
            cond_trades = trades_df[trades_df['market_condition'] == condition]
            if len(cond_trades) > 0:
                print(f"  {condition.upper()}: {len(cond_trades)} trades, "
                      f"winrate: {(cond_trades['result'] > 0).mean():.2%}, "
                      f"profit: {cond_trades['result'].sum():.2%}")

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Рассчитывает максимальную просадку."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

async def main():
    """Запускает бэктест для всех моделей."""
    # Создаем стратегию
    strategy = BaseStrategy(
        min_proba_threshold=0.6,
        risk_per_trade=0.02,
        commission=0.0004,
        slippage=0.0001,
        account_balance=10000
    )
    
    # Создаем бэктестер
    backtester = BacktestWithStrategy(strategy)
    
    # Запускаем бэктест для всех пар и таймфреймов
    all_results = []
    
    for symbol in settings.trading.pairs:
        for timeframe in settings.trading.timeframes.keys():
            try:
                results = await backtester.run_backtest(symbol, timeframe, days=180)
                if not results.empty:
                    results['symbol'] = symbol
                    results['timeframe'] = timeframe
                    all_results.append(results)
            except Exception as e:
                print(f"Error backtesting {symbol} {timeframe}: {e}")
    
    # Сохраняем общие результаты
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv('reports/backtest_results.csv', index=False)
        print(f"\nSaved combined results to reports/backtest_results.csv")
        
        # Общая статистика
        print(f"\nOverall Results:")
        print(f"Total trades: {len(combined_results)}")
        print(f"Overall winrate: {(combined_results['result'] > 0).mean():.2%}")
        print(f"Overall profit: {combined_results['result'].sum():.2%}")
    await backtester.close()

if __name__ == "__main__":
    asyncio.run(main()) 