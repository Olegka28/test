import asyncio
import os
import sys
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings
from src.data.exchange_client import ExchangeClient
from src.data.data_repository import DataRepository
from src.ml.data_builder import DataBuilder
from src.ml.model_manager import ModelManager

PAIR = "BTC/USDT"
TIMEFRAME = "1h"
THRESHOLD = 0.6
TP_PCT = 0.01  # 1%
SL_PCT = 0.01  # 1%

async def backtest():
    print(f"\n--- Backtesting {PAIR} {TIMEFRAME} ---")
    exchange_client = ExchangeClient(settings.exchange)
    data_repo = DataRepository(exchange_client)
    data_builder = DataBuilder(data_repo)
    model_manager = ModelManager(settings.models)

    # 1. Get config and data range (6 месяцев)
    tf_config = settings.trading.timeframes[TIMEFRAME]
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=180)

    # 2. Build dataset
    dataset = await data_builder.build(
        symbol=PAIR,
        timeframe=TIMEFRAME,
        tf_config=tf_config,
        start_date=start_date,
        end_date=end_date
    )
    if not dataset:
        print(f"Could not build dataset for {PAIR} on {TIMEFRAME}. Exiting.")
        await exchange_client.close()
        return

    X, y = dataset
    # y: -1=Short, 1=Long

    # 3. Load model
    model = model_manager.load_model(PAIR, TIMEFRAME)
    features_ref = model_manager.load_features(PAIR, TIMEFRAME)
    if model is None or features_ref is None:
        print(f"No model or features found for {PAIR} {TIMEFRAME}")
        await exchange_client.close()
        return

    # 4. Align features
    missing = [f for f in features_ref if f not in X.columns]
    for f in missing:
        X[f] = 0
    extra = [f for f in X.columns if f not in features_ref]
    if extra:
        X = X.drop(columns=extra)
    X = X[features_ref]  # Ensure order

    # 5. Predict proba
    y_proba = model.predict_proba(X)
    y_pred = model.predict(X)
    # Вероятность выбранного класса
    pred_proba = y_proba[np.arange(len(y_pred)), y_pred]

    # 6. Симуляция сделок
    df = X.copy()
    df['signal'] = y_pred
    df['proba'] = pred_proba
    df['open'] = y.index  # индексы совпадают с исходным df, это timestamps
    df['true_target'] = y.values
    # Для симуляции TP/SL нужны close, high, low — их можно получить из исходных данных
    # Получим OHLCV за тот же период
    ohlc_dataset = await data_repo.get_historical_data(
        symbol=PAIR,
        timeframe=TIMEFRAME,
        start_date=start_date,
        end_date=end_date
    )
    if ohlc_dataset is None or ohlc_dataset.empty:
        print("No OHLCV data for backtest.")
        await exchange_client.close()
        return
    # Совмещаем по индексу (timestamp)
    # Приводим индексы к UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    if ohlc_dataset.index.tz is None:
        ohlc_dataset.index = ohlc_dataset.index.tz_localize('UTC')
    else:
        ohlc_dataset.index = ohlc_dataset.index.tz_convert('UTC')

    df = df.join(ohlc_dataset[['close', 'high', 'low']], how='left')
    df = df.dropna(subset=['close', 'high', 'low'])

    trades = []
    for idx, row in df.iterrows():
        if row['proba'] < THRESHOLD:
            continue
        direction = row['signal']
        entry_price = row['close']
        # Симулируем TP/SL на следующих барах (до конца prediction_horizon)
        horizon = tf_config.prediction_candles
        future = df.loc[idx:].iloc[1:horizon+1]  # следующие N баров
        tp_price = entry_price * (1 + TP_PCT) if direction == 1 else entry_price * (1 - TP_PCT)
        sl_price = entry_price * (1 - SL_PCT) if direction == 1 else entry_price * (1 + SL_PCT)
        exit_price = entry_price
        result = 0
        exit_type = 'none'
        for _, fut in future.iterrows():
            if direction == 1:
                if fut['high'] >= tp_price:
                    exit_price = tp_price
                    result = TP_PCT
                    exit_type = 'tp'
                    break
                if fut['low'] <= sl_price:
                    exit_price = sl_price
                    result = -SL_PCT
                    exit_type = 'sl'
                    break
            else:
                if fut['low'] <= tp_price:
                    exit_price = tp_price
                    result = TP_PCT
                    exit_type = 'tp'
                    break
                if fut['high'] >= sl_price:
                    exit_price = sl_price
                    result = -SL_PCT
                    exit_type = 'sl'
                    break
        else:
            # Если ни TP ни SL не сработал — закрываем по последней цене
            last = future.iloc[-1] if not future.empty else row
            exit_price = last['close']
            result = (exit_price - entry_price) / entry_price * direction
            exit_type = 'close'
        trades.append({
            'open_time': idx,
            'direction': 'long' if direction == 1 else 'short',
            'entry': entry_price,
            'exit': exit_price,
            'result': result,
            'exit_type': exit_type,
            'proba': row['proba'],
        })

    await exchange_client.close()

    # 7. Подсчет метрик
    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("No trades triggered by the model.")
        return
    total_profit = trades_df['result'].sum()
    winrate = (trades_df['result'] > 0).mean()
    n_trades = len(trades_df)
    max_drawdown = (trades_df['result'].cumsum().cummax() - trades_df['result'].cumsum()).max()
    print(f"\nBacktest results for {PAIR} {TIMEFRAME} (6 months):")
    print(f"Total trades: {n_trades}")
    print(f"Winrate: {winrate:.2%}")
    print(f"Total profit: {total_profit:.2%}")
    print(f"Max drawdown: {max_drawdown:.2%}")
    
    # Раздельная статистика по типам позиций
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
    
    print(f"\nSample trades:")
    print(trades_df[['open_time', 'direction', 'entry', 'exit', 'result', 'exit_type', 'proba']].head(10))

if __name__ == "__main__":
    asyncio.run(backtest()) 