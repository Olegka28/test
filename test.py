import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend     import MACD
from ta.volume    import VolumeWeightedAveragePrice
from ta.volatility import BollingerBands
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import time
from datetime import datetime, timedelta
import requests
import os
import matplotlib.pyplot as plt


# 1. Fetch historical data from Coinbase over the last 5 years (paginated)
def fetch_all_klines(symbol: str,
                     interval: str,
                     start_str: str,
                     end_str: str = None,
                     limit: int = 1000,
                     sleep: float = 0.1) -> pd.DataFrame:
    """
    Загружает все свечи для пары symbol с интервалом interval,
    начиная с момента start_str до end_str (или до сейчас).
    """
    url = 'https://api.binance.com/api/v3/klines'
    cols = ['open_time','open','high','low','close','volume',
            'close_time','quote_asset_volume','trades',
            'taker_buy_base','taker_buy_quote','ignore']
    
    start_ts = int(pd.to_datetime(start_str).timestamp() * 1000)
    end_ts   = int(pd.to_datetime(end_str).timestamp() * 1000) if end_str else None

    all_data = []
    while True:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
            'startTime': start_ts
        }
        if end_ts:
            params['endTime'] = end_ts

        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        all_data.extend(data)
        last_open = data[-1][0]
        start_ts = last_open + 1

        if len(data) < limit:
            break

        time.sleep(sleep)

    df = pd.DataFrame(all_data, columns=cols)
    df['open_time']  = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    df.reset_index(drop=True, inplace=True)
    return df


# 2. Compute indicators and label targets (1=long, -1=short, 0=no signal)
def prepare_data(df, window_size=50, horizon=24, threshold=0.01):
    df = df.copy()
    # Technical indicators
    df['rsi'] = RSIIndicator(df['close']).rsi()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    df['vwap'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume']).volume_weighted_average_price()   
    df['bollinger_hband'] = BollingerBands(df['close']).bollinger_hband()
    df['bollinger_lband'] = BollingerBands(df['close']).bollinger_lband()
    df['bollinger_mavg'] = BollingerBands(df['close']).bollinger_mavg()
    df['bollinger_wband'] = BollingerBands(df['close']).bollinger_wband()
    
    # Паттерны свечей 
    df['doji'] = (df['high'] - df['low']) / df['close'] < 0.01
    df['hammer'] = (df['close'] - df['low']) / (df['high'] - df['low']) > 0.6
    df['shooting_star'] = (df['high'] - df['close']) / (df['high'] - df['low']) > 0.6   
    df['engulfing'] = (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
    df['spinning_top'] = (df['high'] - df['low']) / df['close'] < 0.05
    df['hanging_man'] = (df['close'] - df['low']) / (df['high'] - df['low']) > 0.6


    df.dropna(inplace=True)

    prices = df['close'].values
    features = ['close', 'rsi', 'macd', 'macd_signal', 'macd_diff',
                'vwap', 'bollinger_hband', 'bollinger_lband',
                'bollinger_mavg', 'bollinger_wband', 'doji',
                'hammer', 'shooting_star', 'engulfing',
                'spinning_top', 'hanging_man']
    data = df[features].values
    X, y = [], []

    for i in range(window_size, len(prices) - horizon):
        window = data[i-window_size:i].flatten()
        future = prices[i+1:i+1+horizon]
        curr_price = prices[i]
        # Check long and short thresholds
        long_target = curr_price * (1 + threshold)
        short_target = curr_price * (1 - threshold)
        if np.max(future) >= long_target:
            label = 1
        elif np.min(future) <= short_target:
            label = -1
        else:
            label = 0
        X.append(window)
        y.append(label)

    return np.array(X), np.array(y)

def train_and_save_models(symbol, timeframes, start_date, window_size=50, horizon=24, threshold=0.01, test_size=0.2):
    os.makedirs('models', exist_ok=True)

    for tf in timeframes:
        print(f"\n=== Processing timeframe: {tf} ===")
        df = fetch_all_klines(symbol, interval=tf, start_str=start_date)
        print(f"Raw candles: {len(df)}, From {df.index[0]} to {df.index[-1]}")

        X, y = prepare_data(df, window_size, horizon, threshold)
        mask = y != 0
        X, y = X[mask], y[mask]
        y = (y == 1).astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        print(f"Samples: total={len(y)}, train={len(y_train)}, test={len(y_test)}")

        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='error',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8
        )
        model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

        # Evaluate
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Test Accuracy ({tf}): {acc:.4f}")

        # Save model
        model_path = f"models/xgb_{symbol.replace('/','')}_{tf}.json"
        model.save_model(model_path)
        print(f"Model saved to {model_path}")

        # 6) Важность признаков
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(model, importance_type='gain', max_num_features=15)
        plt.title('Feature Importance (gain)')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    symbol = 'ETHUSDT'
    timeframes = ['30m', '1h', '4h']
    start_date = (datetime.utcnow() - timedelta(days=365*3)).strftime('%Y-%m-%d %H:%M:%S')

    train_and_save_models(symbol, timeframes, start_date)
