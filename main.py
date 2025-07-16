import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import optuna
from sklearn.metrics import roc_auc_score

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

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляем возвращения, волатильность, RSI, MACD, ATR и прочие индикаторы.
    """
    df['return_1h']  = df['close'].pct_change(1)
    df['return_6h']  = df['close'].pct_change(6)
    df['return_12h'] = df['close'].pct_change(12)
    df['return_24h'] = df['close'].pct_change(24)

    df['volatility_6h']  = df['return_1h'].rolling(6).std()
    df['volatility_24h'] = df['return_1h'].rolling(24).std()

    df['range_1h']  = df['high'] - df['low']
    df['range_pct'] = df['range_1h'] / df['open']

    # RSI
    window = 14
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # ATR
    df['tr0'] = df['high'] - df['low']
    df['tr1'] = (df['high'] - df['close'].shift()).abs()
    df['tr2'] = (df['low']  - df['close'].shift()).abs()
    df['TR']  = df[['tr0','tr1','tr2']].max(axis=1)
    df['ATR_14'] = df['TR'].rolling(window).mean()

    # очистка
    df.drop(columns=['tr0','tr1','tr2','TR'], inplace=True)
    df.dropna(inplace=True)
    return df

def walk_forward_cv(model_cls, params: dict, X: pd.DataFrame, y: pd.Series,
                    n_splits: int = 5, test_size: float = 0.2):
    """
    Expanding window / walk‑forward validation.
    Возвращает средний ROC‑AUC и список AUC по каждому сплиту.
    """
    n = len(X)
    test_len = int(n * test_size)
    aucs = []

    # вычисляем границы обучающей и тестовой выборок
    for i in range(1, n_splits + 1):
        train_end = int((n - test_len) * i / n_splits)
        test_start = train_end
        test_end   = test_start + test_len

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_test,  y_test  = X.iloc[test_start:test_end],  y.iloc[test_start:test_end]

        # баланс классов
        scale = (len(y_train) - y_train.sum()) / y_train.sum()
        params['scale_pos_weight'] = scale

        model = model_cls(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )
        y_proba = model.predict_proba(X_test)[:, 1]
        aucs.append(roc_auc_score(y_test, y_proba))

    return np.mean(aucs), aucs

def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'tree_method': 'hist'  # можно заменить на 'gpu_hist', если есть GPU
    }
    mean_auc = walk_forward_cv(xgb.XGBClassifier, params, X, y,
                                          n_splits=5, test_size=0.2)
    return mean_auc

if __name__ == '__main__':
    # 1) Загружаем данные
    symbol, interval = 'BTCUSDT', '1h'
    start = (datetime.utcnow() - timedelta(days=365*3)).strftime('%Y-%m-%d %H:%M:%S')
    df = fetch_all_klines(symbol, interval, start)
    
    # 2) Вычисляем индикаторы
    df = compute_indicators(df)
    
    # 3) Готовим таргет и фичи
    df['y'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    features = [
        'volume','return_1h','return_6h','return_12h','return_24h',
        'volatility_6h','volatility_24h','range_pct',
        'RSI_14','MACD','MACD_signal','ATR_14'
    ]
    X, y = df[features], df['y']
    
    # 4) Оптимизация гиперпараметров
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda t: objective(t, X, y), n_trials=50)
    print('Best params:', study.best_params)
    print('Best mean ROC‑AUC:', study.best_value)
    print('AUCs per fold:', study.user_attrs['fold_aucs'])
    
    # 5) Финальное обучение на всех данных
    best_params = study.best_params.copy()
    best_params.update({
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'tree_method': 'hist'
    })
    # баланс классов на всём датасете
    scale_all = (len(y) - y.sum()) / y.sum()
    best_params['scale_pos_weight'] = scale_all

    model = xgb.XGBClassifier(**best_params)
    model.fit(X, y)
    
    # 6) Важность признаков
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, importance_type='gain', max_num_features=15)
    plt.title('Feature Importance (gain)')
    plt.tight_layout()
    plt.show()
