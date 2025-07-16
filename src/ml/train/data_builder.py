import os
import pandas as pd
from typing import List

from src.data.repository.ohlcv_repository import OHLCVRepository
from src.features.processor import compute_all_features
from src.config.setting import settings

# Function to build dataset for training ML models
def build_dataset(symbol: str, timeframes: List[str], history_limit: int = 1000,
                  target_horizon: int = 3, target_threshold: float = 0.005) -> pd.DataFrame:
    """
    Build a dataset for ML training by loading OHLCV data, computing features,
    and labeling target for each timeframe.

    Args:
        symbol (str): Trading pair, e.g. 'BTC/USDT'
        timeframes (List[str]): List of timeframes to include, e.g. ['15m']
        history_limit (int): Number of bars to fetch for each timeframe
        target_horizon (int): Number of bars ahead to compute target
        target_threshold (float): Fractional price move threshold for target labeling

    Returns:
        pd.DataFrame: Dataset with features and 'target' column (1 for LONG, 0 for SHORT)
    """
    df_list: List[pd.DataFrame] = []

    # For simplicity use single timeframe per dataset for now
    for tf in timeframes:
        repo = OHLCVRepository(
            data_dir=settings.DATA_DIR,
            exchange='bybit',  # Change to 'binance' if needed
            api_key=settings.BYBIT_API_KEY,
            secret=settings.BYBIT_API_SECRET
        )
        df_ohlcv = repo.load(symbol, tf, limit=history_limit)
        df_feat = compute_all_features(df_ohlcv)
    
        # Compute future return
        df_feat['future_close'] = df_feat['close'].shift(-target_horizon)
        df_feat['return'] = (df_feat['future_close'] - df_feat['close']) / df_feat['close']
        # Label target: LONG if return >= threshold; SHORT if <= -threshold; else drop
        df_feat['target'] = 0
        df_feat.loc[df_feat['return'] >= target_threshold, 'target'] = 1
        df_feat = df_feat[df_feat['return'].abs() >= target_threshold]

        # Add timeframe identifier as feature
        df_feat['timeframe'] = tf
        df_list.append(df_feat)

    # Concatenate all
    dataset = pd.concat(df_list, axis=0)
    # One-hot encode timeframe
    dataset = pd.get_dummies(dataset, columns=['timeframe'], prefix='tf')
    # Drop intermediate columns
    dataset = dataset.drop(columns=['future_close', 'return'])
    dataset = dataset.dropna()
    return dataset


if __name__ == "__main__":
    # Example usage
    df = build_dataset('BTC/USDT', ['15m'], history_limit=5000)
    print(df.shape)
    print(df['target'].value_counts())
    df.to_csv(os.path.join(settings.MODEL_DIR, 'training_dataset.csv'))
