import pandas as pd

from src.features.ta_features import compute_ta_features
from src.features.pattern_features import compute_pattern_features


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all feature sets: technical indicators and candlestick patterns.

    Args:
        df (pd.DataFrame): OHLCV DataFrame with columns:
            ['open', 'high', 'low', 'close', 'volume'] indexed by datetime.

    Returns:
        pd.DataFrame: DataFrame augmented with all feature columns.
    """
    # Calculate TA indicators
    df_ta = compute_ta_features(df)
    # Calculate candlestick pattern features
    df_patterns = compute_pattern_features(df_ta)
    # Return combined DataFrame
    return df_patterns
