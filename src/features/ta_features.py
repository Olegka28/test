import pandas as pd
import numpy as np

# Monkey-patch numpy to ensure pandas_ta finds NaN attribute
setattr(np, 'NaN', np.nan)
import pandas_ta as ta


def compute_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute common technical analysis features on OHLCV DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with columns ['open','high','low','close','volume'] indexed by datetime.

    Returns:
        pd.DataFrame: Input DataFrame augmented with TA features.
    """
    df = df.copy()
    # Exponential Moving Averages
    df['EMA_9'] = ta.ema(df['close'], length=9)
    df['EMA_21'] = ta.ema(df['close'], length=21)
    df['EMA_50'] = ta.ema(df['close'], length=50)

    # Simple Moving Average for reference
    df['SMA_20'] = ta.sma(df['close'], length=20)

    # Momentum indicators
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_signal'] = macd['MACDs_12_26_9']
    df['MACD_hist'] = macd['MACDh_12_26_9']

    # Volatility indicators
    df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    bb = ta.bbands(df['close'], length=20, std=2)
    df['BB_upper'] = bb['BBU_20_2.0']
    df['BB_lower'] = bb['BBL_20_2.0']
    df['BB_width'] = df['BB_upper'] - df['BB_lower']

    # Trend strength
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    df['ADX_14'] = adx['ADX_14']
    df['DMI_plus'] = adx['DMP_14']
    df['DMI_minus'] = adx['DMN_14']

    # Drop rows with NaN from indicator warm-up
    return df.dropna()
