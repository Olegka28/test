import pandas as pd

def is_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Bullish Engulfing: current candle body engulfs previous body and is bullish.
    """
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    curr_open = df['open']
    curr_close = df['close']
    # previous bearish and current bullish
    cond1 = prev_close < prev_open
    cond2 = curr_close > curr_open
    # engulf body
    cond3 = curr_close > prev_open
    cond4 = curr_open < prev_close
    return (cond1 & cond2 & cond3 & cond4).astype(int)


def is_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Bearish Engulfing: current candle body engulfs previous body and is bearish.
    """
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    curr_open = df['open']
    curr_close = df['close']
    # previous bullish and current bearish
    cond1 = prev_close > prev_open
    cond2 = curr_close < curr_open
    # engulf body
    cond3 = curr_open > prev_close
    cond4 = curr_close < prev_open
    return (cond1 & cond2 & cond3 & cond4).astype(int)


def is_hammer(df: pd.DataFrame) -> pd.Series:
    """
    Hammer: small body at top, long lower shadow (>= 2*body).
    """
    body = (df['close'] - df['open']).abs()
    lower_shadow = (df['open'].where(df['close'] > df['open'], df['close']) - df['low'])
    cond = lower_shadow >= 2 * body
    # small upper shadow
    upper_shadow = (df['high'] - df['close'].where(df['close'] > df['open'], df['open']))
    return (cond & (upper_shadow <= body)).astype(int)


def is_shooting_star(df: pd.DataFrame) -> pd.Series:
    """
    Shooting Star: small body at bottom, long upper shadow (>= 2*body).
    """
    body = (df['close'] - df['open']).abs()
    upper_shadow = (df['high'] - df['open'].where(df['open'] > df['close'], df['close']))
    cond = upper_shadow >= 2 * body
    lower_shadow = (df['open'].where(df['open'] > df['close'], df['close']) - df['low'])
    return (cond & (lower_shadow <= body)).astype(int)


def is_doji(df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
    """
    Doji: body size is very small relative to range.
    threshold: fraction of (high-low) that defines small body.
    """
    body = (df['close'] - df['open']).abs()
    range_ = (df['high'] - df['low'])
    return (body <= threshold * range_).astype(int)


def compute_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute and attach candle pattern features as binary columns.
    """
    df = df.copy()
    df['Bullish_Engulfing'] = is_bullish_engulfing(df)
    df['Bearish_Engulfing'] = is_bearish_engulfing(df)
    df['Hammer'] = is_hammer(df)
    df['Shooting_Star'] = is_shooting_star(df)
    df['Doji'] = is_doji(df)
    return df
