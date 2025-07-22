import pandas as pd
import numpy as np


class TargetGenerator:
    """
    Generates the target variable for the machine learning model.
    """

    def __init__(self, df: pd.DataFrame, prediction_candles: int, price_change_threshold: float = 0.01):
        self.df = df.copy()
        self.prediction_candles = prediction_candles
        self.price_change_threshold = price_change_threshold

    def _calculate_future_returns(self):
        """
        Calculates the max upward and downward price movement in the future window.
        """
        future_highs = self.df['high'].rolling(window=self.prediction_candles).max().shift(-self.prediction_candles)
        future_lows = self.df['low'].rolling(window=self.prediction_candles).min().shift(-self.prediction_candles)

        self.df['future_max_return'] = (future_highs - self.df['close']) / self.df['close']
        self.df['future_min_return'] = (future_lows - self.df['close']) / self.df['close']

    def generate_target(self) -> pd.DataFrame:
        """
        Creates the target variable based on which threshold is hit first.
        -  1: Long signal (price goes up by threshold)
        - -1: Short signal (price goes down by threshold)
        -  0: No signal (neither threshold is met)
        """
        print(f"Generating target with {self.prediction_candles} lookahead candles and {self.price_change_threshold:.2%} threshold...")
        
        self._calculate_future_returns()

        long_condition = self.df['future_max_return'] >= self.price_change_threshold
        short_condition = self.df['future_min_return'] <= -self.price_change_threshold

        # Assign signals based on conditions. Default to 0 (no signal).
        self.df['target'] = 0
        self.df.loc[long_condition, 'target'] = 1
        self.df.loc[short_condition, 'target'] = -1

        # This part is tricky: what if both thresholds are met?
        # A simple approach is to see which one was likely hit "first".
        # We can approximate this by comparing the absolute returns.
        # If |min_return| < |max_return|, it's more likely the short target was hit first.
        both_hit = long_condition & short_condition
        
        # When both are hit, if the path to the low is "shorter" than the path to the high
        self.df.loc[both_hit & (abs(self.df['future_min_return']) < self.df['future_max_return']), 'target'] = -1
        # When both are hit, if the path to the high is "shorter" or equal
        self.df.loc[both_hit & (abs(self.df['future_min_return']) >= self.df['future_max_return']), 'target'] = 1

        # Clean up columns and NaN values
        self.df.drop(columns=['future_max_return', 'future_min_return'], inplace=True)
        self.df.dropna(subset=['target'], inplace=True)
        self.df['target'] = self.df['target'].astype(int)

        print("Target generation complete.")
        return self.df 