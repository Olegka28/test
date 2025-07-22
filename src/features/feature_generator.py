import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Any


class FeatureGenerator:
    """
    Generates features for a given timeframe using a flexible configuration,
    including context from other timeframes.
    """

    def __init__(self, dataframes: Dict[str, pd.DataFrame], base_timeframe: str, config: Dict[str, Any]):
        """
        Args:
            dataframes (Dict[str, pd.DataFrame]): A dictionary of dataframes, keyed by timeframe string (e.g., "15m").
            base_timeframe (str): The primary timeframe for which to generate features.
            config (Dict[str, Any]): The feature configuration for the base_timeframe.
        """
        self.dataframes = {tf: self._synchronize_timezones(df) for tf, df in dataframes.items()}
        self.base_timeframe = base_timeframe
        self.config = config
        self.base_df = self.dataframes[base_timeframe]

        # Mapping from config strings to feature calculation methods
        self.feature_mapping = {
            "ema_5": lambda df: df.ta.ema(length=5),
            "ema_9": lambda df: df.ta.ema(length=9),
            "ema_21": lambda df: df.ta.ema(length=21),
            "ema_50": lambda df: df.ta.ema(length=50),
            "ema_200": lambda df: df.ta.ema(length=200),
            "ema_50_trend": lambda df: self._calculate_trend(df, "ema", 50),
            "ema_100_trend": lambda df: self._calculate_trend(df, "ema", 100),
            "ema_200_trend": lambda df: self._calculate_trend(df, "ema", 200),
            "rsi_7": lambda df: df.ta.rsi(length=7),
            "rsi_14": lambda df: df.ta.rsi(length=14),
            "macd": lambda df: df.ta.macd(fast=12, slow=26, signal=9),
            "macd_hist": lambda df: df.ta.macd(fast=12, slow=26, signal=9).iloc[:, 2], # MACD Hist is the 3rd col
            "obv": lambda df: df.ta.obv(),
            "volume_change": self._calculate_volume_change,
            "volume_spike": self._calculate_volume_spike,
            "cdl_all": self._calculate_custom_candle_patterns,
            "atr_14": lambda df: df.ta.atr(length=14),
            "bbands_20_2_width": self._calculate_bb_width,
            "adx_14": lambda df: df.ta.adx(length=14),
            "supertrend_7_3": lambda df: df.ta.supertrend(period=7, multiplier=3),
            "supertrend_10_3": lambda df: df.ta.supertrend(period=10, multiplier=3),
            "momentum_10": lambda df: df.ta.mom(length=10),
            "time_features": self._add_time_features,
            "market_regime": self._calculate_market_regime,
            "volatility_30": lambda df: df['close'].rolling(window=30).std(),
            "seasonality": self._add_seasonality_features,
        }

    def generate_features(self) -> pd.DataFrame:
        """
        Main method to orchestrate feature generation for base and context timeframes.
        """
        print(f"--- Generating features for {self.base_timeframe} ---")
        print(f"  [LOG] base_df shape before features: {self.base_df.shape}")

        # 1. Generate base features
        base_features = self.config.get("base", [])
        self._calculate_features_for_df(self.base_df, base_features, self.base_timeframe)
        print(f"  [LOG] base_df shape after base features: {self.base_df.shape}")

        # 2. Generate and merge context features
        context_configs = self.config.get("context", {})
        for context_tf, context_features in context_configs.items():
            if context_tf in self.dataframes:
                print(f"Generating context features from {context_tf}...")
                context_df = self.dataframes[context_tf]
                self._calculate_features_for_df(context_df, context_features, context_tf)
                print(f"  - Before merge: base_df.shape={self.base_df.shape}, context_df.shape={context_df.shape}")
                self._merge_context_features(context_df, context_tf)
                print(f"  - After merge: base_df.shape={self.base_df.shape}")
                print(f"  - base_df index range: {self.base_df.index.min()} to {self.base_df.index.max()}")
            else:
                print(f"Warning: Context timeframe {context_tf} not found in provided dataframes.")
        
        print(f"  [LOG] base_df shape after all merges: {self.base_df.shape}")
        # Не удаляем все NaN! Только базовые фичи и target потом
        return self.base_df

    def _calculate_features_for_df(self, df: pd.DataFrame, features: List[str], timeframe: str):
        """Calculates a list of features on a given dataframe."""
        for feature_name in features:
            if feature_name in self.feature_mapping:
                # Store original columns to find the new ones
                original_cols = set(df.columns)
                result = self.feature_mapping[feature_name](df)

                # Assign the result back to the dataframe
                if isinstance(result, pd.Series):
                    # Ensure the series has a name before assignment
                    result.name = result.name or feature_name
                    df[result.name] = result
                elif isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        df[col] = result[col]

                # Rename new columns to include timeframe context
                new_cols = set(df.columns) - original_cols
                rename_dict = {col: f"{col}_{timeframe}" for col in new_cols}
                df.rename(columns=rename_dict, inplace=True)
            else:
                print(f"Warning: Feature '{feature_name}' not implemented.")

    def _merge_context_features(self, context_df: pd.DataFrame, context_tf: str):
        """Merges features from a context dataframe into the base dataframe."""
        context_feature_cols = [col for col in context_df.columns if col.endswith(f"_{context_tf}")]
        
        if not context_feature_cols:
            print(f"No context features found for {context_tf} to merge.")
            return

        # merge_asof is robust for joining timeseries on the nearest preceding index.
        # Both dataframes are already sorted and timezone-aware.
        merged_df = pd.merge_asof(
            left=self.base_df,
            right=context_df[context_feature_cols],
            left_index=True,
            right_index=True,
            direction='forward'
        )
        
        # After merging, NaNs can be introduced for early base_df rows where no context is available yet
        self.base_df = merged_df

    def _synchronize_timezones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensures the dataframe's index is in UTC."""
        if df.index.tz is None:
            df = df.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
        else:
            df = df.tz_convert('UTC')
        return df.sort_index()

    # --- Feature Calculation Implementations ---
    
    def _calculate_trend(self, df: pd.DataFrame, method: str, length: int) -> pd.Series:
        """Calculates trend based on price position relative to a moving average."""
        if method == "ema":
            ma = ta.ema(df["close"], length=length)
        else: # sma
            ma = ta.sma(df["close"], length=length)
        return (df["close"] > ma).astype(int)

    def _calculate_volume_change(self, df: pd.DataFrame) -> pd.Series:
        return df['volume'].pct_change()
    
    def _calculate_volume_spike(self, df: pd.DataFrame) -> pd.Series:
        return (df['volume'] > df['volume'].rolling(window=20).mean() * 2).rename("volume_spike")

    def _calculate_custom_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates a set of custom candle patterns."""
        patterns = pd.DataFrame(index=df.index)
        patterns['CDL_DOJI'] = self._is_doji(df)
        patterns['CDL_ENGULFING_BULL'] = self._is_engulfing(df, bull=True)
        patterns['CDL_ENGULFING_BEAR'] = self._is_engulfing(df, bull=False)
        patterns['CDL_HAMMER'] = self._is_hammer(df)
        patterns['CDL_SHOOTING_STAR'] = self._is_shooting_star(df)
        return patterns

    def _is_doji(self, df: pd.DataFrame, threshold=0.05) -> pd.Series:
        """Detects Doji candles."""
        body_size = (df['close'] - df['open']).abs()
        range_size = df['high'] - df['low']
        return (body_size / range_size.where(range_size != 0)) < threshold

    def _is_engulfing(self, df: pd.DataFrame, bull: bool) -> pd.Series:
        """Detects Engulfing patterns."""
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        
        if bull:
            return (prev_close < prev_open) & (df['close'] > df['open']) & \
                   (df['close'] >= prev_open) & (df['open'] <= prev_close)
        else:
            return (prev_close > prev_open) & (df['close'] < df['open']) & \
                   (df['close'] <= prev_open) & (df['open'] >= prev_close)

    def _is_hammer(self, df: pd.DataFrame, body_thresh=0.3, upper_wick_thresh=0.1) -> pd.Series:
        """Detects Hammer pattern (small body at top, long lower wick)."""
        body_size = (df['close'] - df['open']).abs()
        range_size = (df['high'] - df['low']).where(df['high'] - df['low'] != 0)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)

        return (body_size <= range_size * body_thresh) & \
               (lower_wick >= body_size * 2) & \
               (upper_wick <= range_size * upper_wick_thresh)

    def _is_shooting_star(self, df: pd.DataFrame, body_thresh=0.3, lower_wick_thresh=0.1) -> pd.Series:
        """Detects Shooting Star pattern (small body at bottom, long upper wick)."""
        body_size = (df['close'] - df['open']).abs()
        range_size = (df['high'] - df['low']).where(df['high'] - df['low'] != 0)
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)

        return (body_size <= range_size * body_thresh) & \
               (upper_wick >= body_size * 2) & \
               (lower_wick <= range_size * lower_wick_thresh)

    def _calculate_bb_width(self, df: pd.DataFrame) -> pd.Series:
        bbands = df.ta.bbands(length=20, std=2)
        return bbands['BBB_20_2.0']

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        features['session_hour'] = df.index.hour
        features['weekday'] = df.index.weekday
        return features
    
    def _calculate_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """A simple market regime filter (e.g., bull/bear based on long-term MA)."""
        long_ma = ta.ema(df["close"], length=200)
        return (df["close"] > long_ma).astype(int).rename("market_regime")

    def _add_seasonality_features(self, df: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)
        features['month'] = df.index.month
        features['week_of_year'] = df.index.isocalendar().week.astype(int)
        return features 