import asyncio
from typing import Tuple, Optional, Dict
import pandas as pd
from datetime import datetime

from src.data.data_repository import DataRepository
from src.features.feature_generator import FeatureGenerator
from src.features.target_generator import TargetGenerator
from src.config import TimeframeConfig


class DataBuilder:
    """
    Orchestrates the process of fetching, processing, and preparing data
    for model training, including multi-timeframe context.
    """

    def __init__(self, data_repo: DataRepository):
        self.data_repo = data_repo

    async def _fetch_all_data(
        self,
        symbol: str,
        base_timeframe: str,
        tf_config: TimeframeConfig,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches data for the base timeframe and all required context timeframes.
        """
        all_timeframes = {base_timeframe}.union(tf_config.feature_config.get("context", {}).keys())
        
        tasks = {
            tf: self.data_repo.get_historical_data(
                symbol=symbol,
                timeframe=tf,
                start_date=start_date,
                end_date=end_date
            )
            for tf in all_timeframes
        }

        results = await asyncio.gather(*tasks.values())
        
        dataframes = {}
        for tf, df in zip(tasks.keys(), results):
            if df is None or df.empty:
                raise ValueError(f"Failed to fetch data for required timeframe: {tf}")
            dataframes[tf] = df
        
        return dataframes

    async def build(
        self,
        symbol: str,
        timeframe: str,
        tf_config: TimeframeConfig,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """
        Builds the feature matrix (X) and target vector (y).
        """
        try:
            # 1. Fetch all required historical data
            dataframes = await self._fetch_all_data(symbol, timeframe, tf_config, start_date, end_date)
        except ValueError as e:
            print(e)
            return None

        # 2. Generate features using the new generator
        feature_gen = FeatureGenerator(dataframes, timeframe, tf_config.feature_config)
        df_with_features = feature_gen.generate_features()

        # 3. Generate target variable
        target_gen = TargetGenerator(
            df_with_features, 
            tf_config.prediction_candles, 
            tf_config.price_change_threshold
        )
        final_df = target_gen.generate_target()

        print(f"[LOG] Target value counts before filtering:\n{final_df['target'].value_counts(dropna=False)}")

        # 4. Filter out rows with no signal and NaN in базовых фичах
        base_cols = [col for col in final_df.columns if col.endswith(f"_{timeframe}") or col in ['open', 'high', 'low', 'close', 'volume', 'target']]
        final_df = final_df[final_df['target'] != 0].copy()
        final_df = final_df.dropna(subset=base_cols)
        print(f"[LOG] Shape after filtering: {final_df.shape}")
        print(f"[LOG] Target value counts after filtering:\n{final_df['target'].value_counts(dropna=False)}")
        
        if final_df.empty:
            print("No training examples found after filtering for non-zero targets.")
            return None

        # 5. Separate features (X) and target (y)
        y = final_df['target']
        # Drop raw price data and the target itself
        drop_cols = ['open', 'high', 'low', 'close', 'volume', 'target']
        X = final_df.drop(columns=[col for col in drop_cols if col in final_df.columns])

        print(f"Data building complete. X shape: {X.shape}, y shape: {y.shape}")
        print(f"Target distribution:\n{y.value_counts(normalize=True)}")
        
        return X, y 