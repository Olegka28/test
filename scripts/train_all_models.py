import os
import sys
from pathlib import Path
import logging
import pandas as pd
from src.ml.train.train_model import train_xgb_model
from src.ml.train.data_builder import build_training_data
from src.config.settings import SUPPORTED_PAIRS, SUPPORTED_TIMEFRAMES, DATA_DIR, MODELS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def model_filename(symbol, timeframe):
    return f"xgb_{symbol.replace('/', '')}_{timeframe}.json"

def model_path(symbol, timeframe):
    return os.path.join(MODELS_DIR, model_filename(symbol, timeframe))

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for symbol in SUPPORTED_PAIRS:
        for timeframe in SUPPORTED_TIMEFRAMES:
            model_file = model_path(symbol, timeframe)
            if os.path.exists(model_file):
                logger.info(f"Model already exists: {model_file}, skipping...")
                continue
            logger.info(f"Building training data for {symbol} {timeframe}...")
            X, y = build_training_data(symbol, timeframe)
            logger.info(f"Training model for {symbol} {timeframe}...")
            model, info = train_xgb_model(X, y)
            logger.info(f"Saving model to {model_file}")
            model.save_model(model_file)
            # Save info
            info_file = model_file.replace('.json', '_info.json')
            pd.Series(info).to_json(info_file)
    logger.info("All models trained!")

if __name__ == "__main__":
    main() 