import joblib
import os
from typing import Any
import json

from src.config import ModelSettings


class ModelManager:
    """
    Handles saving and loading of trained machine learning models.
    """

    def __init__(self, settings: ModelSettings):
        self.settings = settings
        if not os.path.exists(self.settings.directory):
            os.makedirs(self.settings.directory)

    def _get_model_path(self, symbol: str, timeframe: str) -> str:
        """Constructs the full path for a model file."""
        filename = f"model_{symbol.replace('/', '_')}_{timeframe}.joblib"
        return os.path.join(self.settings.directory, filename)

    def _get_features_path(self, symbol: str, timeframe: str) -> str:
        filename = f"model_{symbol.replace('/', '_')}_{timeframe}_features.json"
        return os.path.join(self.settings.directory, filename)

    def save_model(self, model: Any, symbol: str, timeframe: str, feature_names=None):
        """
        Saves a trained model to a file and saves feature names as JSON.
        """
        model_path = self._get_model_path(symbol, timeframe)
        print(f"Saving model for {symbol} on {timeframe} to {model_path}...")
        joblib.dump(model, model_path)
        print("Model saved successfully.")
        if feature_names is not None:
            features_path = self._get_features_path(symbol, timeframe)
            with open(features_path, 'w') as f:
                json.dump(feature_names, f)
            print(f"Saved feature names to {features_path}")

    def load_model(self, symbol: str, timeframe: str) -> Any:
        """
        Loads a model from a file.
        Returns the model object or None if the file doesn't exist.
        """
        model_path = self._get_model_path(symbol, timeframe)
        if not os.path.exists(model_path):
            print(f"No model found at {model_path}")
            return None
        
        print(f"Loading model for {symbol} on {timeframe} from {model_path}...")
        model = joblib.load(model_path)
        print("Model loaded successfully.")
        return model

    def load_features(self, symbol: str, timeframe: str):
        features_path = self._get_features_path(symbol, timeframe)
        if not os.path.exists(features_path):
            print(f"No features file found at {features_path}")
            return None
        with open(features_path, 'r') as f:
            return json.load(f) 