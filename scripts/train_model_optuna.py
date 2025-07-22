import os
import sys
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

import optuna
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# make project root importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings
from src.data.exchange_client import ExchangeClient
from src.data.data_repository import DataRepository
from src.ml.data_builder import DataBuilder
from src.ml.model_manager import ModelManager

N_TRIALS = 50  # количество попыток подбора гиперпараметров

MODEL_DIR = settings.models.directory if hasattr(settings.models, 'directory') else "models/"

def model_exists(symbol: str, timeframe: str) -> bool:
    fname = f"model_{symbol.replace('/', '_')}_{timeframe}.joblib"
    return os.path.exists(os.path.join(MODEL_DIR, fname))

def suggest_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Возвращает словарь гиперпараметров для XGBClassifier."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),  # L2
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),   # L1
    }

async def build_dataset(symbol: str, timeframe: str):
    exchange_client = ExchangeClient(settings.exchange)
    data_repo = DataRepository(exchange_client)
    data_builder = DataBuilder(data_repo)

    # 4 года данных
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=1460)

    tf_config = settings.trading.timeframes[timeframe]
    dataset = await data_builder.build(
        symbol=symbol,
        timeframe=timeframe,
        tf_config=tf_config,
        start_date=start_date,
        end_date=end_date,
    )
    await exchange_client.close()
    return dataset

async def tune_and_train(symbol: str, timeframe: str):
    if model_exists(symbol, timeframe):
        print(f"[SKIP] Model already exists for {symbol} {timeframe}")
        return
    print(f"\n==== Optuna tuning for {symbol} {timeframe} ====")
    dataset = await build_dataset(symbol, timeframe)
    if not dataset:
        print("Dataset is empty, skip.")
        return

    X, y = dataset
    y_mapped = y.map({-1: 0, 1: 1})  # бинаризация

    if len(y_mapped.unique()) < 2:
        print("Only one class present, skip tuning.")
        return

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
    )

    def objective(trial: optuna.Trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "random_state": 42,
            **suggest_params(trial),
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_valid)
        return accuracy_score(y_valid, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    print("Best accuracy:", study.best_value)
    print("Best params:", study.best_params)

    # Обучаем финальную модель на всём тренировочном датасете
    best_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42,
        **study.best_params,
    }
    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)

    model_manager = ModelManager(settings.models)
    model_manager.save_model(final_model, symbol, timeframe, feature_names=list(X.columns))
    print(f"Saved tuned model for {symbol} {timeframe}")

async def main():
    for symbol in settings.trading.pairs:
        for timeframe in settings.trading.timeframes.keys():
            await tune_and_train(symbol, timeframe)

if __name__ == "__main__":
    asyncio.run(main()) 