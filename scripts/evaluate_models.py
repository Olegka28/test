import asyncio
import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import settings
from src.data.exchange_client import ExchangeClient
from src.data.data_repository import DataRepository
from src.ml.data_builder import DataBuilder
from src.ml.model_manager import ModelManager

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

async def evaluate_model(symbol: str, timeframe: str):
    print(f"\n--- Evaluating model for {symbol} on {timeframe} ---")
    exchange_client = ExchangeClient(settings.exchange)
    data_repo = DataRepository(exchange_client)
    data_builder = DataBuilder(data_repo)
    model_manager = ModelManager(settings.models)

    # 1. Get config and data range
    tf_config = settings.trading.timeframes[timeframe]
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=730)

    # 2. Build dataset
    dataset = await data_builder.build(
        symbol=symbol,
        timeframe=timeframe,
        tf_config=tf_config,
        start_date=start_date,
        end_date=end_date
    )
    if not dataset:
        print(f"Could not build dataset for {symbol} on {timeframe}. Skipping.")
        await exchange_client.close()
        return None

    X, y = dataset
    y_mapped = y.map({-1: 0, 1: 1})
    if len(y_mapped.unique()) < 2:
        print("Only one class present. Skipping.")
        await exchange_client.close()
        return None

    # 3. Load model
    model = model_manager.load_model(symbol, timeframe)
    features_ref = model_manager.load_features(symbol, timeframe)
    if model is None or features_ref is None:
        print(f"No model or features found for {symbol} {timeframe}")
        await exchange_client.close()
        return None

    # 4. Align features
    missing = [f for f in features_ref if f not in X.columns]
    for f in missing:
        X[f] = 0
    extra = [f for f in X.columns if f not in features_ref]
    if extra:
        X = X.drop(columns=extra)
    X = X[features_ref]  # Ensure order

    # 5. Predict and evaluate
    y_pred = model.predict(X)
    acc = accuracy_score(y_mapped, y_pred)
    report = classification_report(y_mapped, y_pred, target_names=["Short", "Long"], output_dict=True)
    cm = confusion_matrix(y_mapped, y_pred)
    feature_importance = None
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

    # === Calibration Table ===
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)
        # Берём вероятность для предсказанного класса
        pred_proba = y_proba[np.arange(len(y_pred)), y_pred]
        correct = (y_pred == y_mapped.values)
        bins = np.linspace(0.5, 1.0, 6)  # 0.5-0.6, ..., 0.9-1.0
        bin_ids = np.digitize(pred_proba, bins, right=True)
        print(f"\nCalibration table for {symbol} {timeframe}:")
        print(f"{'Bin':<10}{'Count':<10}{'AvgProb':<12}{'Accuracy':<10}")
        for i in range(1, len(bins)):
            mask = bin_ids == i
            if np.sum(mask) == 0:
                continue
            avg_prob = pred_proba[mask].mean()
            acc_bin = correct[mask].mean()
            print(f"{bins[i-1]:.1f}-{bins[i]:.1f}   {np.sum(mask):<10}{avg_prob:<12.3f}{acc_bin:<10.3f}")

    # 6. Save report
    report_df = pd.DataFrame(report).T
    report_path = os.path.join(REPORTS_DIR, f"report_{symbol.replace('/', '_')}_{timeframe}.csv")
    report_df.to_csv(report_path)
    print(f"Saved classification report to {report_path}")

    # 7. Save confusion matrix plot
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Short", "Long"], yticklabels=["Short", "Long"])
    plt.title(f"Confusion Matrix: {symbol} {timeframe}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(REPORTS_DIR, f"cm_{symbol.replace('/', '_')}_{timeframe}.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix to {cm_path}")

    # 8. Save feature importance
    if feature_importance is not None:
        fi_path = os.path.join(REPORTS_DIR, f"feature_importance_{symbol.replace('/', '_')}_{timeframe}.csv")
        feature_importance.to_csv(fi_path)
        print(f"Saved feature importance to {fi_path}")

    await exchange_client.close()
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "accuracy": acc,
        "report_path": report_path,
        "cm_path": cm_path,
        "fi_path": fi_path if feature_importance is not None else None
    }

async def main():
    results = []
    for symbol in settings.trading.pairs:
        for timeframe in settings.trading.timeframes.keys():
            res = await evaluate_model(symbol, timeframe)
            if res:
                results.append(res)
    # Save summary
    summary = pd.DataFrame(results)
    summary_path = os.path.join(REPORTS_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary to {summary_path}")

if __name__ == "__main__":
    asyncio.run(main()) 