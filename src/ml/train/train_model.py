import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from src.config.setting import settings


def train_model(dataset_path: str, model_output_path: str):
    """
    Train an XGBoost classifier on the prepared dataset and save the model.

    Args:
        dataset_path (str): Path to the CSV file with training data (features + 'target').
        model_output_path (str): Path where the trained model will be saved (.pkl).
    """
    # Load dataset
    df = pd.read_csv(dataset_path, index_col=0)
    X = df.drop(columns=['target'])
    y = df['target']

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Initialize and train model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, preds))

    # Save model
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print(f"Model saved to: {model_output_path}")


if __name__ == '__main__':
    # Paths based on settings
    dataset_file = os.path.join(settings.MODEL_DIR, 'training_dataset.csv')
    model_file = os.path.join(settings.MODEL_DIR, 'BTCUSDT_model.pkl')

    # Train and save
    train_model(dataset_file, model_file)
