"""Model training script"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Training settings"""
    data_path: str = "ml/data/churn_data.csv"
    model_path: str = "ml/models/churn_model.pkl"
    metrics_path: str = "ml/models/metrics.json"
    feature_info_path: str = "ml/models/feature_info.json"
    model_type: str = "XGBoost"  # Options: XGBoost, LightGBM
    test_size: float = 0.2
    random_state: int = 42
    mlflow_experiment_name: str = "churn_prediction"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"

    class Config:
        env_file = ".env"
        case_sensitive = False


def load_data(data_path: str) -> pd.DataFrame:
    """Load training data"""
    if not os.path.exists(data_path):
        print(f"Warning: Data file not found at {data_path}")
        print("Creating sample data for demonstration...")
        return create_sample_data()

    return pd.read_csv(data_path)


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample churn data for demonstration"""
    np.random.seed(42)
    
    data = {
        "customer_id": [f"CUST_{i:05d}" for i in range(n_samples)],
        "tenure": np.random.randint(1, 73, n_samples),
        "monthly_charges": np.random.uniform(20, 120, n_samples),
        "total_charges": np.random.uniform(100, 8000, n_samples),
        "contract": np.random.choice(["Month-to-month", "One year", "Two year"], n_samples),
        "payment_method": np.random.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n_samples
        ),
    }
    
    df = pd.DataFrame(data)
    
    # Create a simple churn target based on features
    df["churn"] = (
        (df["tenure"] < 12).astype(int) * 0.3 +
        (df["monthly_charges"] > 80).astype(int) * 0.2 +
        (df["contract"] == "Month-to-month").astype(int) * 0.3 +
        np.random.random(n_samples) * 0.2
    ) > 0.5
    
    df["churn"] = df["churn"].astype(int)
    
    # Save sample data
    os.makedirs(os.path.dirname("ml/data/"), exist_ok=True)
    df.to_csv("ml/data/churn_data.csv", index=False)
    print(f"Sample data saved to ml/data/churn_data.csv")
    
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and target"""
    # Select features
    feature_cols = ["tenure", "monthly_charges", "total_charges"]
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, columns=["contract", "payment_method"], drop_first=True)
    
    # Get all feature columns (excluding target and customer_id)
    all_feature_cols = [col for col in df_encoded.columns if col not in ["churn", "customer_id"]]
    
    X = df_encoded[all_feature_cols].values
    y = df_encoded["churn"].values
    
    return X, y, all_feature_cols


def train_model(X_train, y_train, model_type: str = "XGBoost"):
    """Train the model"""
    if model_type == "XGBoost":
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
        )
    elif model_type == "LightGBM":
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "confusion_matrix": {
            "tn": int(confusion_matrix(y_test, y_pred)[0, 0]),
            "fp": int(confusion_matrix(y_test, y_pred)[0, 1]),
            "fn": int(confusion_matrix(y_test, y_pred)[1, 0]),
            "tp": int(confusion_matrix(y_test, y_pred)[1, 1]),
        },
    }

    return metrics


def main():
    """Main training function"""
    settings = Settings()

    # Set up MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    print("Loading data...")
    df = load_data(settings.data_path)
    print(f"Loaded {len(df)} samples")

    print("Preparing features...")
    X, y, feature_names = prepare_features(df)
    print(f"Features: {feature_names}")

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=settings.test_size, random_state=settings.random_state, stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    with mlflow.start_run():
        print(f"Training {settings.model_type} model...")
        model = train_model(X_train, y_train, settings.model_type)

        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)

        print("\nModel Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")

        # Log to MLflow
        mlflow.log_params({
            "model_type": settings.model_type,
            "test_size": settings.test_size,
            "random_state": settings.random_state,
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        # Save model
        os.makedirs(os.path.dirname(settings.model_path), exist_ok=True)
        with open(settings.model_path, "wb") as f:
            pickle.dump(model, f)
        print(f"\nModel saved to {settings.model_path}")

        # Save metrics
        with open(settings.metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {settings.metrics_path}")

        # Save feature information
        feature_info = {
            "feature_names": feature_names,
            "categorical_features": ["contract", "payment_method"],
            "numeric_features": ["tenure", "monthly_charges", "total_charges"]
        }
        with open(settings.feature_info_path, "w") as f:
            json.dump(feature_info, f, indent=2)
        print(f"Feature info saved to {settings.feature_info_path}")

        mlflow.log_artifact(settings.model_path)
        mlflow.log_artifact(settings.metrics_path)
        mlflow.log_artifact(settings.feature_info_path)

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
