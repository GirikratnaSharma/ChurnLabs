"""Model loading and prediction utilities"""

import os
import pickle
import json
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    model_path: str = "ml/models/churn_model.pkl"
    model_version: str = "v1.0"
    model_type: str = "XGBoost"
    metrics_path: str = "ml/models/metrics.json"
    feature_info_path: str = "ml/models/feature_info.json"

    class Config:
        env_file = ".env"
        case_sensitive = False


class ModelLoader:
    """Load and manage ML models"""

    def __init__(self):
        self.settings = Settings()
        self._model = None
        self._feature_info = None

    def get_model_path(self) -> str:
        """Get the path to the model file"""
        return self.settings.model_path

    def get_model_type(self) -> str:
        """Get the model type"""
        return self.settings.model_type

    def get_model_version(self) -> str:
        """Get the model version"""
        return self.settings.model_version

    def load_model(self):
        """Load the trained model"""
        if self._model is not None:
            return self._model

        model_path = self.get_model_path()
        if not os.path.exists(model_path):
            return None

        try:
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            return self._model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def load_feature_info(self) -> Dict[str, Any]:
        """Load feature information"""
        if self._feature_info is not None:
            return self._feature_info

        feature_info_path = self.settings.feature_info_path
        if not os.path.exists(feature_info_path):
            return {}

        try:
            with open(feature_info_path, "r") as f:
                self._feature_info = json.load(f)
            return self._feature_info
        except Exception as e:
            print(f"Error loading feature info: {e}")
            return {}

    def prepare_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Prepare features for prediction
        Matches the feature engineering from training
        """
        feature_info = self.load_feature_info()
        
        if not feature_info:
            # Fallback to simple numeric features
            feature_vector = [
                features.get("tenure", 0.0),
                features.get("monthly_charges", 0.0),
                features.get("total_charges", 0.0),
            ]
            return np.array([feature_vector])

        # Create a DataFrame with the input features
        df = pd.DataFrame([features])
        
        # One-hot encode categorical features (matching training)
        categorical_features = feature_info.get("categorical_features", [])
        for cat_feat in categorical_features:
            if cat_feat in df.columns:
                df = pd.get_dummies(df, columns=[cat_feat], drop_first=True, prefix=cat_feat)
        
        # Get feature names in the correct order
        feature_names = feature_info.get("feature_names", [])
        
        # Ensure all features exist (fill missing with 0)
        for feat_name in feature_names:
            if feat_name not in df.columns:
                df[feat_name] = 0
        
        # Select features in the correct order
        feature_vector = df[feature_names].values
        
        return feature_vector

    def load_metrics(self) -> Dict[str, Any]:
        """Load model metrics from file"""
        metrics_path = self.settings.metrics_path
        if not os.path.exists(metrics_path):
            return {}

        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metrics: {e}")
            return {}
