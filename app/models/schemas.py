"""Pydantic schemas for API requests and responses"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str


class CustomerFeatures(BaseModel):
    """Customer feature set"""
    tenure: Optional[float] = Field(None, description="Customer tenure in months")
    monthly_charges: Optional[float] = Field(None, description="Monthly charges")
    total_charges: Optional[float] = Field(None, description="Total charges")
    contract: Optional[str] = Field(None, description="Contract type")
    payment_method: Optional[str] = Field(None, description="Payment method")
    # Add more features as needed
    features: Optional[Dict[str, Any]] = Field(None, description="Additional features")


class PredictionRequest(BaseModel):
    """Single prediction request"""
    customer_id: str = Field(..., description="Unique customer identifier")
    features: CustomerFeatures = Field(..., description="Customer features")


class PredictionResponse(BaseModel):
    """Prediction response"""
    customer_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of churn")
    will_churn: bool = Field(..., description="Whether customer will churn")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    customers: List[PredictionRequest] = Field(..., description="List of customers to predict")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")


class ModelInfo(BaseModel):
    """Model information"""
    model_path: str
    model_type: str
    version: str


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[Dict[str, int]] = None
