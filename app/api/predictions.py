"""Prediction endpoints"""

from fastapi import APIRouter, HTTPException
from typing import List
from app.models.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)
from ml.utils.model_loader import ModelLoader

router = APIRouter()
model_loader = ModelLoader()


@router.post("/", response_model=PredictionResponse)
async def predict_churn(request: PredictionRequest):
    """
    Predict churn probability for a single customer
    """
    try:
        model = model_loader.load_model()
        if model is None:
            raise HTTPException(
                status_code=503, detail="Model not available. Please train a model first."
            )

        # Convert request to feature vector
        features = model_loader.prepare_features(request.features)
        
        # Make prediction
        churn_probability = model.predict_proba(features)[0][1]
        churn_prediction = churn_probability >= 0.5

        return PredictionResponse(
            customer_id=request.customer_id,
            churn_probability=float(churn_probability),
            will_churn=bool(churn_prediction),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_churn_batch(request: BatchPredictionRequest):
    """
    Predict churn probability for multiple customers
    """
    try:
        model = model_loader.load_model()
        if model is None:
            raise HTTPException(
                status_code=503, detail="Model not available. Please train a model first."
            )

        predictions = []
        for customer_request in request.customers:
            features = model_loader.prepare_features(customer_request.features)
            churn_probability = model.predict_proba(features)[0][1]
            churn_prediction = churn_probability >= 0.5

            predictions.append(
                PredictionResponse(
                    customer_id=customer_request.customer_id,
                    churn_probability=float(churn_probability),
                    will_churn=bool(churn_prediction),
                )
            )

        return BatchPredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
