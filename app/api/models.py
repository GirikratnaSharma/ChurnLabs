"""Model management endpoints"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import ModelInfo, ModelMetrics
from ml.utils.model_loader import ModelLoader
import os

router = APIRouter()
model_loader = ModelLoader()


@router.get("/", response_model=ModelInfo)
async def get_model_info():
    """Get information about the current model"""
    model_path = model_loader.get_model_path()
    
    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404, detail="No model found. Please train a model first."
        )

    return ModelInfo(
        model_path=model_path,
        model_type=model_loader.get_model_type(),
        version=model_loader.get_model_version(),
    )


@router.get("/metrics", response_model=ModelMetrics)
async def get_model_metrics():
    """Get model performance metrics"""
    try:
        metrics = model_loader.load_metrics()
        return ModelMetrics(**metrics)
    except Exception as e:
        raise HTTPException(
            status_code=404, detail=f"Metrics not available: {str(e)}"
        )
