"""FastAPI application entry point for ChurnLabs"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import predictions, health, models
from app.models.schemas import HealthResponse

app = FastAPI(
    title="ChurnLabs API",
    description="Customer churn prediction platform API",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predictions.router, prefix="/predict", tags=["Predictions"])
app.include_router(models.router, prefix="/models", tags=["Models"])


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to ChurnLabs API",
        "version": "1.0.0",
        "docs": "/docs",
    }
