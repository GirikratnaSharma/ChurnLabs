# ChurnLabs Quick Start Guide

## What is ChurnLabs?

ChurnLabs is a **customer churn prediction platform** that helps subscription businesses identify customers who are likely to cancel their subscriptions. It uses machine learning models (XGBoost/LightGBM) to predict churn probability, allowing businesses to proactively engage with at-risk customers.

## Key Components

1. **FastAPI Backend** (`app/`) - REST API for predictions
2. **ML Pipeline** (`ml/`) - Model training and inference
3. **Streamlit Dashboard** (`dashboard/`) - Interactive visualization
4. **MLflow** - Experiment tracking and model versioning

## Getting Started

### 1. Initial Setup

```bash
# Run the setup script
./scripts/setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python scripts/init_db.py
cp .env.example .env
```

### 2. Train Your First Model

The training script will create sample data if none exists:

```bash
python ml/training/train.py
```

This will:
- Generate sample churn data (if needed)
- Train an XGBoost model
- Save the model to `ml/models/churn_model.pkl`
- Log experiments to MLflow

### 3. Start the API Server

```bash
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### 4. Start the Dashboard

In a new terminal:

```bash
streamlit run dashboard/app.py
```

Visit `http://localhost:8501` to see the interactive dashboard.

### 5. (Optional) Start MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Visit `http://localhost:5000` to view experiment tracking.

## Making Predictions

### Via API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_001",
    "features": {
      "tenure": 12,
      "monthly_charges": 79.99,
      "total_charges": 959.88,
      "contract": "Month-to-month",
      "payment_method": "Electronic check"
    }
  }'
```

### Via Dashboard

1. Navigate to the "Predictions" page
2. Fill in customer information
3. Click "Predict Churn"
4. View churn probability and risk level

## Project Structure

```
ChurnLabs/
├── app/              # FastAPI application
│   ├── api/         # API endpoints
│   └── models/      # Pydantic schemas
├── ml/              # ML pipeline
│   ├── training/    # Training scripts
│   ├── utils/       # Model utilities
│   └── models/      # Saved models
├── dashboard/       # Streamlit dashboard
├── scripts/         # Utility scripts
└── tests/           # Test files
```

## Next Steps

1. **Add your own data**: Replace `ml/data/churn_data.csv` with your customer data
2. **Customize features**: Modify feature engineering in `ml/training/train.py`
3. **Tune models**: Experiment with different hyperparameters
4. **Deploy**: Use Docker Compose or deploy to cloud platforms

## Troubleshooting

- **Model not found**: Run `python ml/training/train.py` first
- **Import errors**: Ensure virtual environment is activated and dependencies are installed
- **Port conflicts**: Change ports in `.env` file or command line arguments

## Need Help?

Check the main [README.md](README.md) for more detailed documentation.
