# ChurnLabs

A production-ready customer churn prediction platform built with Python, ML, and MLOps practices.

## Problem Statement

Subscription businesses lose revenue to customer churn. **ChurnLabs** predicts which customers are likely to churn in the next period so you can target retention campaigns effectively and reduce churn rates.

## What is ChurnLabs?

ChurnLabs is an end-to-end machine learning platform that:

- **Predicts churn probability** for each customer using advanced ML models (XGBoost, LightGBM)
- **Provides actionable insights** through an interactive dashboard
- **Serves predictions via REST API** for integration with existing systems
- **Tracks experiments** with MLflow for model versioning and comparison
- **Follows MLOps best practices** for production deployment

## Features

-  **High-accuracy churn prediction** using ensemble models
-  **Interactive dashboard** for visualizing predictions and insights
-  **RESTful API** for programmatic access
-  **Experiment tracking** with MLflow
-  **Dockerized** for easy deployment
-  **CI/CD pipeline** with GitHub Actions
-  **Comprehensive logging** and monitoring

## Tech Stack

- **ML**: scikit-learn, XGBoost, LightGBM, pandas, numpy
- **Experiment Tracking**: MLflow
- **API**: FastAPI
- **Dashboard**: Streamlit
- **Database**: PostgreSQL (local: SQLite for MVP)
- **DevOps**: Docker, GitHub Actions
- **Language**: Python 3.9+

## Project Structure

```
ChurnLabs/
├── app/                    # FastAPI application
│   ├── api/               # API routes
│   ├── models/            # Pydantic models
│   └── main.py            # FastAPI app entry point
├── ml/                    # ML pipeline
│   ├── data/              # Data processing
│   ├── training/          # Model training scripts
│   ├── models/            # Trained model artifacts
│   └── utils/             # ML utilities
├── dashboard/             # Streamlit dashboard
│   └── app.py             # Dashboard entry point
├── mlflow/                # MLflow tracking server
├── tests/                 # Unit and integration tests
├── docker/                # Docker configurations
├── .github/               # GitHub Actions workflows
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker image definition
└── README.md              # This file
```

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip or conda
- (Optional) Docker and Docker Compose

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ChurnLabs
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Initialize the database**
   ```bash
   python scripts/init_db.py
   ```

6. **Train the initial model** (optional - uses sample data)
   ```bash
   python ml/training/train.py
   ```

7. **Start the FastAPI server**
   ```bash
   uvicorn app.main:app --reload
   ```
   API will be available at `http://localhost:8000`

8. **Start the Streamlit dashboard** (in a new terminal)
   ```bash
   streamlit run dashboard/app.py
   ```
   Dashboard will be available at `http://localhost:8501`

9. **Start MLflow tracking server** (optional, in a new terminal)
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
   MLflow UI will be available at `http://localhost:5000`

### Docker Setup

```bash
docker-compose up -d
```

This will start:
- FastAPI server on port 8000
- Streamlit dashboard on port 8501
- MLflow UI on port 5000
- PostgreSQL database on port 5432

## Usage

### API Endpoints

- `GET /health` - Health check
- `POST /predict` - Predict churn probability for a customer
- `POST /predict/batch` - Batch prediction for multiple customers
- `GET /models` - List available models
- `GET /metrics` - Get model performance metrics

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "12345",
    "features": {
      "tenure": 12,
      "monthly_charges": 79.99,
      "total_charges": 959.88
    }
  }'
```

### Dashboard

Access the Streamlit dashboard at `http://localhost:8501` to:
- View churn predictions
- Explore customer segments
- Analyze model performance
- Generate reports

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
isort .
```

### Type Checking

```bash
mypy .
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

[Add your license here]

## Contact

[Add your contact information here]
