# ChurnLabs: Architecture and Code Walkthrough

This document explains **everything** in the ChurnLabs codebase: what each part does, how data flows, and what happens at runtime. No code changes—just explanation.

---

## 1. High-Level Picture

ChurnLabs has three main “user-facing” pieces:

1. **FastAPI app** – REST API that serves predictions and model info.
2. **ML pipeline** – Scripts that train models and utilities that load them for prediction.
3. **Streamlit dashboard** – Web UI to explore data, run predictions, and view metrics.

They all rely on:

- **Saved artifacts**: trained model (`.pkl`), metrics (`.json`), feature info (`.json`).
- **Config**: `.env` (and defaults in code) for paths, MLflow, etc.
- **Optional**: SQLite DB for storing predictions/customers (currently only created by `init_db.py`, not yet used by the API).

So: **training** produces the artifacts; **API** and **dashboard** consume them.

---

## 2. Dependencies (`requirements.txt`)

- **fastapi, uvicorn** – Web framework and ASGI server for the API.
- **pydantic, pydantic-settings** – Request/response validation and config from env.
- **scikit-learn, xgboost, lightgbm, pandas, numpy** – ML and data handling.
- **mlflow** – Experiment tracking and model logging.
- **streamlit, plotly** – Dashboard and charts.
- **sqlalchemy** – DB layer (for future use; `init_db.py` uses raw `sqlite3`).
- **python-dotenv, python-multipart** – Env loading and file upload support.
- **pytest, pytest-asyncio, httpx** – Tests and async HTTP client.
- **black, isort, mypy, flake8** – Formatting and linting.

---

## 3. FastAPI Application (`app/`)

### 3.1 Entry point: `app/main.py`

- Creates the **FastAPI** app (title, description, version).
- Adds **CORS** so browsers can call the API from other origins (e.g. a frontend).
- **Mounts routers**:
  - Health → no prefix (so `/health`).
  - Predictions → prefix `/predict` (so `/predict` and `/predict/batch`).
  - Models → prefix `/models` (so `/models` and `/models/metrics`).
- **Root route** `GET /`: returns a welcome message and link to `/docs`.

So when you run `uvicorn app.main:app`, all these routes become available.

### 3.2 Schemas: `app/models/schemas.py`

Pydantic models define the **shape** of requests and responses:

- **HealthResponse** – `status`, `message` for `/health`.
- **CustomerFeatures** – One customer’s inputs: `tenure`, `monthly_charges`, `total_charges`, `contract`, `payment_method`, and optional extra `features` dict.
- **PredictionRequest** – `customer_id` + `features` (a `CustomerFeatures` instance).
- **PredictionResponse** – `customer_id`, `churn_probability` (0–1), `will_churn` (bool).
- **BatchPredictionRequest** – list of `PredictionRequest`.
- **BatchPredictionResponse** – list of `PredictionResponse`.
- **ModelInfo** – `model_path`, `model_type`, `version`.
- **ModelMetrics** – `accuracy`, `precision`, `recall`, `f1_score`, `roc_auc`, optional `confusion_matrix`.

FastAPI uses these to validate JSON and generate OpenAPI docs.

### 3.3 Health: `app/api/health.py`

- **Router**: no prefix (mounting is in `main.py`).
- **GET /health**: returns `HealthResponse(status="healthy", message="...")`.
- Used for liveness checks (e.g. load balancers, k8s).

### 3.4 Predictions: `app/api/predictions.py`

- **Single prediction POST /predict/**
  - Uses a **single shared** `ModelLoader()` instance.
  - Calls `model_loader.load_model()` → gets the trained model (or `None` if missing).
  - If no model → **503** “Model not available”.
  - Takes `request.features` and passes them to `model_loader.prepare_features(...)` to get a numeric feature vector.
  - Calls `model.predict_proba(features)[0][1]` → probability of class 1 (churn).
  - Sets `will_churn = churn_probability >= 0.5`.
  - Returns `PredictionResponse(customer_id, churn_probability, will_churn)`.
  - Any other exception → **500** with error message.

- **Batch prediction POST /predict/batch**
  - Same model loading and 503 if no model.
  - Iterates over `request.customers`, and for each:
    - Prepares features the same way.
    - Gets probability and binary prediction.
    - Appends a `PredictionResponse` to a list.
  - Returns `BatchPredictionResponse(predictions=...)`.

Important: `request.features` is a **Pydantic model** (`CustomerFeatures`). `prepare_features` in `model_loader` expects a **dict** (it uses `.get(...)`). So the API should pass something like `request.features.model_dump()` (or `request.features.dict()` in Pydantic v1) into `prepare_features` so keys like `"tenure"`, `"monthly_charges"` are available. Otherwise you may get errors or wrong behavior when the loader tries to build the feature vector.

### 3.5 Models: `app/api/models.py`

- **GET /models/**
  - Uses `model_loader.get_model_path()` and checks if that file exists.
  - If not → **404** “No model found”.
  - If yes → returns `ModelInfo(model_path, model_type, version)` (paths and version come from loader settings).

- **GET /models/metrics**
  - Calls `model_loader.load_metrics()` (reads `ml/models/metrics.json`).
  - Returns **ModelMetrics**; if file missing or invalid → **404**.

So the API does **not** train models; it only **reads** what the training pipeline wrote.

---

## 4. ML Utilities (`ml/utils/model_loader.py`)

### 4.1 Settings

`Settings` (via `pydantic_settings`) loads from `.env` with defaults:

- `model_path` – where the pickle model lives (e.g. `ml/models/churn_model.pkl`).
- `model_version`, `model_type` – for display in `/models`.
- `metrics_path` – path to `metrics.json`.
- `feature_info_path` – path to `feature_info.json` (written by training).

### 4.2 ModelLoader class

- **Caching**: keeps `_model` and `_feature_info` in memory after first load so the API doesn’t re-read disk on every request.
- **get_model_path / get_model_type / get_model_version**: return config values (used by API and dashboard).
- **load_model()**:
  - If already loaded, returns cached model.
  - If `model_path` doesn’t exist, returns `None`.
  - Otherwise opens the file and `pickle.load()`s it → returns the trained classifier (e.g. XGBoost/LightGBM).
- **load_feature_info()**:
  - If already loaded, returns cached dict.
  - Reads `feature_info.json` (from training) which contains `feature_names`, `categorical_features`, `numeric_features`.
  - Needed so inference uses the **same** features in the **same order** as training.
- **prepare_features(features)` (features = dict)**:
  - If no feature info: **fallback** – builds a simple 3‑element vector `[tenure, monthly_charges, total_charges]` so something works even without `feature_info.json`.
  - If feature info exists:
    - Builds a one-row DataFrame from `features`.
    - One-hot encodes the categorical columns listed in `categorical_features` (same logic as in training: `contract`, `payment_method`).
    - Ensures all columns in `feature_names` exist (missing → 0).
    - Returns `df[feature_names].values` so the model gets the same column order as in training.
  - Returns a 2D numpy array (one row).
- **load_metrics()**: reads `metrics_path` JSON and returns the dict (accuracy, precision, recall, f1, roc_auc, confusion_matrix). Used by `/models/metrics` and the dashboard.

So the loader is the **bridge** between raw API input (customer attributes) and the trained model’s expected input (numeric vector).

---

## 5. Training Pipeline (`ml/training/train.py`)

This script is run **manually** (or by a job). It produces the model and metadata that the API and dashboard use.

### 5.1 Settings

- Paths: data CSV, model pickle, metrics JSON, feature info JSON.
- `model_type`: `"XGBoost"` or `"LightGBM"`.
- Train/test split: `test_size`, `random_state`.
- MLflow: `mlflow_tracking_uri` (e.g. `sqlite:///mlflow.db`), `mlflow_experiment_name`.

### 5.2 Data loading

- **load_data(data_path)**:
  - If the CSV exists, loads it with pandas.
  - If not, calls **create_sample_data()**.

### 5.3 Sample data creation

- **create_sample_data(n_samples=1000)**:
  - Builds synthetic rows: `customer_id`, `tenure`, `monthly_charges`, `total_charges`, `contract`, `payment_method`.
  - Defines a **synthetic churn** rule so the model has something to learn: higher churn for short tenure, high monthly charges, month-to-month contract, plus noise.
  - Saves CSV to `ml/data/churn_data.csv` and returns the DataFrame.

So you can run the project without real data; the first training run creates fake data.

### 5.4 Feature preparation (training side)

- **prepare_features(df)**:
  - Keeps numeric columns and one-hot encodes `contract` and `payment_method` with `pd.get_dummies(..., drop_first=True)`.
  - Removes `churn` and `customer_id` to get the list of feature columns.
  - Returns `X` (array), `y` (churn labels), and `feature_names` (list of column names in order).

This order is exactly what gets saved in `feature_info.json` and later used by `model_loader.prepare_features()`.

### 5.5 Model training

- **train_model(X_train, y_train, model_type)**:
  - Instantiates either `XGBClassifier` or `LGBMClassifier` with fixed hyperparameters.
  - Fits on `(X_train, y_train)` and returns the fitted model.

### 5.6 Evaluation

- **evaluate_model(model, X_test, y_test)**:
  - Predicts labels and probabilities on `X_test`.
  - Computes accuracy, precision, recall, f1, ROC AUC, and a 2×2 confusion matrix (stored as `tn`, `fp`, `fn`, `tp` in a dict).
  - Returns a metrics dict (all values suitable for JSON).

### 5.7 main() flow

1. Load data (or create sample).
2. Prepare features → get `X`, `y`, `feature_names`.
3. Train/test split (stratified by `y`).
4. **mlflow.start_run()**:
   - Train the model.
   - Evaluate on test set.
   - Log params (model_type, test_size, random_state) and metrics to MLflow.
   - Log the model with `mlflow.sklearn.log_model(model, "model")`.
   - Save model to disk with `pickle.dump(model, f)` at `model_path`.
   - Write `metrics.json` at `metrics_path`.
   - Write `feature_info.json` with `feature_names`, `categorical_features`, `numeric_features`.
   - Log those files as artifacts in MLflow.
5. Print success.

After this, the API and dashboard can use the model and metrics.

---

## 6. Streamlit Dashboard (`dashboard/app.py`)

Single app with a sidebar to switch pages. All pages use the same `ModelLoader()` and read from the same artifact paths (so they must be run from the project root or paths adjusted).

### 6.1 Home

- Short intro and feature list.
- Checks if `model_path` exists: shows “Model is loaded” or “No model found” and reminds user to run training.

### 6.2 Predictions

- If `model_loader.load_model()` is `None`, shows an error and returns.
- Renders inputs: customer ID, tenure, monthly charges, total charges, contract, payment method.
- On “Predict Churn”:
  - Builds a **dict** `features` (same keys as the API).
  - Calls `model_loader.prepare_features(features)` and `model.predict_proba(...)[0][1]`.
  - Shows churn probability, Will Churn / Will Not Churn, and a risk level (Low/Medium/High).
  - Draws a Plotly gauge for churn risk.

So the dashboard does **the same prediction logic** as the API, but in-process (no HTTP call to the API).

### 6.3 Model Performance

- Reads `ml/models/metrics.json` (same as `/models/metrics`).
- Shows accuracy, precision, recall, F1, ROC AUC in columns.
- If confusion matrix exists, displays it as a heatmap (Plotly).

### 6.4 Data Insights

- Loads `ml/data/churn_data.csv` if present.
- Shows a small table, a pie chart of churn vs no-churn, and histograms of tenure and monthly_charges by churn.

So the dashboard is read-only on artifacts and data; it doesn’t train or change anything.

---

## 7. Database Script (`scripts/init_db.py`)

- Connects to `churnlabs.db` (SQLite in current directory).
- Creates two tables if they don’t exist:
  - **predictions**: id, customer_id, churn_probability, will_churn, created_at.
  - **customers**: customer_id (PK), tenure, monthly_charges, total_charges, contract, payment_method, created_at.
- No other code uses this DB yet; it’s prepared for future use (e.g. storing predictions or customer records).

---

## 8. Config and Environment

- **.env.example**: documents variables for API host/port, database URL, MLflow URI/experiment, model path/version, dashboard port, log level.
- **.env**: copy of `.env.example` (or created by setup script); actual values are loaded by `pydantic_settings` in `model_loader` and in the training script. The API and dashboard use the loader’s settings when they instantiate `ModelLoader()`.

---

## 9. Data Flow Summary

1. **Training** (`python ml/training/train.py`):
   - Data (real or synthetic) → feature prep → train/test split → fit model → evaluate → write `churn_model.pkl`, `metrics.json`, `feature_info.json`, and MLflow artifacts.

2. **Single prediction (API)**:
   - Client sends `POST /predict` with `customer_id` and `features` (JSON).
   - FastAPI validates body as `PredictionRequest`.
   - Predictions router gets model from `ModelLoader`, converts `features` to a feature vector with `prepare_features`, runs `predict_proba`, returns `PredictionResponse`.

3. **Single prediction (Dashboard)**:
   - User fills form → same `features` dict → same `ModelLoader.prepare_features` and `model.predict_proba` in the Streamlit process → results and gauge shown.

4. **Model info and metrics**:
   - API: `GET /models` and `GET /models/metrics` read from loader (file existence and `metrics.json`).
   - Dashboard: Model Performance page reads `metrics.json` directly.

So: **one training pipeline** produces artifacts; **one loader** defines how features are built and how the model is read; **API and dashboard** both depend on that loader and those artifacts. Understanding `model_loader.py` and `train.py` gives you almost the whole story; the rest is routing, validation, and UI.

---

## 10. One Fix to Keep in Mind

In `app/api/predictions.py`, when calling `model_loader.prepare_features(request.features)`, pass a dict. For example:

- `model_loader.prepare_features(request.features.model_dump())` (Pydantic v2), or  
- `model_loader.prepare_features(request.features.dict())` (Pydantic v1),

so that `prepare_features` can use `.get(...)` and key-based access correctly. The same applies for batch: each `customer_request.features` should be converted to a dict before `prepare_features`.

---

That’s the full picture of what’s in the repo and what’s happening end to end.
