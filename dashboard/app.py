"""Streamlit dashboard for ChurnLabs"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ml.utils.model_loader import ModelLoader
import os
import json

# Page configuration
st.set_page_config(
    page_title="ChurnLabs Dashboard",
    page_icon="📊",
    layout="wide",
)

# Initialize model loader
model_loader = ModelLoader()


def load_sample_data():
    """Load sample data for visualization"""
    data_path = "ml/data/churn_data.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None


def main():
    """Main dashboard function"""
    st.title("📊 ChurnLabs Dashboard")
    st.markdown("Customer Churn Prediction Platform")

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Predictions", "Model Performance", "Data Insights"],
    )

    if page == "Home":
        show_home()
    elif page == "Predictions":
        show_predictions()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "Data Insights":
        show_data_insights()


def show_home():
    """Home page"""
    st.header("Welcome to ChurnLabs")
    st.markdown("""
    ChurnLabs is a production-ready customer churn prediction platform that helps 
    subscription businesses identify customers at risk of churning.
    
    ### Features:
    - 🎯 **High-accuracy predictions** using advanced ML models
    - 📊 **Interactive visualizations** of churn patterns
    - 🔍 **Customer segmentation** and risk analysis
    - 📈 **Model performance tracking** with MLflow
    """)

    # Check if model exists
    model_path = model_loader.get_model_path()
    if os.path.exists(model_path):
        st.success("✅ Model is loaded and ready for predictions")
    else:
        st.warning("⚠️ No model found. Please train a model first using: `python ml/training/train.py`")


def show_predictions():
    """Prediction page"""
    st.header("Churn Predictions")

    # Check if model exists
    model = model_loader.load_model()
    if model is None:
        st.error("Model not available. Please train a model first.")
        return

    # Input form
    st.subheader("Enter Customer Information")
    col1, col2 = st.columns(2)

    with col1:
        customer_id = st.text_input("Customer ID", value="CUST_001")
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)

    with col2:
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        )

    if st.button("Predict Churn"):
        features = {
            "tenure": float(tenure),
            "monthly_charges": float(monthly_charges),
            "total_charges": float(total_charges),
            "contract": contract,
            "payment_method": payment_method,
        }

        try:
            feature_vector = model_loader.prepare_features(features)
            churn_probability = model.predict_proba(feature_vector)[0][1]
            will_churn = churn_probability >= 0.5

            # Display results
            st.subheader("Prediction Results")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Churn Probability", f"{churn_probability:.2%}")

            with col2:
                st.metric("Prediction", "Will Churn" if will_churn else "Will Not Churn")

            with col3:
                risk_level = "High" if churn_probability > 0.7 else "Medium" if churn_probability > 0.5 else "Low"
                st.metric("Risk Level", risk_level)

            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Risk"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgreen"},
                        {'range': [50, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")


def show_model_performance():
    """Model performance page"""
    st.header("Model Performance")

    metrics_path = "ml/models/metrics.json"
    if not os.path.exists(metrics_path):
        st.warning("No metrics found. Please train a model first.")
        return

    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
    with col2:
        st.metric("Precision", f"{metrics.get('precision', 0):.2%}")
    with col3:
        st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
    with col4:
        st.metric("F1 Score", f"{metrics.get('f1_score', 0):.2%}")
    with col5:
        st.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.2%}")

    # Confusion matrix
    if "confusion_matrix" in metrics:
        st.subheader("Confusion Matrix")
        cm = metrics["confusion_matrix"]
        cm_df = pd.DataFrame([
            [cm["tn"], cm["fp"]],
            [cm["fn"], cm["tp"]]
        ], index=["Actual: No Churn", "Actual: Churn"],
           columns=["Predicted: No Churn", "Predicted: Churn"])

        fig = px.imshow(
            cm_df.values,
            labels=dict(x="Predicted", y="Actual"),
            x=cm_df.columns,
            y=cm_df.index,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig, use_container_width=True)


def show_data_insights():
    """Data insights page"""
    st.header("Data Insights")

    df = load_sample_data()
    if df is None:
        st.warning("No data available. Please ensure data file exists at ml/data/churn_data.csv")
        return

    st.subheader("Dataset Overview")
    st.dataframe(df.head(10))

    # Churn distribution
    st.subheader("Churn Distribution")
    churn_counts = df["churn"].value_counts()
    fig = px.pie(
        values=churn_counts.values,
        names=["No Churn", "Churn"],
        title="Overall Churn Rate"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature distributions
    st.subheader("Feature Distributions")
    col1, col2 = st.columns(2)

    with col1:
        if "tenure" in df.columns:
            fig = px.histogram(df, x="tenure", color="churn", nbins=30, title="Tenure Distribution")
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "monthly_charges" in df.columns:
            fig = px.histogram(df, x="monthly_charges", color="churn", nbins=30, title="Monthly Charges Distribution")
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
