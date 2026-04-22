import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from datetime import datetime
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    recall_score,
)

# Configuration
st.set_page_config(page_title="Player Spending Classifier", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
LOG_PATH = os.path.join(BASE_DIR, "..", "logs", "predictions.csv")


# Basic security check, would use st.secrets in production, for this capstone password is hardcoded
def check_pwd():
    """Prompt for a password"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if st.session_state.authenticated:
        return True

    st.title("Login")
    password = st.text_input("Enter password:", type="password")
    if st.button("Login"):
        if password == "capstone":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False


# Data and model loading
@st.cache_data
def load_data():
    raw_path = os.path.join(DATA_DIR, "raw", "game_data.csv")
    cleaned_path = os.path.join(DATA_DIR, "clean", "cleaned_dataset.csv")
    raw_df = pd.read_csv(raw_path)
    cleaned_df = pd.read_csv(cleaned_path)
    return raw_df, cleaned_df


@st.cache_resource
def load_models():
    rf = joblib.load(os.path.join(MODELS_DIR, "random_forest.joblib"))
    lr = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    dt = joblib.load(os.path.join(MODELS_DIR, "decision_tree.joblib"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "encoders.joblib"))
    split_data = joblib.load(os.path.join(MODELS_DIR, "split_data.joblib"))
    return rf, lr, dt, encoders, split_data


# Prediction logger (monitoring requirement)
def log_prediction(input_features, predicted_class, confidence):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["timestamp", "input_features", "predicted_class", "confidence"]
            )
        writer.writerow(
            [
                datetime.now().isoformat(),
                str(input_features),
                predicted_class,
                f"{confidence:.4f}",
            ]
        )
