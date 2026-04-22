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

GREEN = "#2ecc71"
ORANGE = "#f39c12"
RED = "#e74c3c"
BLUE = "#4a90d9"


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


# Page: Overview
def page_overview(raw_df):
    st.header("Dataset Overview")
    (
        col1,
        col2,
        col3,
    ) = st.columns(3)
    col1.metric("Total Players", f"{len(raw_df):,}")
    col2.metric("Features", raw_df.shape[1])
    col3.metric("Missing Values", f"{raw_df.isnull().sum().sum():,}")

    st.subheader("Class Distribution")

    target_col = "SpendingSegment"

    counts = raw_df[target_col].value_counts()
    proportions = raw_df[target_col].value_counts(normalize=True)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts.plot.bar(ax=ax, color=[GREEN, ORANGE, RED])
        ax.set_title("Segment Counts")
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        proportions.plot.bar(ax=ax, color=[GREEN, ORANGE, RED])
        ax.set_title("Segment proportions")
        ax.set_ylabel("proportion")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    st.subheader("Sample Data")
    st.dataframe(raw_df.head(20), width="stretch")

    st.subheader("Statistical Summary")
    st.dataframe(raw_df.describe().T, width="stretch")


# Page: Exploration
def page_exploration(raw_df, cleaned_df, encoders):
    st.header("Data Exploration")

    st.subheader("Feature Correlation Heatmap")
    target_col = "SpendingSegment"
    numeric_df = cleaned_df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        square=True,
        linewidths=0.5,
    )
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Interactive scatter plot
    st.subheader("Interactive Scatter Plot")

    num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) < 2:
        st.warning("Not enough numeric columns")
        return
    col1, col2, col3 = st.columns(3)
    with col1:
        x_col = st.selectbox("X-axis", num_cols, index=0)
    with col2:
        y_default = 1 if len(num_cols) > 1 else 0
        y_col = st.selectbox("Y-axis", num_cols, index=y_default)
    with col3:
        segments = ["All", "Whale", "Dolphin", "Minnow"]
        segment_filter = st.selectbox("Filter by segment", segments)

    plot_df = raw_df.copy()
    if segment_filter != "All" and target_col:
        plot_df = plot_df[plot_df[target_col] == segment_filter]

    fig, ax = plt.subplots(figsize=(8, 6))
    if target_col and segment_filter == "All":
        for seg in raw_df[target_col].dropna().unique():
            subset = plot_df[plot_df[target_col] == seg]
            ax.scatter(subset[x_col], subset[y_col], alpha=0.4, label=seg, s=15)
        ax.legend()
    else:
        ax.scatter(plot_df[x_col], plot_df[y_col], alpha=0.4, s=15, color=RED)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Boxplots
    st.subheader("Feature Distributions by Segment")

    if target_col:
        box_feature = st.selectbox("Select feature", num_cols, key="box_feature")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=raw_df,
            x=target_col,
            y=box_feature,
            hue=target_col,
            legend=False,
            ax=ax,
        )
        ax.set_title(f"{box_feature} by Spending Segment")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# Page: Predictions
def page_predictions(rf, encoders, split_data):
    st.header("Player Spending Segment Prediction")
    st.write("Enter player attributes below to predict their spending segment")

    feature_names = split_data["feature_names"]
    target_encoder = encoders.get("SpendingSegment")
    class_names = list(target_encoder.classes_)

    input_values = {}

    col1, col2 = st.columns(2)
    for i, feat in enumerate(feature_names):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            if feat in encoders:
                le = encoders[feat]
                options = list(le.classes_)
                selected = st.selectbox(feat, options, key=f"pred_{feat}")
                input_values[feat] = le.transform([selected])[0]
            else:
                X_train = split_data["X_train"]
                if isinstance(X_train, pd.DataFrame):
                    col_idx = list(X_train.columns).index(feat) if feat in X_train.columns else None
                    if col_idx is not None:
                        col_data = X_train[feat]
                    else:
                        st.error(f"Feature {feat} was not found in training data")
                else: 
                    feat_idx = feature_names.index(feat)
                    col_data = pd.Series(X_train[:, feat_idx])
                
                min_val = float(col_data.min())
                max_val = float(col_data.max())
                mean_val = float(col_data.mean())

                input_values[feat] = st.slider(
                    feat,
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"pred_{feat}"
                )
    if st.button("Classify Player", type="primary"):
        input_array = np.array([[input_values[f] for f in feature_names]])
        prediction = rf.predict(input_array)[0]
        probabilities = rf.predict_proba(input_array)[0]
        predicted_label = class_names[prediction]
        confidence = probabilities[prediction]

        log_prediction(input_values, predicted_label, confidence)

        st.success(f"Predicted Segment: **{predicted_label.upper()}** "
                   f"(confidence: {confidence:.1%})")
        
        # Probability breakdown
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
            "Segment": class_names,
            "Probability": probabilities,
        }).sort_values("Probability", ascending=False)

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.barh(prob_df["Segment"], prob_df["Probability"], color=BLUE)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title("Prediction Probabilities")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # SHAP waterfall for this prediction
        st.subheader("Prediction Explanation (SHAP)")
        explainer = shap.TreeExplainer(rf)
        shap_vals = explainer.shap_values(input_array)

        if isinstance(shap_vals, list):
            sv = shap_vals[prediction][0]
            base = explainer.expected_value[prediction]
        else:
            sv = shap_vals[0, :, prediction]
            base = explainer.expected_value[prediction]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=sv,
                base_values=base,
                data=input_array.flatten(),
                feature_names=feature_names,
            ),
            show=False
        )
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

def main():
    if not check_pwd():
        return

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        [
            "Overview",
            "Exploration",
            "Predictions",
        ],
    )

    raw_df, cleaned_df = load_data()
    rf, lr, dt, encoders, split_data = load_models()

    if page == "Overview":
        page_overview(raw_df)
    elif page == "Exploration":
        page_exploration(raw_df, cleaned_df, encoders)
    elif page == "Predictions":
        page_predictions(rf, encoders, split_data)


if __name__ == "__main__":
    main()
