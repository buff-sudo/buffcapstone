# Quick-Start Guide

## Mobile Game Player Spending Segment Classifier

Zachary Buffington | Western Governors University | Computer Science Capstone

---

## Option 1: Live Dashboard (Recommended)

The dashboard is deployed on Streamlit Community Cloud and requires no local installation.

1. Open the dashboard URL: `https://buffcapstone-kqezaytjfkpxts3tmdi3m5.streamlit.app/`
2. Enter the password: `capstone`
3. Use the sidebar to navigate between pages.

If the live link is unavailable, follow Option 2 below.

---

## Option 2: Local Installation

### Prerequisites

- Python 3.10 or higher
- uv (https://docs.astral.sh/uv/getting-started/installation/)
- Git

### Step 1: Clone the Repository

```
git clone https://github.com/buff-sudo/buffcapstone.git
cd buff-capstone
```

### Step 2: Install Dependencies

```
uv sync
```

This creates a virtual environment and installs all dependencies from `pyproject.toml` in one step.

### Step 3: Verify the Project Structure

Confirm the following files and directories exist:

```
buff-capstone/
├── .gitignore
├── BuffCapstone.ipynb
├── pyproject.toml
├── README.md
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   │   └── game_data.csv
│   └── cleaned/
│       └── cleaned_dataset.csv
├── docs/
│   ├── business_requirements.md
│   ├── quick_start_guide.md
│   ├── BuffCapstone_AB.docx
│   ├── BuffCapstone_CD.docx
│   └── screenshots/
│       ├── missing_values.png
│       ├── target_distribution.png
│       ├── numeric_distributions.png
│       ├── correlation_matrix.png
│       ├── boxplots.png
│       ├── class_imbalance.png
│       ├── confusion_matrices.png
│       ├── shap_summary.png
│       ├── top5_features.png
│       ├── shap_waterfall.png
│       └── feature_dependency_comparison.png
└── models/
    ├── random_forest.joblib
    ├── random_forest_full.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── encoders.joblib
    └── split_data.joblib
```

If the `models/` directory is empty or missing `.joblib` files, you need to run the notebook first (see Step 4).

### Step 4: Run the Notebook (if models are not yet trained)

Open the notebook in Jupyter:

```
uv run jupyter notebook capstone_full_pipeline.ipynb
```

Run all cells from top to bottom. This will:
- Load and profile the raw dataset
- Clean and preprocess the data
- Train the logistic regression, decision tree, and random forest models
- Evaluate all models and generate SHAP analysis
- Save all model artifacts to the `models/` directory
- Save all plots to `docs/screenshots/`

The random forest grid search takes approximately 15-20 minutes on a standard laptop.

### Step 5: Launch the Dashboard

```
uv run streamlit run dashboard/app.py
```

The dashboard will open in your default browser at `http://localhost:8501`.

Enter the password `capstone` to access the application.

---

## Using the Dashboard

### Overview Page

Displays dataset statistics, class distribution charts, a sample data table, and a statistical summary of all features. No interaction required.

### Exploration Page

Three interactive tools for examining the data:

- **Correlation Heatmap:** Shows Pearson correlations between all numeric features.
- **Scatter Plot:** Select any two numeric features from the dropdown menus. Use the segment filter to show all players or restrict to a single spending segment.
- **Boxplots:** Select a feature to view its distribution broken down by spending segment.

### Predictions Page

Enter player attributes using the sliders and dropdown menus, then click "Classify Player." The page displays:

- The predicted spending segment with a confidence percentage.
- A bar chart showing the probability assigned to each segment.
- A SHAP waterfall plot explaining which features pushed the prediction toward the predicted class.

Each prediction is logged automatically for monitoring purposes.

### Model Performance Page

Displays evaluation metrics for all three models on the held-out test set:

- Confusion matrices side by side.
- Full classification reports with per-class precision, recall, and F1-score.
- A comparison table of test accuracy and macro F1 across models.
- Pass/fail indicators for the two success criteria (macro F1 >= 0.80, whale recall > 0.70).

### Monitoring Page

Displays a log of all predictions made through the dashboard, including timestamps, input features, predicted class, and confidence. A bar chart shows the distribution of predicted segments. The log can be cleared with the "Clear prediction log" button.

---

## Troubleshooting

**"ModuleNotFoundError" when launching the dashboard:** Make sure you ran `uv sync` from the project root. If running commands directly, prefix them with `uv run` to use the managed virtual environment.

**Models not found:** Run the notebook first (Step 4) to generate the `.joblib` files in the `models/` directory.

**Streamlit shows a blank page:** Check the terminal for error messages. The most common cause is a missing data file. Verify that `data/raw/game_data.csv` exists.

**SHAP waterfall plot throws a ValueError:** Ensure the SHAP version matches pyproject.toml (>= 0.44). Older versions handle multi-class output differently.

**Grid search is running again even though models exist:** The notebook checks for `models/random_forest.joblib` before training. If the file path changed or was deleted, the notebook will retrain. Verify the file exists at the expected path.
