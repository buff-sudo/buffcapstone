# Business Requirements Document

## Mobile Game Player Spending Segment Classifier

Zachary Buffington | Western Governors University | Computer Science Capstone | April 2026

---

## 1. Business Problem

Free-to-play mobile game studios generate revenue through in-app purchases, but spending is concentrated among a small fraction of players. Swrve's 2019 monetization report found that only 1.63% of players make any purchase, and the top 10% of paying users account for 64.5% of total revenue (Swrve, 2019). Studios currently segment players using static spending thresholds, which cannot identify high-value players early in their lifecycle and cannot distinguish players on upward spending trajectories from one-time purchasers.

The business need is an automated classification system that predicts whether a player will become a high spender (whale), moderate spender (dolphin), or minimal/non-spender (minnow) based on demographic and behavioral attributes observable early in the player's lifecycle.

## 2. Stakeholders

**Product Managers and Marketing Leads (primary users):** These users need segment predictions to allocate promotional budgets, design personalized offers, and prioritize retention campaigns. They interact with the system through the dashboard interface and do not require technical expertise.

**Data Analytics and Engineering Team:** Responsible for maintaining the model pipeline, monitoring prediction quality, and retraining the model on updated data. They interact with the Jupyter Notebook, source code, and model artifacts directly.

**Game Designers:** Indirect stakeholders who benefit from the SHAP feature importance analysis. Knowing which behavioral attributes predict high spending can inform decisions about progression systems, reward pacing, and content gating.

**Players:** End users affected by the downstream decisions the model enables. Responsible use of predictions should improve the player experience through personalization, not exploit spending vulnerabilities.

## 3. Business Objectives

- Classify players into spending segments (whale, dolphin, minnow) using demographic and behavioral features that are available early in the player lifecycle, excluding cumulative purchase amount.
- Achieve a macro-averaged F1-score of at least 0.80 on held-out test data.
- Achieve per-class recall for the whale segment exceeding 0.70.
- Identify the top 5 features most predictive of spending segment membership through SHAP analysis.
- Deliver a reproducible analysis pipeline that can be re-executed end-to-end.

## 4. Scope

**In scope:**

- Data acquisition from the Kaggle Mobile Game In-App Purchases Dataset 2025 (Puri, 2025).
- Exploratory data analysis, data cleaning, missing value imputation, categorical encoding, and class imbalance handling via SMOTE.
- Training and evaluation of three classification models: random forest (primary), logistic regression (baseline), decision tree (baseline).
- SHAP-based feature importance analysis for model interpretability.
- An interactive Streamlit dashboard with classification, exploration, and model performance pages.
- Project documentation including this business requirements document, a quick-start guide, and a rubric-mapped report.

**Out of scope:**

- Integration with a live game analytics platform or real-time player data stream.
- Deployment to a production game backend.
- A/B testing of monetization strategies informed by the model.
- Deep learning or neural network approaches.

## 5. Functional Requirements

**FR-1: Data Preprocessing Pipeline.** The system must load the source CSV, handle missing values through median imputation (numeric) and mode imputation (categorical), encode categorical features, drop non-predictive columns (UserID) and target-proxy columns (InAppPurchaseAmount), and apply SMOTE to balance the training partition.

**FR-2: Descriptive Analysis.** The system must produce statistical summaries, distribution histograms, boxplots by spending segment, and a feature correlation heatmap.

**FR-3: Predictive Classification.** The system must train a random forest classifier on the preprocessed data and return a predicted spending segment for a given set of player features.

**FR-4: Baseline Comparison.** The system must train logistic regression and decision tree models on the same data and report comparative performance metrics.

**FR-5: Model Evaluation.** The system must generate confusion matrices, per-class precision/recall/F1 reports, and macro-averaged F1 scores for all three models.

**FR-6: Feature Importance Analysis.** The system must produce global SHAP summary plots and per-prediction SHAP waterfall plots for the random forest model.

**FR-7: Interactive Dashboard.** The system must provide a web-based dashboard with the following capabilities:
- View dataset statistics and class distributions (Overview page).
- Explore feature relationships through interactive scatter plots and boxplots with segment filtering (Exploration page).
- Input player attributes and receive a segment prediction with SHAP explanation (Predictions page).
- View model evaluation metrics and success criteria results (Performance page).
- View a log of past predictions (Monitoring page).

**FR-8: Prediction Logging.** Each prediction made through the dashboard must be logged with a timestamp, input features, predicted class, and confidence score.

**FR-9: Access Control.** The dashboard must require password authentication before granting access.

## 6. Non-Functional Requirements

**NFR-1: Reproducibility.** The full analysis pipeline must be executable end-to-end from a single Jupyter Notebook using a fixed random seed (42) to reproduce all reported results.

**NFR-2: Performance.** Predictions through the dashboard must return in under 2 seconds. Model training may take longer but is a one-time operation; trained models are serialized and loaded at runtime.

**NFR-3: Portability.** The system must run on Python 3.10+ with dependencies specified in requirements.txt. The dashboard must be deployable to Streamlit Community Cloud from a GitHub repository.

**NFR-4: Data Privacy.** The system uses a synthetic dataset containing no personally identifiable information. The design anticipates production privacy requirements (GDPR, COPPA, CCPA) as documented in the project proposal.

## 7. Data Requirements

**Source:** Mobile Game In-App Purchases Dataset 2025, Kaggle (Puri, 2025).

**Format:** Single CSV file, one row per player.

**Key columns:**

- Demographic: Age, Gender, Country, Device
- Behavioral: GameGenre, FirstPurchaseDaysAfterInstall
- Transactional: PaymentMethod, LastPurchaseDate
- Target: spending_segment (whale, dolphin, minnow)
- Excluded: UserID (non-predictive), InAppPurchaseAmount (target proxy)

**Known data quality issues:** Intentional missing values at 2-5% across columns. Class imbalance with approximately 85% minnows, 13% dolphins, and 2% whales.

## 8. Success Criteria

- Macro-averaged F1-score >= 0.80 on the held-out test set.
- Per-class whale recall > 0.70.
- SHAP analysis produces a ranked feature importance list.
- Dashboard is accessible, functional, and usable without technical training.
- All code is documented and reproducible.

## 9. Assumptions and Constraints

- The synthetic dataset's feature distributions approximate real-world player behavior closely enough to validate the pipeline, though model performance on real data may differ.
- No budget is available for cloud compute, proprietary software, or external contractors.
- The project must be completed within 10 weeks by a single developer.
- Segment labels in the synthetic dataset may be derived mechanically from spending features, which limits the realism of classification results. This is mitigated by excluding InAppPurchaseAmount from the feature set.

## 10. References

Puri, P. (2025). Mobile Game In-App Purchases Dataset 2025. Kaggle. https://www.kaggle.com/datasets/pratyushpuri/mobile-game-in-app-purchases-dataset-2025

Swrve. (2019). Swrve 2019 gaming monetization report. https://cdn2.hubspot.net/hubfs/5516657/Monetization%20Report_final.pdf
