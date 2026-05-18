# Ensemble Methods Projects (Random Forest, XGBoost)

## 📚 Folder Overview
This directory contains advanced machine learning projects utilizing **Ensemble Methods**—algorithms that combine hundreds of models to create a "super predictor."

## 🧠 Core Concepts
*   **Random Forest (Bagging):** Builds hundreds of independent decision trees on random subsets of data. Averages their results to prevent overfitting.
*   **XGBoost & LightGBM (Boosting):** Builds trees *sequentially*. Tree 2 tries to fix the exact mistakes made by Tree 1. Unbelievably powerful.
*   **Isolation Forest:** Specifically designed for anomaly detection.

## 📂 Project Files
*   **`RandomForest_LGBM_XGB_IsolationForest.py` (.ipynb)**
    *   **Dataset:** `faults.csv`
    *   **Description:** Compares multiple ensemble methods (RF, LGBM, XGB) on detecting faults. Also utilizes `IsolationForest` to spot extreme anomalies.
*   **`XGB_LGBM_RandomForest_train.py` (.ipynb)**
    *   **Dataset:** `train.csv` / `test.csv`
    *   **Description:** A heavy-duty pipeline leveraging the power of XGBoost and LightGBM for maximum accuracy.

## 💻 Code Highlight
```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize the best performing algorithms in modern ML
xgb_model = XGBClassifier(learning_rate=0.1, n_estimators=100)
xgb_model.fit(X_train, y_train)
```
