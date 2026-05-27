# 🧠 Scikit-Learn Machine Learning Curriculum

Welcome to the **Scikit-Learn Machine Learning Library**! This repository is organized as a structured, sequential curriculum to teach and demonstrate modern Machine Learning pipelines—from data ingestion and preprocessing to model training, evaluation, and tuning.

---

## 🗺️ Curriculum Structure

The curriculum is split into two primary segments: data engineering/preparation and model fitting/algorithms.

```
Sklearn_Library/
├── 📂 01_Data_Preparation/         # Step 1: Preprocessing & Evaluation
│   ├── README.md                  # Comprehensive Preprocessing Guide
│   ├── 01_Data_Sets.ipynb
│   ├── 02_Data_Cleaning.ipynb
│   ├── 03_Regression_Metrics.ipynb
│   ├── 04_Classification_metrics.ipynb
│   ├── 05_Feature_Selection.ipynb
│   ├── 06_Data_Scaling.ipynb
│   └── 07_Data_Spliting.ipynb
└── 📂 02_ML_Algorithms/            # Step 2: Training & Machine Learning Models
    ├── 📂 01_Linear_Regression/    # Linear and Polynomial Modeling
    │   ├── README.md              # Regression Concepts & Results
    │   ├── 01_Linear_Regression_EX01.ipynb
    │   └── 01_Linear_Regression_PolynomialFeatures_EX02.ipynb
    └── 📂 02_Logistic_Regression/  # Logistic Regression Classification
        └── README.md              # Sigmoid, Decision Boundary, & Log Loss theory
```

---

## 🔗 Module Outlines & Direct Links

### 🛠️ [1. Data Preparation](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/01_Data_Preparation/README.md)
Before training any model, data must be cleaned, transformed, and assessed. This module teaches:
*   **Loading & Generating Data**: Utilizing standard datasets (Iris, Wine, Digits) and synthetic generators (`make_classification`).
*   **Imputation**: Resolving missing values programmatically using `SimpleImputer`.
*   **Evaluation Math**: Deep-dives into regression metrics (MAE, MSE, MedAE) and classification metrics (Precision, Recall, F1-Score, multi-class averaging).
*   **Feature Selection**: Filtering out noise and irrelevant columns using statistical tests (`SelectKBest`, `chi2`) and model weightings (`SelectFromModel`).
*   **Scaling & Normalizing**: Standardizing variance with `StandardScaler` or bounding features with `MinMaxScaler` and `RobustScaler`.
*   **Splitting**: Partitioning data with `train_test_split` while maintaining strict boundaries to prevent data leakage.

### 📈 [2. Linear & Polynomial Regression](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/02_ML_Algorithms/01_Linear_Regression/README.md)
Learn to predict continuous targets:
*   **Simple & Multiple Linear Regression**: Fitting weight coefficients using Ordinary Least Squares.
*   **Polynomial Features**: Projecting low-dimensional linear space into high-dimensional curves using `PolynomialFeatures` to handle non-linear structures.
*   **Overfitting vs. Underfitting**: Detailed comparative tables showing training vs testing scores on real-world datasets (`satf.csv`).

### 📊 [3. Logistic Regression](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/02_ML_Algorithms/02_Logistic_Regression/README.md)
Learn binary and multi-class classification principles:
*   **The Sigmoid Function**: Mapping linear output $z$ to a probability space between 0 and 1.
*   **Decision Boundaries**: Thresholding predicted probabilities at 0.5 to label target predictions.
*   **Log Loss Optimization**: Minimizing cross-entropy cost rather than least squares.
