# 📘 Comprehensive Machine Learning Reference Guide

Welcome to the detailed README guide for Supervised Machine Learning! This document serves as a deep dive into the models, algorithms, and preprocessing techniques used across your projects. It is structured to provide GeeksforGeeks-level explanations, mathematical intuition, practical usage notes, and **the exact Python functions used to run them**.

---

## 📑 Table of Contents
1. [Data Preprocessing Techniques](#1-data-preprocessing-techniques)
2. [Evaluation Metrics](#2-evaluation-metrics)
3. [Regression Models](#3-regression-models)
4. [Logistic Regression](#4-logistic-regression)
5. [Classification Models](#5-classification-models)

---

## ⚙️ 1. Data Preprocessing Techniques

Proper data preparation is critical.

### 1.1 Feature Scaling
Machine learning algorithms using distance (KNN, SVM) or gradient descent require scaled data.
*   **StandardScaler:** Rescales to mean 0, standard deviation 1.
*   **MinMaxScaler:** Rescales strictly between 0 and 1.
*   **Python Implementation:**
    ```python
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    scaler = StandardScaler()
    # CRITICAL: fit_transform on train, but ONLY transform on test!
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```

### 1.2 Categorical Encoding
Models only understand numbers.
*   **Label Encoding:** Converts categories to integers (Red=0, Blue=1). Best for the target variable `y`.
*   **One-Hot Encoding:** Creates binary columns for each category. Best for features `X`.
*   **Python Implementation:**
    ```python
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd

    # Label Encoding (for y)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # One-Hot Encoding (for X)
    X = pd.get_dummies(X, columns=['CategoryColumn'], drop_first=True)
    ```

### 1.3 Handling Imbalanced Data (SMOTE)
*   **SMOTE:** Synthetically generates minority class examples to balance the dataset.
*   **Python Implementation:**
    ```python
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    # CRITICAL: Only apply to training data!
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    ```

---

## 📊 2. Evaluation Metrics

### 2.1 Classification Metrics
*   **Python Implementation for all Classification metrics:**
    ```python
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import confusion_matrix, classification_report
    
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    ```
*   **Accuracy:** Overall correct predictions.
*   **Precision:** Out of predicted positives, how many are actual positives.
*   **Recall:** Out of actual positives, how many did we find.
*   **F1-Score:** Harmonic mean of Precision and Recall.

### 2.2 Regression Metrics
*   **Python Implementation for all Regression metrics:**
    ```python
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    ```
*   **MAE:** Average absolute difference between prediction and actual.
*   **MSE:** Average squared difference. Heavily penalizes outliers.
*   **R-Squared ($R^2$):** Proportion of variance in target explained by the model (closer to 1.0 is better).

---

## 📈 3. Regression Models

Regression models are used when the target variable is continuous (e.g., predicting house prices, temperature, or age).

### 3.1 Linear Regression
**Overview:** Linear regression attempts to model the relationship between two or more variables by fitting a linear equation to observed data. 
*   **How it Works:** It calculates the best-fit line by minimizing the **Residual Sum of Squares (RSS)**.
*   **Equation:** $Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \epsilon$
*   **Advantages:** Simple to implement, computationally efficient, highly interpretable.
*   **Disadvantages:** Assumes a linear relationship; very sensitive to outliers.
*   **Python Implementation:**
    ```python
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    ```

### 3.2 Ridge Regression (L2 Regularization)
**Overview:** Analyzes multiple regression data that suffer from multicollinearity. 
*   **How it Works:** Adds a penalty equivalent to the **square of the magnitude** of coefficients. Shrinks coefficients to prevent overfitting but does not make them zero.
*   **Penalty Term:** $\lambda \sum \beta_i^2$
*   **Advantages:** Prevents overfitting.
*   **Python Implementation:**
    ```python
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    ```

### 3.3 Lasso Regression (L1 Regularization)
**Overview:** Least Absolute Shrinkage and Selection Operator.
*   **How it Works:** Adds a penalty equal to the **absolute value of the magnitude** of coefficients. Can shrink coefficients exactly to zero (acts as feature selection).
*   **Penalty Term:** $\lambda \sum |\beta_i|$
*   **Advantages:** Automatically drops useless features.
*   **Python Implementation:**
    ```python
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=1.0)
    ```

### 3.4 ElasticNet Regression
**Overview:** A middle ground between Ridge and Lasso combining both L1 and L2 penalties.
*   **Python Implementation:**
    ```python
    from sklearn.linear_model import ElasticNet
    model = ElasticNet(alpha=1.0, l1_ratio=0.5)
    ```

---

## ⚖️ 4. Logistic Regression

**Overview:** Despite having "Regression" in its name, Logistic Regression is fundamentally a **Classification** algorithm for predicting binary outcomes.

*   **How it Works:** Instead of a straight line, it fits an "S" shaped logistic function (Sigmoid curve). Output is a probability between 0 and 1.
*   **Mathematical Concept:** 
    *   Sigmoid Function: $P = \frac{1}{1 + e^{-Z}}$
*   **Advantages:** Very fast, efficient, provides probability scores.
*   **Disadvantages:** Assumes a linear decision boundary.
*   **Python Implementation:**
    ```python
    from sklearn.linear_model import LogisticRegression
    # Use predict() for 0/1, use predict_proba() for probabilities
    model = LogisticRegression()
    model.fit(X_train, y_train)
    classes = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    ```

---

## 🌳 5. Classification Models

Classification models are used when the target variable is categorical (e.g., predicting Faulty/Normal).

### 5.1 Decision Trees
**Overview:** A tree-shaped algorithm where internal nodes are "tests" on attributes, and leaf nodes are class labels.
*   **How it Works:** Splits the data based on features resulting in the highest **Information Gain** or lowest **Gini Impurity**.
*   **Advantages:** Extremely intuitive, no scaling needed.
*   **Disadvantages:** High risk of overfitting.
*   **Python Implementation:**
    ```python
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=5)
    ```

### 5.2 Random Forest
**Overview:** An ensemble method constructing a multitude of decision trees.
*   **How it Works (Bagging):** Creates multiple trees using random subsets of training data and features. Final prediction is the mode/average of all trees.
*   **Advantages:** Fixes single-tree overfitting; robust to outliers.
*   **Disadvantages:** Slower to train; acts as a "black box".
*   **Python Implementation:**
    ```python
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
    ```

### 5.3 Gradient Boosting (XGBoost & LightGBM)
**Overview:** Ensemble technique building models sequentially to correct errors of previous models.
*   **How it Works:** Tree 1 makes predictions. Tree 2 corrects Tree 1's errors. Tree 3 corrects Tree 2, and so on.
*   **Advantages:** Often yields the highest accuracy in machine learning.
*   **Disadvantages:** Highly sensitive to hyperparameters.
*   **Python Implementation:**
    ```python
    # Standard Scikit-Learn
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

    # XGBoost
    from xgboost import XGBClassifier
    model = XGBClassifier()

    # LightGBM
    from lightgbm import LGBMClassifier
    model = LGBMClassifier()
    ```

### 5.4 K-Nearest Neighbors (KNN)
**Overview:** A non-parametric, lazy learning algorithm.
*   **How it Works:** Classifies a point by looking at the 'K' closest data points using **Euclidean distance**.
*   **Advantages:** Simple, naturally handles multi-class.
*   **Disadvantages:** Computationally expensive at testing time; **requires scaling**.
*   **Python Implementation:**
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=5)
    ```

### 5.5 Support Vector Machines (SVM)
**Overview:** Finds a hyperplane that distinctly classifies data points by maximizing the margin.
*   **How it Works:** Uses the **Kernel Trick** (e.g., RBF) to map non-linear data into higher dimensions.
*   **Advantages:** Effective in high-dimensional spaces.
*   **Disadvantages:** Slow on large datasets; requires scaling.
*   **Python Implementation:**
    ```python
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', C=1.0)
    ```
