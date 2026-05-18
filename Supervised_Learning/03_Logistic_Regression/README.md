# Logistic Regression Projects

## 📚 Folder Overview
This directory contains projects focused on binary classification (predicting 1 or 0, Faulty or Normal) using **Logistic Regression**.

## 🧠 Core Concept
Despite the name, Logistic Regression is for **classification**. It applies a Sigmoid function ($\sigma(z) = \frac{1}{1 + e^{-z}}$) to a linear equation to output a probability between 0 and 1.
- **Advantages:** Very fast, efficient, provides exact probability scores.
- **Disadvantages:** Assumes a linear decision boundary.

## 📂 Project Files
*   **`LogisticRegression_predictive_maintenance.py` (.ipynb)**
    *   **Dataset:** `predictive_maintenance.csv`
    *   **Description:** Predicts whether industrial machinery will fail (binary classification) based on sensor data.
*   **`logistic_regression.py` (.ipynb)**
    *   **Dataset:** `breast-cancer.csv`
    *   **Description:** A crucial medical application predicting the presence of breast cancer.

## 💻 Code Highlight
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test) # Get exactly how confident the model is
```
