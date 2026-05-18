# Linear Regression Projects

## 📚 Folder Overview
This directory contains projects focused on predicting continuous values (like prices or medical indicators) using **Linear Regression**.

## 🧠 Core Concept
Linear Regression calculates the best-fit line through data points by minimizing the **Residual Sum of Squares (RSS)**. It solves for the equation $Y = mX + b$.
- **Advantages:** Simple, fast, highly interpretable.
- **Disadvantages:** Assumes a strict linear relationship; sensitive to outliers.

## 📂 Project Files
*   **`Data_Preprocessing.py` (.ipynb)**
    *   **Dataset:** `diabetes.csv`
    *   **Description:** Focuses on predicting continuous diabetes metrics. It demonstrates heavy data preprocessing pipelines before fitting the Linear Regression model.
*   **`Terrain Prices Reggression.py` (.ipynb)**
    *   **Dataset:** `train.csv` / `test.csv`
    *   **Description:** A classic regression problem aimed at predicting terrain/housing prices based on geographical and structural features.
*   **`task.py` & `Task_V2.py` (.ipynb)**
    *   **Description:** Iterations of the core regression tasks. `Task_V2` acts as the more refined, comprehensive version of the pipeline.

## 💻 Code Highlight
In these files, you will find the standard implementation:
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
