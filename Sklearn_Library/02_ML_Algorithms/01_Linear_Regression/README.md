# 📈 Linear Regression & Polynomial Modeling

This directory demonstrates simple, multiple, and polynomial linear regression implementations using Scikit-Learn.

---

## 📂 Directory Contents

1.  **[01_Linear_Regression_EX01.ipynb](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/02_ML_Algorithms/01_Linear_Regression/01_Linear_Regression_EX01.ipynb)**: Building a Multiple Linear Regression model on housing data to predict home prices.
2.  **[01_Linear_Regression_PolynomialFeatures_EX02.ipynb](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/02_ML_Algorithms/01_Linear_Regression/01_Linear_Regression_PolynomialFeatures_EX02.ipynb)**: Implementing Polynomial Regression to model non-linear boundaries by transforming input space.

---

## 📖 Key Concepts Explained

### 1. Simple vs. Multiple Linear Regression (`01_Linear_Regression_EX01.ipynb`)
Linear Regression models the relationship between dependent continuous variables ($y$) and one or more independent features ($X$) by fitting a linear equation.
*   **Simple Linear Regression**: Models relationship with a single feature.
    $$y = w_0 + w_1 x_1$$
*   **Multiple Linear Regression**: Models relationship using multiple features.
    $$y = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d$$
    Where:
    *   $y$ is the target value (e.g. house price).
    *   $w_0$ is the y-intercept.
    *   $w_1, \dots, w_d$ are the regression coefficients representing feature importance weightings.
*   **Key Parameters**:
    *   `fit_intercept=True`: Calculates the y-intercept ($w_0$). If False, the line goes through the origin.
    *   `n_jobs=-1`: Uses all available CPU cores to speed up computation.

```python
from sklearn.linear_model import LinearRegression

# Instantiate Linear Regression model
model = LinearRegression(fit_intercept=True, n_jobs=-1)

# Fit model on training set
model.fit(X_train, y_train)

# Evaluate model score (Coefficient of Determination R-squared)
r2 = model.score(X_test, y_test)
```

### 2. Polynomial Feature Expansion (`01_Linear_Regression_PolynomialFeatures_EX02.ipynb`)
When the relationship between features and target is non-linear, fitting a straight line results in high bias (underfitting). We can map features to polynomial terms to allow a linear model to fit non-linear curves.
*   **Mathematical Concept**:
    For a single input feature $x$, a polynomial expansion of degree 3 transforms the feature vector into:
    $$\Phi(x) = [1, x, x^2, x^3]$$
    The model then fits a linear equation in this expanded feature space:
    $$y = w_0 + w_1 x + w_2 x^2 + w_3 x^3$$
    Even though the curve is non-linear relative to $x$, the regression model remains *linear* in terms of coefficients $w_i$, allowing standard linear solver algorithms to be utilized.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate polynomial feature matrix
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train linear regression on the expanded polynomial features
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
```

---

## 📊 Model Evaluation Results

### Multiple Linear Regression (Housing Dataset)
*   **Dataset**: `houses.csv`
*   **Workflow**:
    1. Cleaned missing data via `.dropna()`.
    2. Scaled numerical features using `MinMaxScaler` fitted strictly on `X_train`.
    3. Fitted multiple regression model.
*   **Results**:
    *   **Train $R^2$ Score**: ~0.727
    *   **Test $R^2$ Score**: ~0.717
    *   **Mean Squared Error (MSE)**: 330.09
    *   **Mean Absolute Error (MAE)**: 12.08
    *   **Median Absolute Error (MedAE)**: 9.01

### Polynomial Regression comparison (SAT & GPA Dataset)
*   **Dataset**: `satf.csv`
*   **Workflow**:
    1. Isolated independent variable `high_GPA` and target `univ_GPA`.
    2. Split dataset with `test_size=0.2`.
    3. Compared standard linear model vs. degree-3 polynomial model.
*   **Results Comparison**:

| Metric | Simple Linear Model (Degree 1) | Polynomial Model (Degree 3) |
| :--- | :--- | :--- |
| **Train $R^2$ Score** | 0.616 | 0.639 |
| **Test $R^2$ Score** | 0.568 | 0.561 |
| **Mean Squared Error (MSE)** | 0.11 | 0.11 |
| **Mean Absolute Error (MAE)** | 0.26 | 0.28 |
| **Median Absolute Error (MedAE)** | 0.18 | 0.24 |

*Observation*: While the degree-3 polynomial model achieved a slightly higher training score ($0.639$ vs $0.616$), it suffered from minor overfitting, resulting in slightly lower test performance ($0.561$ vs $0.568$) and higher absolute errors on the test set.
