# 📉 Support Vector Regressor (SVR)

This directory demonstrates the implementation of Support Vector Regression (SVR) models using Scikit-Learn. It covers training, evaluating, and applying different kernel tricks on various datasets.

---

## 📂 Directory Contents

1.  **[01_Support_Vector_Regressor(SVM)_EX01.ipynb](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/02_ML_Algorithms/03_Support_Vector_Machine/01_SVR/01_Support_Vector_Regressor(SVM)_EX01.ipynb)**: Building a standard SVR model on a housing dataset to predict home prices, covering missing data imputation and continuous evaluation metrics.
2.  **[02_Support_Vector_Regressor(SVM)_EX02.ipynb](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/02_ML_Algorithms/03_Support_Vector_Machine/01_SVR/02_Support_Vector_Regressor(SVM)_EX02.ipynb)**: Applying SVR to an Earthquakes dataset. This notebook compares the performance of different kernel tricks (`linear`, `poly`, and `rbf`).
3.  **[03_SVR_Kaggle.ipynb](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/02_ML_Algorithms/03_Support_Vector_Machine/01_SVR/03_SVR_Kaggle.ipynb)**: An end-to-end regression pipeline on a complex Kaggle dataset (House Prices). It demonstrates handling missing values post train/test split to prevent data leakage and evaluating model performance using multiple error metrics.

---

## 📖 Key Concepts Explained

### 1. Support Vector Regression (SVR) Principles
Unlike Ordinary Least Squares regression which aims to minimize the error rate overall, SVR tries to fit the error within a certain threshold. It constructs a "tube" (the $\epsilon$-insensitive tube) around the regression line. 
*   **Margin of Tolerance ($\epsilon$)**: Errors that fall within the $\epsilon$-tube are ignored. The model only penalizes data points that fall outside this boundary.
*   **Support Vectors**: Data points falling on or outside the tube boundary that dictate the shape of the regression function.

```python
from sklearn.svm import SVR

# Instantiate SVR model (defaults to RBF kernel)
model = SVR()

# Fit model on training set
model.fit(X_train, y_train)

# Evaluate model score
r2 = model.score(X_test, y_test)
```

### 2. The Kernel Trick (`02_Support_Vector_Regressor(SVM)_EX02.ipynb`)
For non-linear data, standard linear boundaries perform poorly. SVMs use the **Kernel Trick** to project input features into higher-dimensional spaces where a linear plane can be used to separate or regress the data effectively.
*   `kernel='linear'`: Best for strictly linear relationships.
*   `kernel='poly'`: Polynomial kernel for curves.
*   `kernel='rbf'` (Radial Basis Function): The default and often most powerful kernel capable of mapping into infinite-dimensional spaces.

---

## 📊 Model Evaluation Results

### 1. Housing Dataset (Standard SVR)
*   **Dataset**: `houses.csv`
*   **Results**:
    *   **Train $R^2$ Score**: ~0.332
    *   **Test $R^2$ Score**: ~0.311
    *   **Mean Squared Error (MSE)**: 838.68
    *   **Mean Absolute Error (MAE)**: 14.03
    *   **Median Absolute Error (MedAE)**: 7.16

### 2. Earthquakes Dataset (Kernel Comparison)
*   **Dataset**: `Earthquakes.csv`
*   **Best Kernel**: `rbf`
*   **Results (`rbf`)**:
    *   **Train $R^2$ Score**: ~0.432
    *   **Test $R^2$ Score**: ~0.479
    *   **Mean Squared Error (MSE)**: 431.74
    *   **Mean Absolute Error (MAE)**: 11.05
    *   **Median Absolute Error (MedAE)**: 3.86

### 3. House Prices (Kaggle Dataset)
*   **Dataset**: `train.csv`
*   **Observations**: The dataset contains 80 features and required heavy preprocessing (encoding categorical variables and handling missing data). The initial SVR model achieved negative $R^2$ scores, highlighting the necessity for advanced feature scaling, hyperparameter tuning, and potentially feature selection when dealing with high-dimensional datasets using SVR.
