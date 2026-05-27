# 📊 Logistic Regression Classification Module

Welcome to the **Logistic Regression** module! This section of the curriculum covers supervised learning classification theory and hands-on implementations using Scikit-Learn.

---

## 📖 Theoretical Overview

### 1. Classification via Regression
Unlike Linear Regression which predicts continuous targets, **Logistic Regression** maps input features to discrete classes. It does this by fitting a linear combination of features and then applying a non-linear mapping function.

### 2. The Sigmoid (Logistic) Function
To convert a continuous real-valued prediction $z = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n$ into a probability $p \in [0, 1]$, we use the **Sigmoid function**:

$$S(z) = \frac{1}{1 + e^{-z}}$$

*   As $z \to \infty$, $S(z) \to 1$.
*   As $z \to -\infty$, $S(z) \to 0$.
*   When $z = 0$, $S(z) = 0.5$.

### 3. Decision Boundary
The default threshold for binary classification is $0.5$:

$$\hat{y} = \begin{cases} 1 & \text{if } S(z) \geq 0.5 \\ 0 & \text{if } S(z) < 0.5 \end{cases}$$

This threshold forms the **decision boundary**, dividing the feature space into region classes.

### 4. Log Loss (Binary Cross-Entropy)
We cannot use Mean Squared Error (MSE) to train Logistic Regression because the Sigmoid function introduces non-linearity, making the MSE loss surface non-convex (filled with local minima). Instead, we optimize **Log Loss**:

$$\text{Log Loss} = - \frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(p^{(i)}) + (1 - y^{(i)}) \log(1 - p^{(i)}) \right]$$

This penalty grows exponentially as the predicted probability diverges from the actual class label.

---

## 📂 Notebooks

This folder contains two practical exercises using realistic datasets:

### 1. 🧪 [01_Logistic_Regression_EX01.ipynb](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/02_ML_Algorithms/02_Logistic_Regression/01_Logistic_Regression_EX01.ipynb) (Breast Cancer Dataset)
Classifies tumors as malignant ($0$) or benign ($1$) using features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.
*   **Key Pipeline Steps**:
    *   Ingestion via `load_breast_cancer()`.
    *   Feature plotting of target dimensions.
    *   Null checks (`np.isnan().sum()`) and numerical type validation.
    *   Splitting: $67\%$ train, $33\%$ test (`random_state=44`).
    *   Scaling: `StandardScaler` to ensure the gradient solver converges.
    *   Model instantiation:
        ```python
        LogisticRegression(penalty='l2', solver='sag', C=0.5, random_state=33)
        ```
        *   `penalty='l2'`: Ridge regularization to penalize large weights.
        *   `solver='sag'`: Stochastic Average Gradient descent, optimized for larger datasets.
        *   `C=0.5`: Moderate regularization strength (inverse penalty).
    *   Evaluation: Score printing, confusion matrix mapping with Seaborn heatmap, and complete precision/recall/f1 metrics.

### 2. 🫀 [01_Logistic_Regression_EX02.ipynb](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/02_ML_Algorithms/02_Logistic_Regression/01_Logistic_Regression_EX02.ipynb) (Heart Disease Dataset)
Predicts the presence of heart disease in patients using a CSV clinical dataset.
*   **Key Pipeline Steps**:
    *   Ingestion of custom file `heart.csv` via Pandas.
    *   Analysis of feature types and null properties.
    *   Splitting and scaling identical to EX01.
    *   Logistic Regression fitting with $L_2$ regularization.
    *   Outputting predicted probabilities using `.predict_proba()` alongside binary classifications (`.predict()`).
    *   Generation of classification reports on train and test folds.
