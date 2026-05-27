# 🛠️ Module 01: Data Preparation

This module covers the essential first steps in the Machine Learning workflow: loading datasets, cleaning missing data, scaling features, performing feature selection, splitting data into train/test subsets, and evaluating model performance using regression and classification metrics.

---

## 📂 Directory Contents

1.  **[01_Data_Sets.ipynb](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/01_Data_Preparation/01_Data_Sets.ipynb)**: Loading built-in Scikit-Learn datasets and generating synthetic datasets for modeling.
2.  **[02_Data_Cleaning.ipynb](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/01_Data_Preparation/02_Data_Cleaning.ipynb)**: Handling missing values via univariate imputation.
3.  **[03_Regression_Metrics.ipynb](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/01_Data_Preparation/03_Regression_Metrics.ipynb)**: Metrics for evaluating regression models.
4.  **[04_Classification_metrics.ipynb](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/01_Data_Preparation/04_Classification_metrics.ipynb)**: Metrics for evaluating classification models.
5.  **[05_Feature_Selection.ipynb](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/01_Data_Preparation/05_Feature_Selection.ipynb)**: Selecting the most predictive features.
6.  **[06_Data_Scaling.ipynb](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/01_Data_Preparation/06_Data_Scaling.ipynb)**: Feature scaling and normalization techniques.
7.  **[07_Data_Spliting.ipynb](file:///home/ibrahimshnouda/GitHub/AI/Sklearn_Library/01_Data_Preparation/07_Data_Spliting.ipynb)**: Partitioning data into train and test sets without data leakage.

---

## 📖 Key Concepts Explained

### 1. Data Sets & Generation (`01_Data_Sets.ipynb`)
Scikit-Learn provides built-in toy datasets as well as generators for synthetic datasets:
*   **Toy Datasets**: Wrapped in `Bunch` objects (dictionary-like objects), accessed via `load_*`:
    *   `load_iris()`: Iris flower multi-class classification dataset (150 samples, 4 features).
    *   `load_digits()`: Handwritten digits 8x8 image classification dataset (1797 samples, 64 features).
    *   `load_wine()`: Wine chemical analysis classification dataset (178 samples, 13 features).
    *   `load_breast_cancer()`: Breast cancer diagnostic classification dataset (569 samples, 30 features).
*   **Synthetic Data Generators**:
    *   `make_regression(n_samples, n_features, ...)`: Generates a random regression dataset.
    *   `make_classification(n_samples, n_features, n_classes, n_informative, ...)`: Generates a random classification dataset with informative and redundant features.

```python
from sklearn.datasets import load_iris, make_regression

# Loading built-in dataset
iris = load_iris()
X, y = iris.data, iris.target

# Generating synthetic regression data
X_reg, y_reg = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)
```

### 2. Data Cleaning & Imputation (`02_Data_Cleaning.ipynb`)
Real-world data often has missing values (represented as `NaN`, `Null`, or sometimes `0`). Scikit-Learn provides `SimpleImputer` to replace these missing values using univariate statistical strategies.
*   **Imputation Strategies**:
    *   `mean`: Replaces missing values with the mean of the column (best for numeric, normally distributed data).
    *   `median`: Replaces missing values with the median of the column (best for numeric data with outliers).
    *   `most_frequent`: Replaces missing values with the mode (best for categorical features).
    *   `constant`: Replaces missing values with a user-provided fill value.

```python
import numpy as np
from sklearn.impute import SimpleImputer

# Define imputer to replace NaNs with column means
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data_cleaned = imputer.fit_transform(raw_data)
```

### 3. Regression Metrics (`03_Regression_Metrics.ipynb`)
Used to measure the error between predicted continuous values ($\hat{y}$) and true target values ($y$).
*   **Mean Absolute Error (MAE)**: Measures average magnitude of errors without considering direction.
    $$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$
*   **Mean Squared Error (MSE)**: Measures average squared difference. Large errors are penalized heavily due to squaring.
    $$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
*   **Median Absolute Error (MedAE)**: Particularly robust to outliers because it uses the median.
    $$\text{MedAE} = \text{median}(|y_1 - \hat{y}_1|, \dots, |y_n - \hat{y}_n|)$$
*   **Multioutput configurations**:
    *   `multioutput='raw_values'`: Returns error for each target separately in multi-target regression.
    *   `multioutput='uniform_average'`: Returns the average error across all outputs.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
medae = median_absolute_error(y_true, y_pred)
```

### 4. Classification Metrics (`04_Classification_metrics.ipynb`)
Evaluate models predicting discrete class labels.
*   **Confusion Matrix**: A table layout displaying the counts of True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).
*   **Accuracy**: Ratio of correct predictions to total predictions.
    $$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$
*   **Precision**: Out of all predicted positives, how many are actually positive.
    $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
*   **Recall (Sensitivity)**: Out of all actual positives, how many were correctly predicted positive.
    $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
*   **F1-Score**: Harmonic mean of precision and recall. Useful for imbalanced datasets.
    $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
*   **Averaging Multi-Class Predictions**:
    *   `average=None`: Returns metrics per class.
    *   `average='macro'`: Computes metrics per class and finds their unweighted mean.
    *   `average='micro'`: Computes global metrics by counting total true positives, false negatives, and false positives.

```python
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

cm = confusion_matrix(y_true, y_pred, labels=['class_A', 'class_B'])
acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='macro')
f1 = f1_score(y_true, y_pred, average='macro')
```

### 5. Feature Selection (`05_Feature_Selection.ipynb`)
Reduces the dimensionality of input features to improve model training speeds and generalization.
*   **Univariate Feature Selection**: Evaluates each feature individually using statistical tests:
    *   `SelectPercentile`: Retains a specified percentage of highest-scoring features.
    *   `SelectKBest`: Retains the $k$ highest-scoring features.
    *   `GenericUnivariateSelect`: A flexible selector with configurable modes.
    *   *Scoring Functions*: `chi2` (for categorical inputs/targets) and `f_classif` (ANOVA F-value for numerical features).
*   **Model-Based Selection**:
    *   `SelectFromModel`: Uses feature importances or coefficients derived from a fitted estimator (e.g. `LinearRegression` weights or `RandomForestClassifier` feature importances) to discard uninformative features.

```python
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Select top 5 features using Chi-Squared test
selector_k = SelectKBest(score_func=chi2, k=5)
X_new_k = selector_k.fit_transform(X, y)

# Select features using RandomForest importance weights
selector_model = SelectFromModel(estimator=RandomForestClassifier(n_estimators=20))
X_new_model = selector_model.fit_transform(X, y)
```

### 6. Feature Scaling & Normalization (`06_Data_Scaling.ipynb`)
Ensures that all numerical features contribute equally to model convergence, preventing features with larger magnitudes from dominating.
*   **StandardScaler**: Centers data around 0 with unit variance.
    $$z = \frac{x - \mu}{\sigma}$$
*   **MinMaxScaler**: Shrinks data within a bounded range (default is `[0, 1]`). Sensitive to outliers.
    $$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
*   **RobustScaler**: Scaled using median and Interquartile Range (IQR), making it robust to extreme outliers.
    $$x_{scaled} = \frac{x - \text{median}}{\text{IQR}}$$
*   **Normalizer**: Scales individual samples (rows) to have a unit norm ($L_1$ or $L_2$ norm), commonly used in text classification.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardizing features
scaler_std = StandardScaler()
X_train_scaled = scaler_std.fit_transform(X_train)
X_test_scaled = scaler_std.transform(X_test)
```

### 7. Data Splitting (`07_Data_Spliting.ipynb`)
Data splitting partitions the dataset into training and test datasets.
*   **`train_test_split` Parameters**:
    *   `test_size`: Proportion of dataset to include in the test split (e.g., 0.2 represents 20%).
    *   `random_state`: Controls shuffling to ensure reproducibility.
    *   `shuffle`: Shuffles the data before splitting (highly recommended).
*   **Data Leakage Prevention**:
    *   Always split the data *before* performing imputation or scaling.
    *   Fit transformers (`fit_transform`) **ONLY** on the training set, then transform the test set using `.transform()`. This prevents testing information from leaking into the training pipeline.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
```
