# 🛡️ Support Vector Classifier (SVC)

This directory is dedicated to the implementation of Support Vector Classification (SVC) models using Scikit-Learn. It focuses on finding the optimal hyperplane to classify categorical data and utilizing kernel tricks for non-linearly separable datasets.

---

## 📂 Directory Contents

1.  **[02_Support_Vector_Regressor(SVM)_EX02.ipynb](./02_Support_Vector_Regressor(SVM)_EX02.ipynb)**: *Note: This notebook is currently a placeholder/template adopted from the SVR module demonstrating the Kernel Trick (Linear, Polynomial, RBF) on the Earthquakes dataset. It serves as a foundation for implementing SVC.*

---

## 📖 Key Concepts in SVC

### 1. Support Vector Classification Principles
Unlike Support Vector Regression (SVR) which fits a tube around continuous data, Support Vector Classification (SVC) aims to find the optimal hyperplane that separates data points of different classes with the maximum margin.

*   **Maximum Margin**: The distance between the separating hyperplane and the closest data points (support vectors) of any class. SVC maximizes this margin.
*   **Support Vectors**: The critical data points that lie closest to the decision surface. They are the only points that influence the position and orientation of the hyperplane.

```python
from sklearn.svm import SVC

# Instantiate SVC model (defaults to RBF kernel)
classifier = SVC(kernel='rbf', C=1.0)

# Fit model on training set
classifier.fit(X_train, y_train)

# Predict classes
predictions = classifier.predict(X_test)

# Evaluate accuracy
accuracy = classifier.score(X_test, y_test)
```

### 2. The Kernel Trick
Real-world classification datasets are rarely linearly separable in their original feature space. SVC utilizes the **Kernel Trick** to project data into a higher-dimensional space where a linear separating hyperplane can be found.

*   `kernel='linear'`: Best for simple, linearly separable data.
*   `kernel='poly'`: Polynomial kernel for decision boundaries that form curves.
*   `kernel='rbf'` (Radial Basis Function): The default kernel. It maps data into infinite-dimensional spaces, making it highly effective for complex, non-linear classification tasks.

### 3. Hyperparameter Tuning
*   **C (Regularization Parameter)**: Controls the trade-off between achieving a low training error and a low testing error that generalizes well. A large `C` aims for better classification of all training points (smaller margin), while a small `C` encourages a larger margin (potentially allowing some misclassifications).
*   **Gamma**: Defines how far the influence of a single training example reaches for non-linear kernels (`rbf`, `poly`). Low values mean 'far', high values mean 'close'.

---

## 🚀 Next Steps
- Implement a dedicated classification dataset (e.g., Iris or Breast Cancer) using `SVC`.
- Evaluate the classifier using classification metrics such as the Confusion Matrix, Precision, Recall, and F1-Score.
