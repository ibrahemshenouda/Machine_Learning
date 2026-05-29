# 🛡️ Support Vector Classifier (SVC)

This directory is dedicated to the implementation of Support Vector Classification (SVC) models using Scikit-Learn. It focuses on finding the optimal hyperplane to classify categorical data and utilizing kernel tricks for non-linearly separable datasets.

---

## 📂 Directory Contents

1.  **[02_Support_Vector_Classifier(SVC).ipynb](./02_Support_Vector_Classifier(SVC).ipynb)**: An end-to-end classification pipeline using the standard Breast Cancer dataset. This notebook demonstrates the differences between `linear` and `rbf` kernels, uses PCA (Principal Component Analysis) to visualize decision boundaries in 2D, and evaluates model performance using confusion matrices and classification reports.

---

## 📖 Key Concepts in SVC

### 1. Support Vector Classification Principles
Unlike Support Vector Regression (SVR) which fits a tube around continuous data, Support Vector Classification (SVC) aims to find the optimal hyperplane that separates data points of different classes with the maximum margin.

*   **Maximum Margin**: The distance between the separating hyperplane and the closest data points (support vectors) of any class. SVC maximizes this margin.
*   **Support Vectors**: The critical data points that lie closest to the decision surface. They are the only points that influence the position and orientation of the hyperplane.

```python
from sklearn.svm import SVC

# Instantiate SVC model (defaults to RBF kernel)
classifier = SVC(kernel='rbf', C=0.1, gamma='auto')

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

### 4. Visualizing Decision Boundaries (PCA)
In order to visualize decision boundaries for high-dimensional data (like the Breast Cancer dataset with 30 features), we can use **Principal Component Analysis (PCA)** to reduce the feature space to 2 dimensions. This allows the `DecisionBoundaryDisplay` module to plot how different kernels separate the target classes.

---

## 📊 Model Evaluation Highlights
*   **Dataset**: Breast Cancer Dataset (Binary Classification: Benign vs. Malignant)
*   **Metrics**: 
    *   **Accuracy Score**: Comparing test and train performance.
    *   **Classification Report**: Precision, Recall, and F1-Score breakdown for each class.
    *   **Confusion Matrix**: Visualized using `ConfusionMatrixDisplay` to examine True Positives, False Positives, etc.
*   **Visualizations**: Uses `DecisionBoundaryDisplay` combined with PCA to visually contrast the linear decision boundary against the non-linear boundaries formed by the RBF kernel.
