# Support Vector Machine (SVM) Projects

## 📚 Folder Overview
This directory contains projects focused on **Support Vector Classifiers** and Anomaly Detection.

## 🧠 Core Concept
SVC finds a hyperplane (a line or sheet) that distinctly separates classes with the maximum possible margin. For complex non-linear data, it uses the **Kernel Trick** (like `rbf`) to map data into 3D space to find a cut.
- **Advantages:** Highly effective in high-dimensional spaces.
- **Disadvantages:** Very slow on huge datasets; highly sensitive to unscaled data.

## 📂 Project Files
*   **`SVM.py` (.ipynb)**
    *   **Dataset:** `classData.csv`
    *   **Description:** Demonstrates a standard SVM implementation for classification.
*   **`OCSVM.py` (.ipynb)**
    *   **Dataset:** `ai4i2020.csv`
    *   **Description:** Uses **One-Class SVM**. This is an unsupervised technique specifically designed to learn what "Normal" data looks like, so it can flag anything else as an Anomaly.

## 💻 Code Highlight
```python
from sklearn.svm import SVC, OneClassSVM

# Standard SVC for Classification
model = SVC(kernel='rbf', C=1.0)

# One-Class SVM for Anomaly Detection (Unsupervised)
anomaly_detector = OneClassSVM(nu=0.05, kernel='rbf')
anomaly_detector.fit(X_normal_data)
```
