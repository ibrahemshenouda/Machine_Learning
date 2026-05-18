# K-Nearest Neighbors (KNN) Projects

## 📚 Folder Overview
This directory focuses on the **K-Nearest Neighbors** algorithm, a distance-based learning model.

## 🧠 Core Concept
KNN is a "lazy learner." To classify a new data point, it looks at the 'K' closest data points in the training dataset (usually using **Euclidean distance**) and takes a majority vote.
- **Advantages:** Simple to understand, handles multi-class naturally.
- **Disadvantages:** Very slow at prediction time. **Requires scaled data!**

## 📂 Project Files
*   **`K-NN.py` (.ipynb)**
    *   **Dataset:** `classData.csv`
    *   **Description:** Demonstrates the implementation of KNN. In the code, you will see the critical step of using `StandardScaler` or `MinMaxScaler` before feeding data to the model.

## 💻 Code Highlight
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Crucial Step for KNN!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Initialize KNN with 5 neighbors
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)
```
