# Decision Tree Projects

## 📚 Folder Overview
This directory contains projects utilizing the highly interpretable **Decision Tree** algorithm.

## 🧠 Core Concept
Decision Trees split data based on "Yes/No" questions at each node, maximizing **Information Gain** or minimizing **Gini Impurity**.
- **Advantages:** Intuitive flowchart structure; no feature scaling required.
- **Disadvantages:** Extremely prone to overfitting (memorizing the training data).

## 📂 Project Files
*   **`Decision_tree_titanic.py` (.ipynb)**
    *   **Dataset:** Titanic Dataset (`datasets_11657_16098_train.csv`)
    *   **Description:** The classic Titanic survival prediction challenge. It predicts whether a passenger survived (1) or not (0) based on age, gender, and ticket class using the branching logic of a Decision Tree.

## 💻 Code Highlight
```python
from sklearn.tree import DecisionTreeClassifier
# max_depth is crucial to prevent the tree from overfitting!
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
```
