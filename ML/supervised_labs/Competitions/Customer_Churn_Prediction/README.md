# 🏆 Customer Churn Prediction — Kaggle Competition

## 📌 Competition Overview
Predict whether a bank customer will **churn (leave)** based on their demographic and account information.

| Field | Details |
|---|---|
| **Platform** | Kaggle |
| **Task** | Binary Classification |
| **Target** | `Exited` — `1 = Churned`, `0 = Stayed` |
| **Evaluation Metric** | ROC-AUC |

## 🤖 Model Used — Naive Bayes (GaussianNB)

The core classifier in this notebook is **Gaussian Naive Bayes** from scikit-learn.

```python
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train_processed, y_train)

# Generate submission probabilities
submission_proba = gnb.predict_proba(X_test_processed)[:, 1]
```

**Why GaussianNB for this problem?**
- Customer features (age, balance, credit score, tenure) are continuous → Gaussian assumption applies naturally.
- Extremely fast to train, making it a great **baseline** before exploring heavier models.
- Probabilistic output (`.predict_proba`) plugs directly into ROC-AUC evaluation.

## 🔧 Pipeline Summary

```
Raw CSV (train.csv / test.csv)
       ↓
Drop irrelevant columns (id, CustomerId, Surname)
       ↓
Encode categoricals (Geography, Gender)
       ↓
RandomizedSearchCV (hyperparameter tuning)
       ↓
GaussianNB (fit on training set)
       ↓
submission.csv  ← predicted churn probabilities for test set
```

## 📂 Files

| File | Description |
|---|---|
| `NB_Customer_Churn_Competition.ipynb` | Full notebook: EDA, preprocessing pipeline, model training, and `submission.csv` generation |

## 📊 Features Used

| Feature | Type | Notes |
|---|---|---|
| `CreditScore` | Numeric | |
| `Geography` | Categorical | One-hot encoded |
| `Gender` | Categorical | Label encoded |
| `Age` | Numeric | Key churn predictor |
| `Tenure` | Numeric | |
| `Balance` | Numeric | |
| `NumOfProducts` | Numeric | |
| `HasCrCard` | Binary | |
| `IsActiveMember` | Binary | |
| `EstimatedSalary` | Numeric | |

## 💡 Key Takeaway
Naive Bayes serves as a **strong, interpretable baseline** for churn prediction. Its speed
and probabilistic outputs make it ideal for rapid iteration before moving to ensemble methods
(XGBoost, Random Forest) in later stages of the competition.
