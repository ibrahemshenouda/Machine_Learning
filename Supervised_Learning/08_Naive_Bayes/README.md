# Naive Bayes Classifier

## 📚 Folder Overview
This directory covers **Naive Bayes** — a family of fast, probabilistic classifiers built on
**Bayes' Theorem** with a strong conditional-independence assumption between features.

## 🧠 Core Concept
Despite being "naïve," these models are surprisingly powerful, particularly for text and
high-dimensional data. The core prediction rule is:

```
P(class | features) ∝ P(class) × ∏ P(feature_i | class)
```

| Variant | Likelihood Assumption | Best For |
|---|---|---|
| **GaussianNB** | Continuous features follow a Gaussian distribution | Numerical / medical data |
| **MultinomialNB** | Count / frequency features | Text (word counts, TF-IDF) |
| **BernoulliNB** | Binary presence/absence features | Text (word presence) |

- **Advantages:** Extremely fast to train, works well with small datasets, naturally handles multi-class problems, and is robust to irrelevant features.
- **Disadvantages:** The independence assumption rarely holds in practice; can be outperformed when features are highly correlated.

## 📂 Project Files

| File | Dataset | Task | NB Variant |
|---|---|---|---|
| `NB_Spam_Classifier.ipynb` | `emails.csv` (word-frequency features) | Binary spam detection | **MultinomialNB** |
| `NB_Customer_Churn_Competition.ipynb` | Kaggle competition `train.csv / test.csv` | Binary churn prediction (competition submission) | **GaussianNB + full ML pipeline** |

## 📊 Datasets

### 1 · Spam E-mail Dataset (`NB_Spam_Classifier.ipynb`)
- **Features:** Word-frequency counts extracted from raw emails
- **Target:** `Prediction` — `1 = Spam`, `0 = Not Spam`
- **Model:** `MultinomialNB` — the natural choice for discrete count features

### 2 · Customer Churn — Kaggle Competition (`NB_Customer_Churn_Competition.ipynb`)
- **Features:** Customer demographics & account info (mix of numeric + categorical)
- **Target:** Binary churn flag
- **Pipeline:** Preprocessing → RandomizedSearchCV → Naive Bayes → `submission.csv` generation

## 💻 Code Highlight

```python
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# --- Spam Classifier (MultinomialNB) ---
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# --- Churn Prediction (GaussianNB) ---
gnb = GaussianNB()
gnb.fit(X_train_processed, y_train)
submission_proba = gnb.predict_proba(X_test_processed)[:, 1]
```

## 🔑 Key Takeaways
1. **MultinomialNB** is the go-to choice for text/count data — it is fast and effective.
2. **GaussianNB** handles continuous features without requiring scaling.
3. Naive Bayes is a strong **baseline** in competition pipelines thanks to its speed and interpretability.
4. Despite the naïve assumption, it often performs surprisingly well on real-world data.
