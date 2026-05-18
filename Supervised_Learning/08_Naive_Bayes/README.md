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

> 🏆 The **Customer Churn Prediction** (Kaggle competition) notebook using **GaussianNB**
> has been moved to [`Competitions/Customer_Churn_Prediction/`](../Competitions/Customer_Churn_Prediction/).

## 📊 Dataset

**Spam E-mail Dataset** (`NB_Spam_Classifier.ipynb`)
- **Features:** Word-frequency counts extracted from raw emails
- **Target:** `Prediction` — `1 = Spam`, `0 = Not Spam`
- **Model:** `MultinomialNB` — the natural choice for discrete count features

## 💻 Code Highlight

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)

print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))
```

## 🔑 Key Takeaways
1. **MultinomialNB** is the go-to choice for text/count data — fast and effective.
2. **GaussianNB** handles continuous features without requiring feature scaling.
3. Naive Bayes is a strong **baseline** — low latency, interpretable, and surprisingly accurate.
4. Despite the naïve assumption, it often performs well on real-world data.
