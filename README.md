# 📘 Comprehensive Machine Learning & Neural Networks Curriculum Guide

Welcome to the central reference guide for the Machine Learning and Artificial Neural Networks workspace. This repository is structured as a professional, textbook-style learning path spanning data preprocessing, supervised models, unsupervised learning, and advanced deep learning architectures.

---

## 📂 Repository Structure

The workspace is organized into three primary modules: **Machine Learning (ML)**, **Scikit-Learn Curriculum (Sklearn_Library)**, and **Artificial Neural Networks (ANN)**.

```
Machine_Learning/
├── ML/
│   ├── ML_Algorithms_Deep_dive/    # Lecture study guides (ML concepts, Math)
│   ├── supervised_labs/             # Hands-on labs organized by model type
│   │   ├── 01_Data_Exploration_and_Preprocessing/
│   │   ├── 02_Linear_Regression/
│   │   ├── 03_Logistic_Regression/
│   │   ├── 04_K_Nearest_Neighbors/
│   │   ├── 05_Support_Vector_Machines/
│   │   ├── 06_Decision_Trees/
│   │   ├── 07_Ensemble_Methods/
│   │   ├── 08_Naive_Bayes/
│   │   └── Competitions/            # Customer Churn & Health Status challenges
│   └── unsupervised_labs/           # Clustering & PCA labs
│       ├── Clustering/
│       └── Dimensionality_Reduction/
Sklearn_Library/                     # Scikit-Learn Preprocessing & ML Curriculum
├── 01_Data_Preparation/             # Datasets, Imputation, Selection, Scaling, Splitting
└── 02_ML_Algorithms/                # Linear Regression & Logistic Regression models
└── ANN/
    ├── Neural_network_Deep_dive/    # Neural network theory & backpropagation notes
    └── neural_network_labs/         # Neural network implementations
        ├── Basic_ANNs/              # MLP classification networks
        └── Autoencoders/            # Image compression, denoising, & anomaly detection
```

---

## 📑 Table of Contents
1. [Data Preprocessing Techniques](#1-data-preprocessing-techniques)
2. [Evaluation Metrics](#2-evaluation-metrics)
3. [Supervised Learning Models](#3-supervised-learning-models)
   - [Linear & Regularized Regression](#31-linear--regularized-regression)
   - [Logistic Regression](#32-logistic-regression)
   - [Decision Trees & Ensemble Methods](#33-decision-trees--ensemble-methods)
   - [K-Nearest Neighbors (KNN)](#34-k-nearest-neighbors-knn)
   - [Support Vector Machines (SVM)](#35-support-vector-machines-svm)
   - [Naive Bayes](#36-naive-bayes)
4. [Unsupervised Learning Paradigms](#4-unsupervised-learning-paradigms)
   - [Clustering (K-Means, GMM, FCM)](#41-clustering)
   - [Dimensionality Reduction (PCA)](#42-dimensionality-reduction)
5. [Artificial Neural Networks (ANN)](#5-artificial-neural-networks-ann)
   - [Feedforward MLPs for Classification](#51-feedforward-mlps-for-classification)
   - [Autoencoder Architectures](#52-autoencoder-architectures)
6. [Scikit-Learn Preprocessing & Algorithms Library](#6-scikit-learn-preprocessing--algorithms-library)

---

## ⚙️ 1. Data Preprocessing Techniques

Proper data preparation is critical for model convergence and performance.

### 1.1 Feature Scaling
Algorithms relying on distance (KNN, SVM) or gradient descent require scaled data.
*   **StandardScaler:** Rescales to mean 0, standard deviation 1.
*   **MinMaxScaler:** Rescales strictly between 0 and 1.
*   **Python Implementation:**
    ```python
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    scaler = StandardScaler()
    # CRITICAL: fit_transform on train, but ONLY transform on test!
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```
    *   📁 **Implementation Lab:** See [`ML/supervised_labs/02_Linear_Regression/Data_Preprocessing.ipynb`](./ML/supervised_labs/02_Linear_Regression/Data_Preprocessing.ipynb).
    *   📁 **Sklearn Prep Lab:** See [`Sklearn_Library/01_Data_Preparation/06_Data_Scaling.ipynb`](./Sklearn_Library/01_Data_Preparation/06_Data_Scaling.ipynb).

### 1.2 Categorical Encoding
*   **Label Encoding:** Converts categories to integers (Red=0, Blue=1). Best for the target variable `y`.
*   **One-Hot Encoding:** Creates binary columns for each category. Best for features `X`.
*   **Python Implementation:**
    ```python
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd

    # Label Encoding (for y)
    le = LabelEncoder()
    y = le.fit_transform(y)

    # One-Hot Encoding (for X)
    X = pd.get_dummies(X, columns=['CategoryColumn'], drop_first=True)
    ```

### 1.3 Handling Missing Data & Imputation
*   **SimpleImputer:** Replaces missing values using statistics like mean, median, mode, or a constant.
    *   📁 **Sklearn Prep Lab:** See [`Sklearn_Library/01_Data_Preparation/02_Data_Cleaning.ipynb`](./Sklearn_Library/01_Data_Preparation/02_Data_Cleaning.ipynb).

### 1.4 Feature Selection
*   **Statistical/Model Selectors:** Filters out noise columns using univariate tests (`SelectKBest`, `chi2`) or model importances (`SelectFromModel`).
    *   📁 **Sklearn Prep Lab:** See [`Sklearn_Library/01_Data_Preparation/05_Feature_Selection.ipynb`](./Sklearn_Library/01_Data_Preparation/05_Feature_Selection.ipynb).

### 1.5 Handling Imbalanced Data (SMOTE)
*   **SMOTE:** Synthetically generates minority class examples to balance the dataset.
*   **Python Implementation:**
    ```python
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    # CRITICAL: Only apply to training data!
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    ```

---

## 📊 2. Evaluation Metrics

### 2.1 Classification Metrics
*   **Accuracy:** Overall correct predictions.
*   **Precision:** Out of predicted positives, how many are actual positives.
*   **Recall:** Out of actual positives, how many did we find.
*   **F1-Score:** Harmonic mean of Precision and Recall.
    *   📁 **Sklearn Prep Lab:** See [`Sklearn_Library/01_Data_Preparation/04_Classification_metrics.ipynb`](./Sklearn_Library/01_Data_Preparation/04_Classification_metrics.ipynb).

### 2.2 Regression Metrics
*   **MAE:** Average absolute difference between prediction and actual.
*   **MSE:** Average squared difference. Heavily penalizes outliers.
*   **R-Squared ($R^2$):** Proportion of variance in target explained by the model.
    *   📁 **Sklearn Prep Lab:** See [`Sklearn_Library/01_Data_Preparation/03_Regression_Metrics.ipynb`](./Sklearn_Library/01_Data_Preparation/03_Regression_Metrics.ipynb).

---

## 📈 3. Supervised Learning Models

### 3.1 Linear & Regularized Regression
*   **Linear Regression:** Fits a linear equation by minimizing the Residual Sum of Squares (RSS).
*   **Ridge Regression (L2):** Adds a squared penalty ($\lambda \sum \beta_i^2$) to prevent overfitting by shrinking coefficients.
*   **Lasso Regression (L1):** Adds an absolute penalty ($\lambda \sum |\beta_i|$), acting as feature selection by driving weights exactly to zero.
*   **ElasticNet:** Combines both L1 and L2 penalties.
*   📁 **Implementation Lab:** See [`ML/supervised_labs/02_Linear_Regression/task.ipynb`](./ML/supervised_labs/02_Linear_Regression/task.ipynb) & [`ML/supervised_labs/02_Linear_Regression/Polynomial_Regression_Auto_MPG.ipynb`](./ML/supervised_labs/02_Linear_Regression/Polynomial_Regression_Auto_MPG.ipynb).
*   📁 **Sklearn Curriculum Lab:** See [`Sklearn_Library/02_ML_Algorithms/01_Linear_Regression/01_Linear_Regression_EX01.ipynb`](./Sklearn_Library/02_ML_Algorithms/01_Linear_Regression/01_Linear_Regression_EX01.ipynb) & [`Sklearn_Library/02_ML_Algorithms/01_Linear_Regression/01_Linear_Regression_PolynomialFeatures_EX02.ipynb`](./Sklearn_Library/02_ML_Algorithms/01_Linear_Regression/01_Linear_Regression_PolynomialFeatures_EX02.ipynb).

### 3.2 Logistic Regression
*   **Sigmoid Mapping:** Models categorical binary outcomes using a sigmoid function: $P = \frac{1}{1 + e^{-Z}}$
*   📁 **Implementation Lab:** See [`ML/supervised_labs/03_Logistic_Regression/logistic_regression.ipynb`](./ML/supervised_labs/03_Logistic_Regression/logistic_regression.ipynb) & [`ML/supervised_labs/03_Logistic_Regression/MultiClass_LogisticRegression_EngineFault.ipynb`](./ML/supervised_labs/03_Logistic_Regression/MultiClass_LogisticRegression_EngineFault.ipynb).

### 3.3 Decision Trees & Ensemble Methods
*   **Decision Tree:** Recursively splits data based on Information Gain/Gini Impurity.
*   **Random Forest:** Ensemble of decision trees constructed on bagged datasets and random features.
*   **Gradient Boosting (XGBoost / LightGBM):** Sequentially constructs trees, where each new tree corrects the residual errors of the previous ones.
*   📁 **Implementation Lab:** See [`ML/supervised_labs/06_Decision_Trees/Decision_tree_titanic.ipynb`](./ML/supervised_labs/06_Decision_Trees/Decision_tree_titanic.ipynb) & [`ML/supervised_labs/07_Ensemble_Methods/RandomForest_LGBM_XGB_IsolationForest.ipynb`](./ML/supervised_labs/07_Ensemble_Methods/RandomForest_LGBM_XGB_IsolationForest.ipynb).

### 3.4 K-Nearest Neighbors (KNN)
*   Classifies samples based on the majority vote of their nearest neighbors using distance metrics.
*   📁 **Implementation Lab:** See [`ML/supervised_labs/04_K_Nearest_Neighbors/K-NN.ipynb`](./ML/supervised_labs/04_K_Nearest_Neighbors/K-NN.ipynb).

### 3.5 Support Vector Machines (SVM)
*   Finds the optimal separating hyperplane that maximizes the margin. Employs the Kernel Trick (e.g., RBF) for non-linear decision bounds.
*   📁 **Implementation Lab:** See [`ML/supervised_labs/05_Support_Vector_Machines/SVM.ipynb`](./ML/supervised_labs/05_Support_Vector_Machines/SVM.ipynb).

### 3.6 Naive Bayes
*   Probabilistic classifiers based on Bayes' Theorem, assuming independence among features.
*   📁 **Implementation Lab:** See [`ML/supervised_labs/08_Naive_Bayes/NB_Spam_Classifier.ipynb`](./ML/supervised_labs/08_Naive_Bayes/NB_Spam_Classifier.ipynb) & [`ML/supervised_labs/Competitions/Customer_Churn_Prediction/NB_Customer_Churn_Prediction.ipynb`](./ML/supervised_labs/Competitions/Customer_Churn_Prediction/NB_Customer_Churn_Prediction.ipynb).

---

## 🔮 4. Unsupervised Learning Paradigms

### 4.1 Clustering
*   **K-Means:** Partitions data into $K$ centroids using distance minimizations.
*   **Gaussian Mixture Models (GMM):** Probabilistic clustering representing sub-populations as gaussian distributions.
*   **Fuzzy C-Means (FCM):** Allows points to have partial degrees of membership in multiple clusters.
*   📁 **Implementation Lab:** See [`ML/unsupervised_labs/Clustering/KMeans_GMM_FuzzyClustering.ipynb`](./ML/unsupervised_labs/Clustering/KMeans_GMM_FuzzyClustering.ipynb).

### 4.2 Dimensionality Reduction
*   **PCA (Principal Component Analysis):** Orthogonally projects data into eigenvectors corresponding to the largest eigenvalues to retain maximum variance.
*   📁 **Implementation Lab:** See [`ML/unsupervised_labs/Dimensionality_Reduction/PCA_LogisticRegression.ipynb`](./ML/unsupervised_labs/Dimensionality_Reduction/PCA_LogisticRegression.ipynb).

---

## 🧠 5. Artificial Neural Networks (ANN)

Deep Learning models built using Keras and TensorFlow.

### 5.1 Feedforward MLPs for Classification
*   Implements dense layers with activation functions (ReLU, Sigmoid, Softmax) to handle complex feature distributions.
*   📁 **Implementation Lab:** See [`ANN/neural_network_labs/Basic_ANNs/ANN_Sensor_Readings_4.ipynb`](./ANN/neural_network_labs/Basic_ANNs/ANN_Sensor_Readings_4.ipynb) & [`ANN/neural_network_labs/Basic_ANNs/ANN_Breast_Cancer_Keras.ipynb`](./ANN/neural_network_labs/Basic_ANNs/ANN_Breast_Cancer_Keras.ipynb).

### 5.2 Autoencoder Architectures
*   **Standard Autoencoder:** Reconstructs inputs from a compressed bottleneck dimension for unsupervised feature learning.
*   **Denoising Autoencoder:** Reconstructs clean data from artificially corrupted noisy input features.
*   **Anomaly Detection:** Flags outliers by identifying transactions/records that exhibit high reconstruction errors when passed through an autoencoder trained only on normal samples.
*   📁 **Implementation Lab:** See [`ANN/neural_network_labs/Autoencoders/Autoencoder_MNIST.ipynb`](./ANN/neural_network_labs/Autoencoders/Autoencoder_MNIST.ipynb) & [`ANN/neural_network_labs/Autoencoders/Autoencoder_Anomaly_Detection_CreditCard.ipynb`](./ANN/neural_network_labs/Autoencoders/Autoencoder_Anomaly_Detection_CreditCard.ipynb).

---

## 🔬 6. Scikit-Learn Preprocessing & Algorithms Library

A dedicated curriculum directory detailing standalone pipeline components using Scikit-Learn.

*   📁 **Library Root:** See [`Sklearn_Library/`](./Sklearn_Library)
*   📁 **01_Data_Preparation:** Detailed preprocessing steps (built-in datasets, imputation, feature selection, scaling, splitting) and evaluation metrics. Link to [`01_Data_Preparation/README.md`](./Sklearn_Library/01_Data_Preparation/README.md).
*   📁 **02_ML_Algorithms/01_Linear_Regression:** Linear and Polynomial Regression modeling examples. Link to [`02_ML_Algorithms/01_Linear_Regression/README.md`](./Sklearn_Library/02_ML_Algorithms/01_Linear_Regression/README.md).
*   📁 **02_ML_Algorithms/02_Logistic_Regression:** Concept guide explaining classification principles. Link to [`02_ML_Algorithms/02_Logistic_Regression/README.md`](./Sklearn_Library/02_ML_Algorithms/02_Logistic_Regression/README.md).

