# Data Exploration & Fault Detection

## 📚 Folder Overview
This directory contains foundational notebooks focused on dataset exploration and preprocessing before explicit modeling occurs.

## 📂 Project Files
*   **`engine_fault_detection_dataset.py` (.ipynb)**
    *   **Dataset:** `engine_fault_detection_dataset.csv`
    *   **Description:** This project focuses on the heavy lifting of Data Science: Loading, inspecting, cleaning, and preparing the raw engine fault data. 
    *   *Note:* The final predictive model (Logistic Regression) for this dataset was spun out into the `Logistic_Regression` folder. This notebook serves as the pure data-preparation pipeline.

## 💻 Code Highlight
In these files, you will find heavy pandas usage:
```python
import pandas as pd

# Inspecting the raw fault data
df = pd.read_csv('engine_fault_detection_dataset.csv')
print(df.info())
print(df.isnull().sum())

# Preparing for future modeling
X = df.drop('Fault', axis=1)
y = df['Fault']
```
