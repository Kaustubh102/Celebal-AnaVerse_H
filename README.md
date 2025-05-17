
# 🛰️ Sensor Anomaly Detection - Celebal Challenge

This repository presents a complete pipeline to identify anomalies in sensor readings. Leveraging **XGBoost**, this project follows a robust flow—from data preprocessing and feature engineering to model tuning and prediction generation.

---

## 🎯 Objective

Develop a high-performing anomaly detection model focused on **F1-score optimization**, ensuring generalizability and robustness on unseen data through structured stages:

- Data preprocessing and exploratory analysis
- Feature creation and enhancement
- Model development with **XGBoost**
- Hyperparameter tuning using **Optuna**
- Handling class imbalance with **SMOTE**
- Generating submission-ready predictions

---

## 📂 Dataset Description

- **train.parquet** → Labeled training data with sensor features and anomaly flags.
- **test.parquet** → Unlabeled sensor data to predict anomalies.

Dataset includes sensor readings (X1, X2, ..., X5) and time-engineered features (year, month, weekend).

---

## ⚙️ Tech Stack & Libraries

- Python 3.11
- pandas, numpy
- xgboost
- optuna (hyperparameter optimization)
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib, seaborn (visualization)

---

## ✨ Feature Engineering Highlights

Derived from the `Date` column:

```python
df['year'] = pd.to_datetime(df['Date']).dt.year
df['month'] = pd.to_datetime(df['Date']).dt.month
df['day'] = pd.to_datetime(df['Date']).dt.day
df['dayofweek'] = pd.to_datetime(df['Date']).dt.dayofweek
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
```

---

## 🚀 Model Pipeline

### Preprocessing
- Extracted features from datetime columns
- Dropped unnecessary columns (`Date`)
- Applied **RobustScaler**

### Resampling
- Addressed class imbalance using **SMOTE**

### Model Training
- Trained an **XGBClassifier**
- Tuned hyperparameters via **Optuna**
- Evaluated using **F1-Score** with 3-fold cross-validation

---

## 📊 Results

- ✅ Final F1-Score: **0.78**
- ✅ Predictions generated on **test.parquet**
- ✅ Submission-ready predictions saved to CSV

---

## 🔎 Insights from Analysis

Through the anomaly exploration phase, an interesting pattern emerged:
> **Anomalies were predominantly observed at the beginning and the end of the year**, hinting at potential operational or environmental factors affecting sensor behavior during these periods.

---

## 🏅 Achievement

Proudly secured **99th Rank Globally** 🏆 in the **Celebal Sensor Anomaly Detection Challenge**, competing with data science professionals worldwide.

---

## 📈 Visualization

Graphs showcasing anomaly distribution over the year and feature importances are included in the `visualization` notebook.

---

## 💼 Author

**Kaustubh Yewale**

---
