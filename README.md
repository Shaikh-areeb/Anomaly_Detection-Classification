# 🚀 Anomaly Detection using Isolation Forest

## 📌 Project Overview
This project focuses on detecting anomalies (outliers) in a dataset using **unsupervised machine learning techniques**. The workflow includes **EDA, statistical analysis, visualization, and model building** using Isolation Forest.

---

## 🎯 Objectives
- Perform Exploratory Data Analysis (EDA)
- Detect outliers using statistical methods
- Visualize data distribution
- Apply Isolation Forest for anomaly detection
- Evaluate model performance

---

## 📊 Exploratory Data Analysis (EDA)
- Checked data types and structure  
- Performed descriptive statistics  
- Identified skewness and distribution  
- Checked missing values  

---

## 📈 Statistical Analysis
- Mean, Median, Standard Deviation  
- Percentile analysis  
- IQR method for outlier detection  

---

## 📉 Visualization
- Histogram  
- Boxplot  
- Distribution plots  
- Correlation heatmap  

---

## 🤖 Model - Isolation Forest
Isolation Forest is an unsupervised algorithm used to detect anomalies by isolating outliers.

---

## ⚙️ Hyperparameter Tuning

```python
param_grid = {
    'n_estimators': [50, 100],
    'contamination': [0.01, 0.02, 0.05],
    'max_samples': [128, 256]
}

model = IsolationForest(random_state=42)

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1
)
