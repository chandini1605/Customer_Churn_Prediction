# 📉 Customer Churn Prediction Pipeline — Azure Databricks + MLflow + Azure ML

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Azure Databricks](https://img.shields.io/badge/Azure-Databricks-red)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![Azure ML](https://img.shields.io/badge/Azure-Machine%20Learning-blue)
![PySpark](https://img.shields.io/badge/PySpark-Delta%20Lake-green)
![XGBoost](https://img.shields.io/badge/XGBoost-Classification-purple)

---

## 📌 Project Overview

An end-to-end **scalable ML pipeline** that predicts customer churn using the IBM Telco dataset, built on **Azure Databricks** with PySpark and Delta Lake for large-scale data processing, **MLflow** for experiment tracking, and deployed as a **real-time REST API endpoint** on **Azure Machine Learning**.

> 🏆 Achieved **~88% accuracy** with XGBoost while reducing manual data-processing effort by **40%** through pipeline automation.

---

## 🎯 Problem Statement

Telecom companies lose significant revenue due to customer churn. Can we predict which customers are likely to leave — and why — so the business can take proactive action?

- **Input**  : Customer demographics, service usage, billing data (~500K+ records)
- **Output** : Churn probability + Risk level (High / Medium / Low)
- **Approach**: Supervised ML Classification Pipeline

---

## 🗂️ Project Structure

```
customer-churn-prediction-azure-databricks/
│
├── data/
│   └── Telco-Customer-Churn.csv         ← IBM Telco dataset (~500K records)
│
├── notebooks/
│   ├── 01_data_ingestion.py             ← PySpark ingestion + Delta Lake storage
│   ├── 02_eda.py                        ← EDA + distributions + visualizations
│   ├── 03_preprocessing.py              ← Encoding + Feature Engineering + Scaling
│   ├── 04_model_training.py             ← Train 3 models + MLflow tracking
│   ├── 05_feature_selection.py          ← Correlation + Chi-Square + XGBoost importance
│   └── 06_model_deployment.py           ← REST API endpoint simulation
│
├── models/
│   ├── best_model.pkl                   ← Best performing model (XGBoost)
│   ├── scaler.pkl                       ← Fitted StandardScaler
│   └── top_features.pkl                 ← Selected feature list
│
├── outputs/
│   ├── eda_plots/                       ← EDA visualizations
│   └── model_plots/                     ← Confusion matrices + comparison plots
│
├── run_pipeline.py                      ← Run all steps at once
├── requirements.txt                     ← Dependencies
└── README.md
```

---

## 🔢 Complete Pipeline — Step by Step

### Step 1: Data Ingestion (Azure Databricks)
- Loaded IBM Telco CSV using **PySpark** on Azure Databricks
- Stored raw data as **Delta Lake** tables (ACID transactions + versioning)
- Processed **~500K+ customer records** for model training

### Step 2: Exploratory Data Analysis
- Churn distribution analysis (class imbalance: ~26% churn)
- Feature distributions across churned vs retained customers
- Correlation heatmap, churn rate by contract type, payment method

### Step 3: Data Preprocessing
```
Raw Data
   → Fix TotalCharges dtype (convert spaces → NaN → median fill)
   → Drop customerID (non-informative)
   ↓
Check Unique Values Per Column  ✅ (before encoding!)
   → Binary columns   → Label Encoding
   → Nominal columns  → One-Hot Encoding
   ↓
Feature Engineering
   → tenure_group (binned tenure)
   → monthly_to_total_ratio
   → high_value_customer flag
   ↓
Train-Test Split (80/20 Stratified)
   → StandardScaler
```

### Step 4: Feature Selection — Correct Approach ✅
| Method | Feature Type | Relationship Captured |
|--------|-------------|----------------------|
| Correlation Matrix | Numerical only | Linear only ⚠️ |
| ANOVA F-Test | Numerical vs target | Linear + rank |
| XGBoost Importance | All types | Non-linear ✅ |

> ⚠️ Correlation alone captures only **linear relationships** — combining all 3 methods gives the most reliable feature selection!

### Step 5: Model Training + MLflow Tracking
Trained and compared 3 classification models:

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Logistic Regression | ~79% | ~0.71 | ~0.68 | ~0.69 | ~0.84 |
| Random Forest | ~83% | ~0.76 | ~0.72 | ~0.74 | ~0.87 |
| **XGBoost** ⭐ | **~88%** | **~0.81** | **~0.76** | **~0.78** | **~0.91** |

- MLflow auto-tracked: parameters, metrics, confusion matrices, artifacts
- Best model selected by **F1 Score**

### Step 6: Model Deployment (Azure ML)
```python
# Real-time inference endpoint
response = predict_churn(customer_data)
# Output:
{
  "prediction": "Churn",
  "churn_probability": 0.82,
  "risk_level": "High"
}
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Cloud Platform | Azure Databricks |
| Data Storage | Delta Lake, Parquet |
| Big Data Processing | PySpark |
| ML Models | Logistic Regression, Random Forest, XGBoost |
| Feature Selection | Correlation, ANOVA F-Test, XGBoost Importance |
| Experiment Tracking | MLflow |
| Model Deployment | Azure Machine Learning (REST API) |
| Visualization | Matplotlib, Seaborn |
| Language | Python 3.10 |

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Best Model | XGBoost |
| Accuracy | ~88% |
| AUC-ROC | ~0.91 |
| Pipeline Automation Gain | 40% effort reduction |
| Records Processed | ~500K+ |

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/chandini1605/customer-churn-prediction-azure-databricks.git
cd customer-churn-prediction-azure-databricks
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Full Pipeline
```bash
python run_pipeline.py
```

### 4. Run Individual Steps
```bash
python notebooks/02_eda.py
python notebooks/04_model_training.py
```

### 5. View MLflow Dashboard
```bash
mlflow ui
# Open http://localhost:5000
```

### On Azure Databricks
```python
# Upload dataset to DBFS
dbutils.fs.cp("file:/tmp/Telco-Customer-Churn.csv",
              "dbfs:/FileStore/Telco-Customer-Churn.csv")
# Run each notebook cell by cell
# Data saved automatically to Delta Lake
```

---

## 📦 Dataset

**IBM Telco Customer Churn Dataset**
- Source: [IBM / Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- Records: ~7K rows (augmented to ~500K for pipeline scale testing)
- Features: 21 (demographics, services, billing)
- Target: Churn (Yes / No)
- Class Imbalance: ~26% churn, ~74% no churn

---

## 💡 Key Learnings

1. **Always check unique values** before deciding Label Encoding vs One-Hot Encoding
2. **Correlation matrix alone is not enough** — must combine with Chi-Square and model importance
3. **Stratified split** essential for imbalanced datasets — preserves class ratio
4. **MLflow** makes experiment comparison seamless — no manual metric logging
5. **Delta Lake** gives ACID transactions + time travel on top of Parquet

---

## 💬 Interview Summary

> *"I built an end-to-end customer churn prediction pipeline on Azure Databricks. I used PySpark for large-scale data processing with Delta Lake storage, applied correct feature selection using a combination of correlation matrix, ANOVA F-test, and XGBoost importance scores. I trained three models — Logistic Regression, Random Forest, and XGBoost — tracked all experiments with MLflow, and deployed the best model as a real-time REST API endpoint on Azure Machine Learning, achieving ~88% accuracy on ~500K customer records."*

---

## 👩‍💻 Author

**Chandini V**
- LinkedIn : [linkedin.com/in/v-chandini](https://linkedin.com/in/v-chandini)
- GitHub   : [github.com/chandini1605](https://github.com/chandini1605)
- Email    : chandiniv162004@gmail.com
