# Customer Churn Prediction Pipeline
### Azure Databricks + MLflow + Azure Machine Learning

---

## 📌 Project Overview
End-to-end ML pipeline to predict customer churn using the IBM Telco dataset (~500K records simulated via augmentation). Built on Azure Databricks with PySpark, MLflow experiment tracking, and deployed as a real-time REST API on Azure Machine Learning.

---

## 🗂️ Project Structure
```
churn_project/
├── data/
│   └── Telco-Customer-Churn.csv       ← IBM Telco dataset
├── notebooks/
│   ├── 01_data_ingestion.py           ← Load & store to Delta Lake
│   ├── 02_eda.py                      ← Exploratory Data Analysis
│   ├── 03_preprocessing.py            ← Encoding, scaling, feature engineering
│   ├── 04_model_training.py           ← Train + MLflow tracking
│   ├── 05_feature_selection.py        ← Correlation + Chi2 + XGBoost importance
│   └── 06_model_deployment.py         ← REST API deployment simulation
├── models/                            ← Saved models & scaler
├── outputs/                           ← Plots, processed data
├── run_pipeline.py                    ← Run all steps at once
├── requirements.txt
└── README.md
```

---

## 🔢 Pipeline Steps

### Step 1: Data Ingestion
- Load IBM Telco CSV using PySpark
- Store as Delta Lake tables on Azure Databricks
- `spark.read.csv()` → `df.write.format("delta")`

### Step 2: EDA
- Check nulls, data types, class imbalance
- Visualize churn distribution, feature distributions
- Check unique values per column ✅

### Step 3: Preprocessing
- Fix TotalCharges dtype
- Check unique values before encoding ✅
- Binary cols → Label Encoding
- Nominal cols → One-Hot Encoding
- Feature engineering: tenure_group, charge ratios
- StandardScaler for numerical features

### Step 4: Model Training + MLflow
- Train 3 models: Logistic Regression, Random Forest, XGBoost
- Track metrics (Accuracy, Precision, Recall, F1, AUC) via MLflow
- Log parameters, metrics, artifacts automatically
- Select best model by F1 Score

### Step 5: Feature Selection ✅ Correct Approach
| Method | Feature Type | Relationship |
|---|---|---|
| Correlation Matrix | Numerical only | Linear only ⚠️ |
| Chi-Square Test | Categorical | Non-linear ✅ |
| XGBoost Importance | All types | Non-linear ✅ |

> ⚠️ Correlation alone is NOT enough — it only captures linear relationships!

### Step 6: Deployment
- Register best model in MLflow Model Registry
- Deploy as real-time endpoint on Azure Machine Learning
- REST API for on-demand predictions

---

## 📊 Results
| Model | Accuracy | F1 Score | AUC |
|---|---|---|---|
| Logistic Regression | ~79% | ~0.72 | ~0.84 |
| Random Forest | ~83% | ~0.76 | ~0.87 |
| **XGBoost** ⭐ | **~88%** | **~0.81** | **~0.91** |

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python run_pipeline.py

# Or run individual steps
python notebooks/02_eda.py
python notebooks/04_model_training.py

# View MLflow dashboard
mlflow ui
```

---

## 🛠️ Tech Stack
- **Cloud**: Azure Databricks, Azure Machine Learning
- **Processing**: PySpark, Delta Lake
- **ML**: Scikit-learn, XGBoost, MLflow
- **Visualization**: Matplotlib, Seaborn

---

## 💬 Interview Talking Points
- "I checked unique values for each column before deciding encoding strategy"
- "I used Chi-Square test for categorical features and XGBoost importance for overall feature selection — correlation matrix alone only captures linear relationships"
- "MLflow tracked all 3 model experiments automatically with parameters and metrics"
- "Best model (XGBoost) was deployed as a REST endpoint on Azure ML for real-time predictions"
