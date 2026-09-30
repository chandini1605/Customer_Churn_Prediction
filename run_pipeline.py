# ============================================================
# MAIN PIPELINE RUNNER
# Customer Churn Prediction - Run All Steps
# ============================================================

import subprocess
import sys
import os

steps = [
    ("Step 1: Data Ingestion",        "notebooks/01_data_ingestion.py"),
    ("Step 2: EDA",                   "notebooks/02_eda.py"),
    ("Step 3: Preprocessing",         "notebooks/03_preprocessing.py"),
    ("Step 4: Model Training+MLflow", "notebooks/04_model_training.py"),
    ("Step 5: Feature Selection",     "notebooks/05_feature_selection.py"),
    ("Step 6: Deployment Simulation", "notebooks/06_model_deployment.py"),
]

print("=" * 60)
print("  CUSTOMER CHURN PREDICTION PIPELINE")
print("  Azure Databricks + MLflow + Azure ML")
print("=" * 60)

for name, script in steps:
    print(f"\n▶ Running {name}...")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✅ {name} — Complete")
    else:
        print(f"  ❌ {name} — Failed")
        print(result.stderr[-500:])

print("\n" + "=" * 60)
print("✅ Pipeline Complete!")
print("   Outputs saved in: outputs/")
print("   Models saved in : models/")
print("   Run 'mlflow ui' to view experiment tracking")
print("=" * 60)
