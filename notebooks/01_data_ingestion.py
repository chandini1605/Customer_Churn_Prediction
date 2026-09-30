# ============================================================
# STEP 1: DATA INGESTION & STORAGE
# Customer Churn Prediction Pipeline - Azure Databricks
# ============================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when
import pandas as pd

# Initialize Spark Session
# On Azure Databricks this is auto-created; locally use below
spark = SparkSession.builder \
    .appName("ChurnPrediction_Ingestion") \
    .getOrCreate()

# ── 1. Load Raw CSV Data ──────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading Raw Dataset")
print("=" * 60)

# Load dataset (IBM Telco Customer Churn)
df = spark.read.csv(
    "data/Telco-Customer-Churn.csv",
    header=True,
    inferSchema=True
)

print(f"✅ Dataset loaded successfully!")
print(f"   Rows    : {df.count():,}")
print(f"   Columns : {len(df.columns)}")

# ── 2. Preview Data ───────────────────────────────────────────
print("\nSchema:")
df.printSchema()

print("\nSample Records:")
df.show(5, truncate=False)

# ── 3. Save as Delta Lake / Parquet ──────────────────────────
print("\nSaving to Delta/Parquet format...")

# On Databricks: df.write.format("delta").save("/mnt/churn/raw")
# Locally: save as parquet
df.write.mode("overwrite").parquet("outputs/raw_data.parquet")

print("✅ Data saved to outputs/raw_data.parquet")

# ── 4. Basic Info ─────────────────────────────────────────────
print(f"\nTotal Records  : {df.count():,}")
print(f"Total Features : {len(df.columns)}")
print(f"Target Column  : Churn")

spark.stop()
