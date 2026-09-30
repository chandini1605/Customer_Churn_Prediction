# ============================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# Customer Churn Prediction Pipeline
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs/eda_plots", exist_ok=True)

# Load dataset
df = pd.read_csv("data/Telco-Customer-Churn.csv")
print("=" * 60)
print("STEP 2: Exploratory Data Analysis")
print("=" * 60)

# ── 1. Basic Info ─────────────────────────────────────────────
print("\n📌 Shape:", df.shape)
print("\n📌 Data Types:\n", df.dtypes)
print("\n📌 First 5 rows:\n", df.head())

# ── 2. Check Missing Values ───────────────────────────────────
print("\n📌 Missing Values:")
print(df.isnull().sum())

# TotalCharges has spaces - fix it
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(f"\n   TotalCharges nulls after conversion: {df['TotalCharges'].isnull().sum()}")

# ── 3. Check Unique Values for Each Column ────────────────────
# ✅ CORRECT STEP - Must check before encoding!
print("\n📌 Unique Values Per Column:")
print("-" * 40)
for col in df.columns:
    print(f"   {col:25s} → {df[col].nunique()} unique : {df[col].unique()[:5]}")

# ── 4. Target Variable Analysis ───────────────────────────────
print("\n📌 Churn Distribution (Value Counts):")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True).round(3) * 100)

# Plot churn distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df['Churn'].value_counts().plot(kind='bar', ax=axes[0], color=['steelblue','tomato'], edgecolor='black')
axes[0].set_title('Churn Distribution (Count)')
axes[0].set_xlabel('Churn')
axes[0].set_ylabel('Count')
axes[0].tick_params(rotation=0)

df['Churn'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                                colors=['steelblue','tomato'], startangle=90)
axes[1].set_title('Churn Distribution (%)')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('outputs/eda_plots/01_churn_distribution.png', dpi=150)
plt.close()
print("✅ Saved: 01_churn_distribution.png")

# ── 5. Numerical Features Distribution ────────────────────────
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(num_cols):
    df[col].hist(ax=axes[i], bins=30, color='steelblue', edgecolor='black')
    axes[i].set_title(f'Distribution of {col}')
    axes[i].set_xlabel(col)
plt.tight_layout()
plt.savefig('outputs/eda_plots/02_numerical_distributions.png', dpi=150)
plt.close()
print("✅ Saved: 02_numerical_distributions.png")

# ── 6. Churn by Key Categorical Features ─────────────────────
cat_cols = ['Contract', 'InternetService', 'PaymentMethod', 'gender']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    churn_rate = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    churn_rate.plot(kind='bar', ax=axes[i], color='tomato', edgecolor='black')
    axes[i].set_title(f'Churn Rate by {col}')
    axes[i].set_ylabel('Churn Rate (%)')
    axes[i].tick_params(rotation=30)

plt.tight_layout()
plt.savefig('outputs/eda_plots/03_churn_by_category.png', dpi=150)
plt.close()
print("✅ Saved: 03_churn_by_category.png")

# ── 7. Correlation Matrix (Numerical Only) ────────────────────
# ⚠️ Note: Only shows linear relationships among numerical cols
df_temp = df.copy()
df_temp['Churn_num'] = (df_temp['Churn'] == 'Yes').astype(int)
corr = df_temp[['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn_num']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Correlation Matrix (Numerical Features Only)\n⚠️ Only captures linear relationships')
plt.tight_layout()
plt.savefig('outputs/eda_plots/04_correlation_matrix.png', dpi=150)
plt.close()
print("✅ Saved: 04_correlation_matrix.png")

print("\n✅ EDA Complete! All plots saved in outputs/eda_plots/")
