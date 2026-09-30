# ============================================================
# STEP 3: DATA PREPROCESSING & FEATURE ENGINEERING
# Customer Churn Prediction Pipeline
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency, f_oneway
import pickle
import os

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("=" * 60)
print("STEP 3: Preprocessing & Feature Engineering")
print("=" * 60)

# ── 1. Fix Data Types ─────────────────────────────────────────
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.drop(columns=['customerID'], inplace=True)
print("✅ Fixed TotalCharges dtype, dropped customerID")

# ── 2. Handle Missing Values ──────────────────────────────────
print(f"\n📌 Missing values: {df.isnull().sum().sum()}")

# ── 3. Check Unique Values Before Encoding ────────────────────
# ✅ CORRECT STEP: Always check unique values before encoding!
print("\n📌 Checking Unique Values Before Encoding:")
print("-" * 50)
binary_cols = []
nominal_cols = []

for col in df.select_dtypes(include='object').columns:
    unique_vals = df[col].unique()
    n_unique = len(unique_vals)
    print(f"   {col:25s} → {n_unique} unique → {unique_vals}")

    if n_unique == 2:
        binary_cols.append(col)
    elif n_unique > 2 and col != 'Churn':
        nominal_cols.append(col)

print(f"\n   Binary columns  (Label Encode) : {binary_cols}")
print(f"   Nominal columns (One-Hot Encode): {nominal_cols}")

# ── 4. Encode Target Variable ─────────────────────────────────
df['Churn'] = (df['Churn'] == 'Yes').astype(int)
print("\n✅ Target encoded: Yes=1, No=0")
print(f"   Churn distribution:\n{df['Churn'].value_counts()}")

# ── 5. Label Encoding for Binary Columns ─────────────────────
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])
    print(f"   Label Encoded: {col}")

# ── 6. One-Hot Encoding for Nominal Columns ───────────────────
df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
print(f"\n✅ One-Hot Encoded {len(nominal_cols)} nominal columns")
print(f"   Shape after encoding: {df.shape}")

# ── 7. Feature Engineering ────────────────────────────────────
print("\n📌 Feature Engineering...")
df['tenure_group'] = pd.cut(df['tenure'],
    bins=[0, 12, 24, 48, 60, 72],
    labels=[1, 2, 3, 4, 5]).astype(float)

df['monthly_to_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
df['high_value_customer'] = (df['MonthlyCharges'] > df['MonthlyCharges'].median()).astype(int)
df['tenure_group'].fillna(df['tenure_group'].mode()[0], inplace=True)
print("✅ Created: tenure_group, monthly_to_total_ratio, high_value_customer")

# ── 8. Feature Selection ──────────────────────────────────────
# ✅ CORRECT: Multiple methods, not just correlation!
print("\n📌 Feature Selection (Multiple Methods):")

X = df.drop('Churn', axis=1)
y = df['Churn']

# Method 1: Correlation (numerical only — limited to linear)
print("\n   Method 1: Correlation with target (numerical only — linear relationships)")
num_cols = X.select_dtypes(include=[np.number]).columns
corr_scores = abs(df[num_cols].corrwith(y)).sort_values(ascending=False)
print(corr_scores.head(10))

# Method 2: Chi-Square for categorical (already encoded as 0/1 dummies)
print("\n   Method 2: ANOVA F-Test (numerical vs binary target)")
from sklearn.feature_selection import f_classif, SelectKBest
selector = SelectKBest(f_classif, k='all')
selector.fit(X, y)
f_scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
print("   Top 10 features by F-score:")
print(f_scores.head(10))

# Select top features
top_features = f_scores.head(15).index.tolist()
X = X[top_features]
print(f"\n✅ Selected top {len(top_features)} features")

# ── 9. Train-Test Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n✅ Train-Test Split (80/20 stratified):")
print(f"   X_train: {X_train.shape}, X_test: {X_test.shape}")

# ── 10. Feature Scaling ───────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
print("✅ StandardScaler applied")

# Save preprocessed data & scaler
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
pickle.dump(top_features, open('models/top_features.pkl', 'wb'))

np.save('outputs/X_train.npy', X_train_scaled)
np.save('outputs/X_test.npy',  X_test_scaled)
np.save('outputs/y_train.npy', y_train.values)
np.save('outputs/y_test.npy',  y_test.values)

print("\n✅ Preprocessed data saved to outputs/")
print("✅ Scaler saved to models/scaler.pkl")
