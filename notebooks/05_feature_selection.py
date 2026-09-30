# ============================================================
# STEP 5: CORRECT FEATURE SELECTION (All Methods)
# ✅ Correlation Matrix → Chi-Square → XGBoost Importance
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2, f_classif, SelectKBest
from scipy.stats import chi2_contingency
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

import os
os.makedirs("outputs/feature_plots", exist_ok=True)

df = pd.read_csv("data/Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.drop(columns=['customerID'], inplace=True)
df['Churn'] = (df['Churn'] == 'Yes').astype(int)

print("=" * 60)
print("STEP 5: Correct Feature Selection")
print("⚠️  Correlation alone is NOT enough!")
print("=" * 60)

# ── Method 1: Correlation Matrix ──────────────────────────────
print("\n📌 METHOD 1: Correlation Matrix")
print("   ⚠️  Only captures LINEAR relationships")
print("   ⚠️  Only works for NUMERICAL columns")

num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
corr = df[num_cols + ['Churn']].corr()['Churn'].drop('Churn').abs().sort_values(ascending=False)
print(corr)

plt.figure(figsize=(8, 5))
sns.heatmap(df[num_cols + ['Churn']].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix\n(⚠️ Linear relationships only, numerical features only)')
plt.tight_layout()
plt.savefig('outputs/feature_plots/01_correlation_matrix.png', dpi=150)
plt.close()
print("✅ Saved correlation matrix plot")

# ── Method 2: Chi-Square Test (Categorical → Target) ──────────
print("\n📌 METHOD 2: Chi-Square Test")
print("   ✅ Works for CATEGORICAL features vs target")

cat_cols = ['gender', 'Partner', 'Dependents', 'PhoneService',
            'InternetService', 'Contract', 'PaymentMethod', 'PaperlessBilling']

chi2_results = {}
for col in cat_cols:
    ct = pd.crosstab(df[col], df['Churn'])
    chi2_stat, p_val, dof, expected = chi2_contingency(ct)
    chi2_results[col] = {'chi2': chi2_stat, 'p_value': p_val}
    sig = "✅ Significant" if p_val < 0.05 else "❌ Not significant"
    print(f"   {col:25s} chi2={chi2_stat:.2f}  p={p_val:.4f}  {sig}")

chi2_df = pd.DataFrame(chi2_results).T.sort_values('chi2', ascending=False)
plt.figure(figsize=(10, 5))
chi2_df['chi2'].plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('Chi-Square Test — Categorical Feature Importance\n(✅ Captures non-linear relationships for categorical cols)')
plt.xlabel('Chi2 Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/feature_plots/02_chi2_scores.png', dpi=150)
plt.close()
print("✅ Saved chi-square plot")

# ── Method 3: XGBoost Feature Importance ──────────────────────
print("\n📌 METHOD 3: XGBoost Feature Importance")
print("   ✅ Captures NON-LINEAR relationships")
print("   ✅ Works for ALL feature types")
print("   ✅ Most reliable method")

df_encoded = df.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

xgb = XGBClassifier(n_estimators=100, random_state=42,
                     eval_metric='logloss', use_label_encoder=False)
xgb.fit(X, y)

feat_imp = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n   Top 10 Important Features:")
print(feat_imp.head(10))

plt.figure(figsize=(10, 6))
feat_imp.head(10).plot(kind='barh', color='tomato', edgecolor='black')
plt.title('XGBoost Feature Importance\n(✅ Best method — captures all relationships)')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/feature_plots/03_xgboost_importance.png', dpi=150)
plt.close()
print("✅ Saved XGBoost importance plot")

# ── Summary: Why All 3 Methods Are Needed ─────────────────────
print("\n" + "=" * 60)
print("📊 SUMMARY: Why All 3 Methods Are Needed")
print("=" * 60)
summary = pd.DataFrame({
    'Method': ['Correlation Matrix', 'Chi-Square Test', 'XGBoost Importance'],
    'Feature Type': ['Numerical only', 'Categorical only', 'All types'],
    'Relationship': ['Linear only', 'Non-linear', 'Non-linear'],
    'Best For': ['Quick numerical check', 'Categorical significance', 'Final selection']
})
print(summary.to_string(index=False))
print("\n✅ Feature selection complete!")
