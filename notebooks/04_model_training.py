# ============================================================
# STEP 4: MODEL TRAINING WITH MLFLOW TRACKING
# Customer Churn Prediction Pipeline
# ============================================================

import numpy as np
import pandas as pd
import pickle
import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report,
                              confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

os.makedirs("outputs/model_plots", exist_ok=True)

# Load preprocessed data
X_train = np.load('outputs/X_train.npy')
X_test  = np.load('outputs/X_test.npy')
y_train = np.load('outputs/y_train.npy')
y_test  = np.load('outputs/y_test.npy')

print("=" * 60)
print("STEP 4: Model Training with MLflow Tracking")
print("=" * 60)
print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# ── MLflow Setup ─────────────────────────────────────────────
# On Azure Databricks: mlflow is pre-configured
# Locally: tracks to ./mlruns folder
mlflow.set_experiment("Customer_Churn_Prediction")

# ── Define Models ────────────────────────────────────────────
models = {
    "Logistic_Regression": LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42
    ),
    "Random_Forest": RandomForestClassifier(
        n_estimators=100, class_weight='balanced',
        random_state=42, n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=100, scale_pos_weight=3,
        random_state=42, eval_metric='logloss',
        use_label_encoder=False
    )
}

results = {}

# ── Train & Track Each Model ─────────────────────────────────
for model_name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}")

    with mlflow.start_run(run_name=model_name):

        # Log model parameters
        mlflow.log_params(model.get_params())

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob)

        # Log metrics to MLflow
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)
        mlflow.log_metric("f1_score",  f1)
        mlflow.log_metric("roc_auc",   auc)

        # Log model artifact
        if 'XGBoost' in model_name:
            import mlflow.xgboost
            mlflow.xgboost.log_model(model, model_name)
        else:
            mlflow.sklearn.log_model(model, model_name)

        # Print results
        print(f"   Accuracy  : {acc:.4f}")
        print(f"   Precision : {prec:.4f}")
        print(f"   Recall    : {rec:.4f}")
        print(f"   F1 Score  : {f1:.4f}")
        print(f"   ROC AUC   : {auc:.4f}")
        print(f"\n   Classification Report:\n{classification_report(y_test, y_pred)}")

        results[model_name] = {
            'model': model,
            'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'auc': auc,
            'y_pred': y_pred, 'y_prob': y_prob
        }

        # Confusion Matrix plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['No Churn','Churn'],
                    yticklabels=['No Churn','Churn'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        path = f'outputs/model_plots/cm_{model_name}.png'
        plt.savefig(path, dpi=150)
        plt.close()
        mlflow.log_artifact(path)

# ── Feature Importance (XGBoost) ─────────────────────────────
print("\n📌 XGBoost Feature Importance:")
top_features = pickle.load(open('models/top_features.pkl', 'rb'))
xgb_model = results['XGBoost']['model']
feat_imp = pd.Series(xgb_model.feature_importances_, index=top_features).sort_values(ascending=False)
print(feat_imp.head(10))

plt.figure(figsize=(10, 6))
feat_imp.head(10).plot(kind='barh', color='steelblue', edgecolor='black')
plt.title('XGBoost Feature Importance\n(Captures non-linear relationships)')
plt.xlabel('Importance Score')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/model_plots/feature_importance_xgboost.png', dpi=150)
plt.close()
print("✅ Saved: feature_importance_xgboost.png")

# ── Model Comparison ──────────────────────────────────────────
print("\n📌 Model Comparison:")
comp_df = pd.DataFrame({
    name: {
        'Accuracy': r['accuracy'],
        'Precision': r['precision'],
        'Recall': r['recall'],
        'F1': r['f1'],
        'AUC': r['auc']
    } for name, r in results.items()
}).T
print(comp_df.round(4))

comp_df.plot(kind='bar', figsize=(12, 5), edgecolor='black')
plt.title('Model Comparison')
plt.ylabel('Score')
plt.xticks(rotation=15)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('outputs/model_plots/model_comparison.png', dpi=150)
plt.close()

# ── Select Best Model ─────────────────────────────────────────
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']
print(f"\n🏆 Best Model: {best_model_name}")
print(f"   F1 Score : {results[best_model_name]['f1']:.4f}")
print(f"   Accuracy : {results[best_model_name]['accuracy']:.4f}")

# Save best model
pickle.dump(best_model, open('models/best_model.pkl', 'wb'))
print(f"✅ Best model saved to models/best_model.pkl")
print("\n✅ All models trained and tracked with MLflow!")
print("   Run 'mlflow ui' to view experiment dashboard")
