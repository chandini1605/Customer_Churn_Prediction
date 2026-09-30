# ============================================================
# STEP 6: MODEL DEPLOYMENT AS REST API
# Simulates Azure Machine Learning Endpoint
# ============================================================

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("=" * 60)
print("STEP 6: Model Deployment Simulation")
print("=" * 60)

# Load model & scaler
best_model  = pickle.load(open('models/best_model.pkl', 'rb'))
scaler      = pickle.load(open('models/scaler.pkl', 'rb'))
top_features = pickle.load(open('models/top_features.pkl', 'rb'))

print(f"✅ Model loaded: {type(best_model).__name__}")
print(f"✅ Scaler loaded")
print(f"✅ Top features: {top_features}")

# ── Predict Function (simulates endpoint) ────────────────────
def predict_churn(input_data: dict) -> dict:
    """
    Simulates Azure ML real-time inference endpoint.
    Input : dict of feature values
    Output: churn prediction + probability
    """
    df = pd.DataFrame([input_data])

    # Ensure correct feature order
    df = df.reindex(columns=top_features, fill_value=0)

    # Scale
    scaled = scaler.transform(df)

    # Predict
    pred = best_model.predict(scaled)[0]
    prob = best_model.predict_proba(scaled)[0][1]

    return {
        "prediction": "Churn" if pred == 1 else "No Churn",
        "churn_probability": round(float(prob), 4),
        "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
    }

# ── Test with Sample Input ────────────────────────────────────
print("\n📌 Testing inference endpoint with sample customer...")

sample = {feat: 0 for feat in top_features}

result = predict_churn(sample)
print(f"\n   Prediction       : {result['prediction']}")
print(f"   Churn Probability: {result['churn_probability']}")
print(f"   Risk Level       : {result['risk_level']}")

print("\n" + "=" * 60)
print("📌 Azure ML Deployment Steps (Reference)")
print("=" * 60)
print("""
On Azure Machine Learning:

1. Register model:
   mlflow.sklearn.log_model(model, "churn_model")
   model_uri = f"runs:/{run_id}/churn_model"
   mv = mlflow.register_model(model_uri, "ChurnPredictor")

2. Create endpoint:
   from azure.ai.ml import MLClient
   from azure.ai.ml.entities import ManagedOnlineEndpoint
   endpoint = ManagedOnlineEndpoint(name="churn-endpoint")
   ml_client.online_endpoints.begin_create_or_update(endpoint)

3. Deploy:
   from azure.ai.ml.entities import ManagedOnlineDeployment
   deployment = ManagedOnlineDeployment(
       name="churn-deploy",
       endpoint_name="churn-endpoint",
       model=registered_model,
       instance_type="Standard_DS2_v2",
       instance_count=1
   )
   ml_client.online_deployments.begin_create_or_update(deployment)

4. Invoke endpoint (REST API call):
   import requests
   response = requests.post(
       url="https://churn-endpoint.azureml.net/score",
       headers={"Authorization": f"Bearer {api_key}"},
       json={"data": [[feature_values]]}
   )
   print(response.json())
""")
print("✅ Deployment script complete!")
