import joblib, json
import pandas as pd
import tensorflow as tf

model = tf.keras.models.load_model("../artifacts/best_model.keras")
scaler = joblib.load("../artifacts/scaler.pkl")
feature_columns = joblib.load("../artifacts/feature_columns.pkl")
numeric_cols = joblib.load("../artifacts/numeric_cols.pkl")
with open("../artifacts/model_info.json") as f:
    model_info = json.load(f)

BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

def build_vector(raw):
    row = {
        "gender": BINARY_MAP[raw["gender"]],
        "SeniorCitizen": int(raw["SeniorCitizen"]),
        "Partner": BINARY_MAP[raw["Partner"]],
        "Dependents": BINARY_MAP[raw["Dependents"]],
        "tenure": float(raw["tenure"]),
        "PhoneService": BINARY_MAP[raw["PhoneService"]],
        "PaperlessBilling": BINARY_MAP[raw["PaperlessBilling"]],
        "MonthlyCharges": float(raw["MonthlyCharges"]),
        "TotalCharges": float(raw["TotalCharges"]),
    }
    df_row = pd.DataFrame([row])
    multi_fields = {k: raw[k] for k in [
        "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
        "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
        "Contract","PaymentMethod"]}
    for col, val in multi_fields.items():
        df_row[f"{col}_{val}"] = 1

    df_row = df_row.reindex(columns=feature_columns, fill_value=0)
    df_row[numeric_cols] = scaler.transform(df_row[numeric_cols])

    print("\n--- kolom non-zero setelah reindex ---")
    nonzero = df_row.loc[:, (df_row != 0).any(axis=0)]
    print(nonzero.T)

    X = df_row.values.astype("float32")
    if model_info["input_shape_type"] == "cnn":
        X = X.reshape(1, X.shape[1], 1)
    return X

# CASE 1: risiko tinggi (harusnya)
high_risk = {
    "gender": "Male", "SeniorCitizen": "1", "Partner": "No", "Dependents": "No",
    "tenure": "2", "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": "109.80", "TotalCharges": "219.60"
}

# CASE 2: risiko rendah (harusnya)
low_risk = {
    "gender": "Female", "SeniorCitizen": "0", "Partner": "Yes", "Dependents": "Yes",
    "tenure": "72", "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": "69.90", "TotalCharges": "5032.80"
}

for name, case in [("HIGH RISK", high_risk), ("LOW RISK", low_risk)]:
    X = build_vector(case)
    prob = model.predict(X, verbose=0).ravel()[0]
    print(f"\n=== {name} → probability: {prob:.4f} ===")
