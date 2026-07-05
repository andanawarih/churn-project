import joblib, json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

app = Flask(__name__)

model = tf.keras.models.load_model("artifacts/best_model.keras")
scaler = joblib.load("artifacts/scaler.pkl")
feature_columns = joblib.load("artifacts/feature_columns.pkl")
numeric_cols = joblib.load("artifacts/numeric_cols.pkl")
with open("artifacts/model_info.json") as f:
    model_info = json.load(f)

BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

@app.route("/")
def index():
    return render_template("index.html", model_name=model_info["best_model_name"],
                            metrics=model_info["metrics"])

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    row = {
        "gender": BINARY_MAP[data["gender"]],
        "SeniorCitizen": int(data["SeniorCitizen"]),
        "Partner": BINARY_MAP[data["Partner"]],
        "Dependents": BINARY_MAP[data["Dependents"]],
        "tenure": float(data["tenure"]),
        "PhoneService": BINARY_MAP[data["PhoneService"]],
        "PaperlessBilling": BINARY_MAP[data["PaperlessBilling"]],
        "MonthlyCharges": float(data["MonthlyCharges"]),
        "TotalCharges": float(data["TotalCharges"]),
    }
    df_row = pd.DataFrame([row])

    # One-hot manual untuk kolom multi-kategori, sesuai kolom training
    multi_fields = {
        "MultipleLines": data["MultipleLines"],
        "InternetService": data["InternetService"],
        "OnlineSecurity": data["OnlineSecurity"],
        "OnlineBackup": data["OnlineBackup"],
        "DeviceProtection": data["DeviceProtection"],
        "TechSupport": data["TechSupport"],
        "StreamingTV": data["StreamingTV"],
        "StreamingMovies": data["StreamingMovies"],
        "Contract": data["Contract"],
        "PaymentMethod": data["PaymentMethod"],
    }
    for col, val in multi_fields.items():
        dummy_col = f"{col}_{val}"
        df_row[dummy_col] = 1

    # Reindex agar kolomnya persis sama urutan & jumlahnya seperti saat training
    df_row = df_row.reindex(columns=feature_columns, fill_value=0)

    # Scaling fitur numerik
    df_row[numeric_cols] = scaler.transform(df_row[numeric_cols])

    X_input = df_row.values.astype("float32")
    if model_info["input_shape_type"] == "cnn":
        X_input = X_input.reshape(1, X_input.shape[1], 1)

    prob = float(model.predict(X_input).ravel()[0])
    risk = "Tinggi" if prob >= 0.6 else ("Sedang" if prob >= 0.3 else "Rendah")

    return jsonify({"probability": round(prob * 100, 2), "risk_label": risk})

if __name__ == "__main__":
    app.run(debug=True)