import pandas as pd
import numpy as np
import json, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

os.makedirs("artifacts", exist_ok=True)
np.random.seed(42)
tf.random.set_seed(42)

# =========================================================
# 1. INPUT
# =========================================================
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop(columns=["customerID"], inplace=True)

# =========================================================
# 2. PREPROCESSING
# =========================================================
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# FIX: assignment langsung, bukan inplace=True pada chained selection
# (di pandas Copy-on-Write, df[col].fillna(..., inplace=True) tidak benar-benar mengubah df)
median_total_charges = df["TotalCharges"].median()
df["TotalCharges"] = df["TotalCharges"].fillna(median_total_charges)

print("NaN di TotalCharges setelah fillna:", df["TotalCharges"].isnull().sum())

def cap_outliers_iqr(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return series.clip(lower, upper)

numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
outlier_report = {}
for col in numeric_cols:
    before = df[col].copy()
    df[col] = cap_outliers_iqr(df[col])
    outlier_report[col] = int((before != df[col]).sum())
print("Jumlah nilai yang dicapping per kolom (outlier):", outlier_report)

# =========================================================
# 3. TRANSFORMATION
# =========================================================
binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]
for col in binary_cols:
    df[col] = df[col].map(binary_map)

multi_cat_cols = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                   "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                   "Contract", "PaymentMethod"]
df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

y = df["Churn"].astype(int)
X = df.drop(columns=["Churn"]).astype("float64")

print("\n--- Cek dtype setelah transformasi ---")
print(X.dtypes.value_counts())
print("Ada nilai NaN di X:", X.isnull().values.any())
print("Kolom yang mengandung NaN:", X.columns[X.isnull().any()].tolist())
print("Ada nilai Inf di X:", np.isinf(X.values).any())

# Safety net terakhir: kalau masih ada NaN tersisa di kolom manapun, isi dengan median kolom itu
if X.isnull().values.any():
    X = X.fillna(X.median())
    print("NaN ditemukan dan sudah diisi dengan median masing-masing kolom.")

feature_columns = list(X.columns)
joblib.dump(feature_columns, "artifacts/feature_columns.pkl")

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
joblib.dump(scaler, "artifacts/scaler.pkl")
joblib.dump(numeric_cols, "artifacts/numeric_cols.pkl")

# Catatan: tidak ada windowing karena data bersifat cross-sectional.

# =========================================================
# 4. DATA SPLITTING (80 : 10 : 10, stratified)
# =========================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

X_train_np_raw = X_train.values.astype("float32")
y_train_np_raw = y_train.values
X_val_np = X_val.values.astype("float32")
X_test_np = X_test.values.astype("float32")
y_val_np, y_test_np = y_val.values, y_test.values

# =========================================================
# PENANGANAN IMBALANCE: oversampling kelas minoritas di TRAIN SAJA
# =========================================================
idx_majority = np.where(y_train_np_raw == 0)[0]
idx_minority = np.where(y_train_np_raw == 1)[0]

rng = np.random.RandomState(42)
idx_minority_upsampled = rng.choice(idx_minority, size=len(idx_majority), replace=True)
idx_balanced = np.concatenate([idx_majority, idx_minority_upsampled])
rng.shuffle(idx_balanced)

X_train_np = X_train_np_raw[idx_balanced]
y_train_np = y_train_np_raw[idx_balanced]

print(f"\nDistribusi training sebelum oversampling: {np.bincount(y_train_np_raw)}")
print(f"Distribusi training setelah oversampling: {np.bincount(y_train_np)}")

n_features = X_train_np.shape[1]

print("\n--- Sanity check data training final ---")
print("NaN di X_train:", np.isnan(X_train_np).sum())
print("Inf di X_train:", np.isinf(X_train_np).sum())
print("Min/Max X_train:", X_train_np.min(), X_train_np.max())

if np.isnan(X_train_np).sum() > 0:
    raise RuntimeError("Masih ada NaN di X_train setelah semua penanganan — cek dataset mentah lebih lanjut.")

# =========================================================
# 5. MODEL 1 — MLP (Dense Network, tanpa BatchNorm — input sudah discale)
# =========================================================
def build_mlp(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
        layers.Dropout(0.3),
        layers.Dense(16, activation="relu", kernel_initializer="he_normal"),
        layers.Dense(1, activation="sigmoid")
    ], name="MLP")
    model.compile(
        optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

mlp = build_mlp(n_features)
es = callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
nan_guard = callbacks.TerminateOnNaN()

hist_mlp = mlp.fit(
    X_train_np, y_train_np,
    validation_data=(X_val_np, y_val_np),
    epochs=150, batch_size=64,
    callbacks=[es, nan_guard], verbose=1
)

print("\nVal accuracy per epoch (MLP):", [round(v, 4) for v in hist_mlp.history["val_accuracy"]])
print("Val loss per epoch (MLP):", [round(v, 4) for v in hist_mlp.history["val_loss"]])
print("Train accuracy per epoch (MLP):", [round(v, 4) for v in hist_mlp.history["accuracy"]])

mlp_weights_ok = not np.isnan(mlp.get_weights()[0]).any()
print(f"MLP weights valid (tidak NaN): {mlp_weights_ok}")
if not mlp_weights_ok:
    raise RuntimeError("Training MLP masih menghasilkan bobot NaN — perlu investigasi lebih lanjut sebelum lanjut.")

# =========================================================
# 6. MODEL 2 — 1D-CNN (tanpa BatchNorm)
# =========================================================
X_train_cnn = X_train_np.reshape(-1, n_features, 1)
X_val_cnn   = X_val_np.reshape(-1, n_features, 1)
X_test_cnn  = X_test_np.reshape(-1, n_features, 1)

def build_cnn(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal"),
        layers.MaxPooling1D(2),
        layers.Conv1D(16, 3, activation="relu", padding="same", kernel_initializer="he_normal"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation="relu", kernel_initializer="he_normal"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ], name="CNN1D")
    model.compile(
        optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

cnn = build_cnn(n_features)
es2 = callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
nan_guard2 = callbacks.TerminateOnNaN()

hist_cnn = cnn.fit(
    X_train_cnn, y_train_np,
    validation_data=(X_val_cnn, y_val_np),
    epochs=150, batch_size=64,
    callbacks=[es2, nan_guard2], verbose=1
)

print("\nVal accuracy per epoch (CNN1D):", [round(v, 4) for v in hist_cnn.history["val_accuracy"]])
print("Val loss per epoch (CNN1D):", [round(v, 4) for v in hist_cnn.history["val_loss"]])
print("Train accuracy per epoch (CNN1D):", [round(v, 4) for v in hist_cnn.history["accuracy"]])

cnn_weights_ok = not np.isnan(cnn.get_weights()[0]).any()
print(f"CNN weights valid (tidak NaN): {cnn_weights_ok}")
if not cnn_weights_ok:
    raise RuntimeError("Training CNN masih menghasilkan bobot NaN — perlu investigasi lebih lanjut sebelum lanjut.")

# =========================================================
# 7. EVALUASI & KOMPARASI
# =========================================================
def evaluate(model, X_test_data, y_true, name):
    y_prob = model.predict(X_test_data).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    return metrics, y_prob

metrics_mlp, _ = evaluate(mlp, X_test_np, y_test_np, "MLP")
metrics_cnn, _ = evaluate(cnn, X_test_cnn, y_test_np, "CNN1D")

comparison_df = pd.DataFrame([metrics_mlp, metrics_cnn])
print("\nPerbandingan Model:\n", comparison_df)

comparison_df.set_index("model")[["accuracy", "precision", "recall", "f1"]].plot(kind="bar", figsize=(8, 5))
plt.title("Perbandingan Metrik MLP vs CNN1D")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("artifacts/model_comparison.png")
plt.close()

# =========================================================
# 8. SIMPAN MODEL TERBAIK (berdasarkan F1-Score)
# =========================================================
best = metrics_mlp if metrics_mlp["f1"] >= metrics_cnn["f1"] else metrics_cnn
best_model = mlp if best["model"] == "MLP" else cnn
best_input_shape = "flat" if best["model"] == "MLP" else "cnn"

best_model.save("artifacts/best_model.keras")
with open("artifacts/model_info.json", "w") as f:
    json.dump({
        "best_model_name": best["model"],
        "input_shape_type": best_input_shape,
        "metrics": {"MLP": metrics_mlp, "CNN1D": metrics_cnn}
    }, f, indent=2)

print(f"\nModel terbaik: {best['model']} (F1={best['f1']:.4f}) → disimpan di artifacts/best_model.keras")