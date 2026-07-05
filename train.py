import pandas as pd
import numpy as np
import json, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)
from sklearn.utils.class_weight import compute_class_weight
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

X_train_np = X_train.values.astype("float32")
y_train_np = y_train.values
X_val_np = X_val.values.astype("float32")
X_test_np = X_test.values.astype("float32")
y_val_np, y_test_np = y_val.values, y_test.values

# =========================================================
# PENANGANAN IMBALANCE: class_weight (menggantikan oversampling)
# =========================================================
# class_weight memberikan bobot lebih tinggi ke kelas minoritas saat training
# tanpa menduplikasi data — menghasilkan precision yang lebih baik
weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train_np)
class_weight_dict = {0: weights[0], 1: weights[1]}

print(f"\nDistribusi training: {np.bincount(y_train_np)}")
print(f"Class weights: {class_weight_dict}")

n_features = X_train_np.shape[1]

print("\n--- Sanity check data training final ---")
print("NaN di X_train:", np.isnan(X_train_np).sum())
print("Inf di X_train:", np.isinf(X_train_np).sum())
print("Min/Max X_train:", X_train_np.min(), X_train_np.max())

if np.isnan(X_train_np).sum() > 0:
    raise RuntimeError("Masih ada NaN di X_train setelah semua penanganan — cek dataset mentah lebih lanjut.")

# =========================================================
# 5. MODEL 1 — MLP (Dense Network + BatchNormalization)
# =========================================================
def build_mlp(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
        layers.Dense(1, activation="sigmoid")
    ], name="MLP")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

mlp = build_mlp(n_features)
mlp.summary()

es = callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
nan_guard = callbacks.TerminateOnNaN()

hist_mlp = mlp.fit(
    X_train_np, y_train_np,
    validation_data=(X_val_np, y_val_np),
    epochs=200, batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[es, reduce_lr, nan_guard], verbose=1
)

print("\nVal accuracy per epoch (MLP):", [round(v, 4) for v in hist_mlp.history["val_accuracy"]])
print("Val loss per epoch (MLP):", [round(v, 4) for v in hist_mlp.history["val_loss"]])
print("Train accuracy per epoch (MLP):", [round(v, 4) for v in hist_mlp.history["accuracy"]])

mlp_weights_ok = not np.isnan(mlp.get_weights()[0]).any()
print(f"MLP weights valid (tidak NaN): {mlp_weights_ok}")
if not mlp_weights_ok:
    raise RuntimeError("Training MLP masih menghasilkan bobot NaN — perlu investigasi lebih lanjut sebelum lanjut.")

# =========================================================
# 6. MODEL 2 — 1D-CNN (+ BatchNormalization)
# =========================================================
X_train_cnn = X_train_np.reshape(-1, n_features, 1)
X_val_cnn   = X_val_np.reshape(-1, n_features, 1)
X_test_cnn  = X_test_np.reshape(-1, n_features, 1)

def build_cnn(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(64, 3, activation="relu", padding="same", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation="relu", padding="same", kernel_initializer="he_normal"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu", kernel_initializer="he_normal"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ], name="CNN1D")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

cnn = build_cnn(n_features)
cnn.summary()

es2 = callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
reduce_lr2 = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
nan_guard2 = callbacks.TerminateOnNaN()

hist_cnn = cnn.fit(
    X_train_cnn, y_train_np,
    validation_data=(X_val_cnn, y_val_np),
    epochs=200, batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[es2, reduce_lr2, nan_guard2], verbose=1
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
import seaborn as sns

def evaluate(model, X_test_data, y_true, name):
    y_prob = model.predict(X_test_data).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_prob)
    }
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Tidak Churn", "Churn"])
    print(f"\n=== {name} ===")
    print(report)
    print("Confusion Matrix:\n", cm)
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    return metrics, y_prob, y_pred, cm, report

metrics_mlp, prob_mlp, pred_mlp, cm_mlp, report_mlp = evaluate(mlp, X_test_np, y_test_np, "MLP")
metrics_cnn, prob_cnn, pred_cnn, cm_cnn, report_cnn = evaluate(cnn, X_test_cnn, y_test_np, "CNN1D")

comparison_df = pd.DataFrame([metrics_mlp, metrics_cnn])
print("\nPerbandingan Model:\n", comparison_df)

# =========================================================
# 7a. GRAFIK TRAINING — Akurasi & Loss per Model (terpisah)
# =========================================================
for hist, name in [(hist_mlp, "MLP"), (hist_cnn, "CNN1D")]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Grafik Akurasi ---
    axes[0].plot(hist.history["accuracy"], label="Train Accuracy", color="#2563eb", linewidth=2)
    axes[0].plot(hist.history["val_accuracy"], label="Val Accuracy", color="#dc2626", linewidth=2)
    axes[0].set_title(f"{name} — Akurasi per Epoch", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Accuracy", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # --- Grafik Loss ---
    axes[1].plot(hist.history["loss"], label="Train Loss", color="#2563eb", linewidth=2)
    axes[1].plot(hist.history["val_loss"], label="Val Loss", color="#dc2626", linewidth=2)
    axes[1].set_title(f"{name} — Loss per Epoch", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Loss", fontsize=11)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = f"artifacts/training_{name.lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")

# =========================================================
# 7b. TESTING — Confusion Matrix Heatmap per Model
# =========================================================
labels = ["Tidak Churn", "Churn"]

for cm, name in [(cm_mlp, "MLP"), (cm_cnn, "CNN1D")]:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax, annot_kws={"size": 18},
                linewidths=1, linecolor="white")
    ax.set_xlabel("Prediksi", fontsize=13, fontweight="bold")
    ax.set_ylabel("Aktual", fontsize=13, fontweight="bold")
    ax.set_title(f"Confusion Matrix — {name}", fontsize=15, fontweight="bold")

    # Tambahkan keterangan TP, TN, FP, FN di sudut
    tn, fp, fn, tp = cm.ravel()
    summary_text = f"TP={tp}  TN={tn}  FP={fp}  FN={fn}"
    ax.text(0.5, -0.12, summary_text, ha="center", va="top",
            transform=ax.transAxes, fontsize=11, color="#555",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#ccc"))

    plt.tight_layout()
    fname = f"artifacts/confusion_matrix_{name.lower()}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"Saved: {fname}")

# =========================================================
# 7c. TESTING — Tabel Hasil Prediksi (Classification Report)
# =========================================================
for metrics, name in [(metrics_mlp, "MLP"), (metrics_cnn, "CNN1D")]:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")

    table_data = [
        ["Accuracy", f"{metrics['accuracy']*100:.2f}%"],
        ["Precision", f"{metrics['precision']*100:.2f}%"],
        ["Recall", f"{metrics['recall']*100:.2f}%"],
        ["F1-Score", f"{metrics['f1']*100:.2f}%"],
        ["AUC-ROC", f"{metrics['auc_roc']*100:.2f}%"],
    ]

    table = ax.table(cellText=table_data,
                     colLabels=["Metrik", "Nilai"],
                     cellLoc="center", loc="center",
                     colWidths=[0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.8)

    # Styling header
    for j in range(2):
        cell = table[0, j]
        cell.set_facecolor("#141414")
        cell.set_text_props(color="white", fontweight="bold")

    # Styling rows
    for i in range(1, len(table_data) + 1):
        for j in range(2):
            cell = table[i, j]
            cell.set_facecolor("#f5f4ee" if i % 2 == 0 else "#ffffff")
            cell.set_edgecolor("#e0e0e0")

    ax.set_title(f"Hasil Prediksi — {name}", fontsize=15, fontweight="bold", pad=20)
    plt.tight_layout()
    fname = f"artifacts/hasil_prediksi_{name.lower()}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

# =========================================================
# 7d. Perbandingan Metrik (Bar Chart) & ROC Curve
# =========================================================
# --- Bar Chart Perbandingan Metrik ---
fig, ax = plt.subplots(figsize=(10, 6))
comparison_df.set_index("model")[["accuracy", "precision", "recall", "f1", "auc_roc"]].plot(
    kind="bar", ax=ax, colormap="Set2", edgecolor="black", linewidth=0.5
)
ax.set_title("Perbandingan Metrik MLP vs CNN1D", fontsize=14, fontweight="bold")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.legend(loc="lower right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
plt.tight_layout()
plt.savefig("artifacts/model_comparison.png", dpi=150)
plt.close()
print("Saved: artifacts/model_comparison.png")

# --- ROC Curve ---
fig, ax = plt.subplots(figsize=(8, 6))
for name, y_prob, color in [("MLP", prob_mlp, "#2563eb"), ("CNN1D", prob_cnn, "#dc2626")]:
    fpr, tpr, _ = roc_curve(y_test_np, y_prob)
    auc_val = roc_auc_score(y_test_np, y_prob)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC = {auc_val:.3f})")
ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Random (AUC = 0.500)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curve — MLP vs CNN1D", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("artifacts/roc_curve.png", dpi=150)
plt.close()
print("Saved: artifacts/roc_curve.png")

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

print(f"\nModel terbaik: {best['model']} (F1={best['f1']:.4f}) -> disimpan di artifacts/best_model.keras")