# Churnlytics — Customer Churn Prediction

Sistem prediksi churn pelanggan menggunakan Deep Learning (MLP & CNN1D), dibangun dengan TensorFlow/Keras dan disajikan melalui web app Flask.

## 📁 Struktur Project

```
churn-project/
├── app.py                  # Web app Flask (endpoint prediksi)
├── train.py                # Script training model (MLP & CNN1D)
├── requirements.txt        # Daftar dependensi Python
├── .gitignore              # File yang diabaikan Git
├── README.md               # Dokumentasi project (file ini)
│
├── data/                   # Dataset
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── artifacts/              # Hasil training (model, scaler, metadata)
│   ├── best_model.keras    # Model terbaik (otomatis dipilih dari MLP/CNN1D)
│   ├── scaler.pkl          # StandardScaler untuk fitur numerik
│   ├── feature_columns.pkl # Urutan kolom fitur saat training
│   ├── numeric_cols.pkl    # Daftar kolom numerik
│   ├── model_info.json     # Info model terbaik & metrik evaluasi
│   └── model_comparison.png# Grafik perbandingan MLP vs CNN1D
│
├── templates/              # Template HTML (Jinja2)
│   └── index.html
│
├── static/                 # Aset frontend
│   ├── css/style.css
│   └── js/script.js
│
└── notebooks/              # Script debug & eksplorasi
    ├── debug_predict.py
    ├── debug_activations.py
    └── inspect_weights.py
```

## 🚀 Cara Menjalankan

### 1. Install dependensi

```bash
pip install -r requirements.txt
```

### 2. Training model (opsional, artifacts sudah tersedia)

```bash
python train.py
```

Akan menghasilkan file-file di folder `artifacts/`.

### 3. Jalankan web app

```bash
python app.py
```

Buka browser ke **http://127.0.0.1:5000**

## 🧠 Tentang Model

Project ini membandingkan dua arsitektur Deep Learning:

| Model | Deskripsi |
|-------|-----------|
| **MLP** | Multi-Layer Perceptron (Dense Network) |
| **CNN1D** | 1D Convolutional Neural Network |

Model terbaik dipilih otomatis berdasarkan **F1-Score** pada data test, lalu disimpan ke `artifacts/best_model.keras`.

### Pipeline Data
1. **Input** — Load dataset Telco Customer Churn
2. **Preprocessing** — Handle missing values, capping outlier (IQR)
3. **Transformation** — Binary encoding, one-hot encoding, StandardScaler
4. **Splitting** — 80% train, 10% validasi, 10% test (stratified)
5. **Oversampling** — Random oversampling kelas minoritas (train saja)
6. **Training** — MLP & CNN1D dengan EarlyStopping
7. **Evaluasi** — Accuracy, Precision, Recall, F1-Score
8. **Deploy** — Flask web app dengan prediksi real-time

## 👥 Kelompok 5

Mata Kuliah: Machine Learning Praktikum — Semester 4
