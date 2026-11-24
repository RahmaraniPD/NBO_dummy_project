"""
MELATIH & MEMILIH MODEL NEXT BEST OFFER
----------------------------------------------------
Langkah-langkah:
  1. Baca data
  2. Pisahkan fitur (X) dan target (y)
  3. Bagi data: train (80%) dan test (20%)
  4. Bangun pipeline preprocessing + model
  5. Bandingkan 3 model dengan cross-validation
  6. Tuning model terbaik (hyperparameter)
  7. Evaluasi final di data test
  8. Simpan model ke file .pkl (dipakai di UI nanti)

Run :  python 02_latih_model.py
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # supaya grafik bisa disimpan tanpa membuka jendela
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay, top_k_accuracy_score)
from sklearn.inspection import permutation_importance

# ---------- 1. Baca data ----------
df = pd.read_csv("data/data_pelanggan.csv")
print("Jumlah data:", df.shape)

# ---------- 2. Pisahkan fitur dan target ----------
KOLOM_ANGKA = ["usia", "penghasilan_bulanan_juta", "jumlah_produk_dimiliki",
               "lama_jadi_customer_bulan", "riwayat_telat_bayar", "buka_app_30hari",
               "sisa_cicilan_bulan", "rasio_cicilan_penghasilan"]
KOLOM_KATEGORI = ["jenis_kelamin", "kota", "pekerjaan", "produk_terakhir"]
FITUR = KOLOM_ANGKA + KOLOM_KATEGORI
TARGET = "produk_berikutnya"

X = df[FITUR]
y = df[TARGET]

# ---------- 3. Bagi train & test ----------
# stratify=y menjaga proporsi tiap produk tetap sama di train dan test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape, "| Test:", X_test.shape)

# ---------- 4. Preprocessing ----------
# Angka -> distandarkan (rata-rata 0). Kategori -> diubah jadi kolom 0/1 (One-Hot).
preprocessing = ColumnTransformer([
    ("angka", StandardScaler(), KOLOM_ANGKA),
    ("kategori", OneHotEncoder(handle_unknown="ignore"), KOLOM_KATEGORI),
])

# ---------- 5. Bandingkan beberapa model ----------
kandidat = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
hasil_cv = {}
print("\n=== Perbandingan model (5-fold cross validation, metrik: F1 macro) ===")
for nama, model in kandidat.items():
    pipe = Pipeline([("prep", preprocessing), ("model", model)])
    skor = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1_macro")
    hasil_cv[nama] = skor.mean()
    print(f"{nama:22s} F1 macro = {skor.mean():.3f}  (+/- {skor.std():.3f})")

model_terbaik_nama = max(hasil_cv, key=hasil_cv.get)
print("\nModel terbaik sebelum tuning:", model_terbaik_nama)

# ---------- 6. Tuning hyperparameter ----------
grid_parameter = {
    "Logistic Regression": {"model__C": [0.1, 1, 10]},
    "Random Forest": {"model__n_estimators": [200, 400],
                      "model__max_depth": [None, 10, 20],
                      "model__min_samples_leaf": [1, 5]},
    "Gradient Boosting": {"model__learning_rate": [0.03, 0.1],
                          "model__max_depth": [3, 5],
                          "model__max_iter": [100, 200]},
}

pipe_terbaik = Pipeline([("prep", preprocessing), ("model", kandidat[model_terbaik_nama])])
grid = GridSearchCV(pipe_terbaik, grid_parameter[model_terbaik_nama],
                    cv=cv, scoring="f1_macro", n_jobs=-1)
grid.fit(X_train, y_train)
print("Parameter terbaik:", grid.best_params_)
print(f"F1 macro (CV) setelah tuning: {grid.best_score_:.3f}")

model_final = grid.best_estimator_

# ---------- 7. Evaluasi final di data test ----------
y_pred = model_final.predict(X_test)
proba = model_final.predict_proba(X_test)
kelas = model_final.classes_

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="macro")
top2 = top_k_accuracy_score(y_test, proba, k=2, labels=kelas)

print("\n=== HASIL FINAL DI DATA TEST ===")
print(f"Accuracy       : {acc:.3f}")
print(f"F1 macro       : {f1:.3f}")
print(f"Top-2 accuracy : {top2:.3f}  (produk asli ada di 2 rekomendasi teratas)")
print("\nLaporan per produk:")
print(classification_report(y_test, y_pred))

# Confusion matrix -> disimpan sebagai gambar
fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred, labels=kelas),
                       display_labels=kelas).plot(ax=ax, xticks_rotation=30, colorbar=False)
plt.title(f"Confusion Matrix - {model_terbaik_nama}")
plt.tight_layout()
plt.savefig("output/confusion_matrix.png", dpi=120)
plt.close()

# Feature importance -> fitur apa yang paling berpengaruh
imp = permutation_importance(model_final, X_test, y_test, n_repeats=5,
                             random_state=42, scoring="f1_macro", n_jobs=-1)
df_imp = pd.Series(imp.importances_mean, index=FITUR).sort_values()
df_imp.plot(kind="barh", figsize=(7, 5), title="Fitur paling berpengaruh")
plt.xlabel("Penurunan F1 jika fitur diacak")
plt.tight_layout()
plt.savefig("output/feature_importance.png", dpi=120)
plt.close()
print("\nFitur paling berpengaruh:")
print(df_imp.sort_values(ascending=False).round(4).to_string())

# ---------- 8. Simpan model ----------
joblib.dump(model_final, "model/model_nbo.pkl")
with open("model/info_model.json", "w") as f:
    json.dump({"fitur": FITUR, "kelas": list(kelas), "model": model_terbaik_nama,
               "accuracy": round(acc, 3), "f1_macro": round(f1, 3)}, f, indent=2)
print("\nModel tersimpan di model/model_nbo.pkl")

# ---------- 9. Contoh pemakaian (ini yang nanti dipakai UI) ----------
contoh = X_test.iloc[[0]]
p = model_final.predict_proba(contoh)[0]
urut = np.argsort(p)[::-1]
print("\nContoh prediksi 1 pelanggan:")
print(contoh.T.to_string(header=False))
for i in urut:
    print(f"  {kelas[i]:20s} {p[i]*100:5.1f}%")
print(">> Next Best Offer:", kelas[urut[0]])
