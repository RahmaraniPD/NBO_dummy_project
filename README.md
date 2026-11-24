# Next Best Offer Assistant

Project portofolio data analyst: model **Next Best Offer (NBO)** untuk merekomendasikan produk pembiayaan berikutnya kepada pelanggan, lengkap dengan aplikasi web sederhana untuk agen.

> Seluruh data adalah **data dummy** buatan sendiri, bukan data perusahaan.

## Latar Belakang
Agen sering bingung produk apa yang paling tepat ditawarkan ke pelanggan existing. Project ini membangun model prediksi untuk memberi rekomendasi produk berikutnya, dan UI agar agen bisa langsung memakainya.

## Fitur
1. **Model prediksi** (Python, scikit-learn): membandingkan Logistic Regression, Random Forest, dan Gradient Boosting, lalu tuning model terbaik.
2. **Aplikasi web agen** (Flask):
   - Upload data pelanggan (CSV)
   - Tombol **Predict** per pelanggan, menampilkan Next Best Offer dan peluang tiap produk
   - Tombol **Hubungi Pelanggan** yang membuka WhatsApp dengan pesan penawaran otomatis

## Produk yang Diprediksi
Motor Baru, Mobil Bekas, Elektronik, Dana Tunai Agunan

## Hasil Model (data test)
| Metrik | Nilai |
|---|---|
| Accuracy | 0,65 |
| F1 macro | 0,59 |
| Top-2 accuracy | 0,87 |

Catatan: data sengaja diberi noise agar menyerupai perilaku nyata. Produk Elektronik paling sulit diprediksi karena proporsi datanya paling kecil (imbalanced data).

## Tech Stack
Python, pandas, scikit-learn, matplotlib, Flask, HTML/CSS

## Struktur Project
```
├── 01_buat_data_dummy.py   # membuat data pelanggan dummy
├── 02_latih_model.py       # training, evaluasi, dan simpan model
├── app.py                  # aplikasi Flask
├── templates/              # halaman HTML
├── static/style.css
├── data/                   # data dummy + contoh file upload
├── model/                  # model terlatih (.pkl)
└── output/                 # confusion matrix & feature importance
```

## Cara Menjalankan
```bash
pip install -r requirements.txt
python 01_buat_data_dummy.py
python 02_latih_model.py
python app.py
```
Buka http://127.0.0.1:5000 lalu upload `data/contoh_upload.csv`.

## Screenshot
(tambahkan gambar di folder `screenshots/` lalu tampilkan di sini)

## Pengembangan Berikutnya
- Login agen
- Prediksi massal dengan hasil yang bisa diunduh
- Penanganan imbalanced data (class weight / SMOTE)
- Integrasi WhatsApp Business API
