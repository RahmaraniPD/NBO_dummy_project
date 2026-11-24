# Next Best Offer Assistant

Project portofolio data analyst: model **Next Best Offer (NBO)** untuk merekomendasikan produk pembiayaan berikutnya kepada customer, lengkap dengan aplikasi web sederhana untuk agen.

> Seluruh data adalah **data dummy** buatan sendiri, bukan data perusahaan.

## Latar Belakang
Untuk mempermudah agen dalam melakukan penawaran yang paling tepat untuk existing customer, dibangun model prediksi untuk memberi rekomendasi produk berikutnya, dan dibuat UI agar agen bisa langsung memakainya.

## Fitur
1. **Model prediksi** (Python, scikit-learn): membandingkan Logistic Regression, Random Forest, dan Gradient Boosting, lalu tuning model terbaik.
2. **Aplikasi web agen** (Flask):
   - Upload data pelanggan (CSV)
   - Tombol **Predict** per customer, menampilkan Next Best Offer dan peluang tiap produk
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
├── 01_buat_data_dummy.py   # membuat data dummy untuk training
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
<img width="1609" height="629" alt="image" src="https://github.com/user-attachments/assets/38c1789a-68c4-4d61-8f10-e6c9ea5c047f" />
<img width="1604" height="676" alt="image" src="https://github.com/user-attachments/assets/ac56105d-e935-45be-a4c4-5520a217ae17" />
<img width="1603" height="839" alt="image" src="https://github.com/user-attachments/assets/92baac53-db73-447c-9da9-184d733c3df7" />

## Pengembangan Berikutnya
- Prediksi massal dengan hasil yang bisa diunduh
- Penanganan imbalanced data (class weight / SMOTE)
- Integrasi WhatsApp Business API
