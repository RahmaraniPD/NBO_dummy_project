"""
UI AGEN (FLASK)
---------------------------
Langkah-langkah:
  1. Halaman utama  : agen upload file CSV data pelanggan
  2. Halaman daftar : tabel pelanggan, tiap baris ada tombol "Predict"
  3. Halaman predict: tampil Next Best Offer + persentase + tombol "Hubungi Pelanggan"
                      (tombol membuka WhatsApp dengan pesan penawaran otomatis)

Run : python app.py -> lalu buka http://127.0.0.1:5000
"""

import json
import os
from urllib.parse import quote

import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "my_nbo"   # dibutuhkan untuk flash message

FILE_UPLOAD = os.path.join("uploads", "data_upload.csv")

# ---------- Muat model hasil Langkah 1 (cukup sekali saat aplikasi start) ----------
model = joblib.load("model/model_nbo.pkl")
with open("model/info_model.json") as f:
    INFO = json.load(f)
FITUR = INFO["fitur"]          # daftar kolom yang dibutuhkan model
KELAS = list(model.classes_)   # daftar produk

# Kolom tambahan yang dibutuhkan UI
KOLOM_WAJIB_UI = ["customer_id", "nama", "no_whatsapp"]

# Template pesan WhatsApp per produk
PESAN = {
    "Motor Baru": "Ada penawaran spesial pembiayaan Motor Baru dengan cicilan ringan untuk kamu.",
    "Mobil Bekas": "Ada penawaran pembiayaan Mobil Bekas dengan proses cepat dan tenor fleksibel untuk kamu.",
    "Elektronik": "Ada penawaran pembiayaan Elektronik dengan cicilan 0% untuk kamu.",
    "Dana Tunai Agunan": "Ada penawaran Dana Tunai nih untuk kamu, cair cepat dan bunga kompetitif.",
}


def baca_data():
    """Baca file upload terakhir. Return None kalau belum ada."""
    if not os.path.exists(FILE_UPLOAD):
        return None
    return pd.read_csv(FILE_UPLOAD, dtype={"no_whatsapp": str})


# ---------- Halaman 1: upload ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file_csv")
        if not file or file.filename == "":
            flash("Pilih file CSV terlebih dahulu.", "error")
            return redirect(url_for("index"))
        if not file.filename.lower().endswith(".csv"):
            flash("Format file harus .csv", "error")
            return redirect(url_for("index"))

        try:
            df = pd.read_csv(file, dtype={"no_whatsapp": str})
        except Exception:
            flash("File tidak bisa dibaca. Pastikan formatnya CSV yang benar.", "error")
            return redirect(url_for("index"))

        # Cek semua kolom yang dibutuhkan ada
        kolom_hilang = [k for k in KOLOM_WAJIB_UI + FITUR if k not in df.columns]
        if kolom_hilang:
            flash("Kolom berikut tidak ditemukan: " + ", ".join(kolom_hilang), "error")
            return redirect(url_for("index"))

        df.to_csv(FILE_UPLOAD, index=False)
        flash(f"Upload berhasil: {len(df)} pelanggan.", "success")
        return redirect(url_for("daftar"))

    return render_template("index.html", kolom=KOLOM_WAJIB_UI + FITUR)


# ---------- Halaman 2: daftar pelanggan ----------
@app.route("/daftar")
def daftar():
    df = baca_data()
    if df is None:
        flash("Belum ada data. Silakan upload dulu.", "error")
        return redirect(url_for("index"))

    # Pagination sederhana: 20 baris per halaman
    per_halaman = 20
    halaman = request.args.get("page", 1, type=int)
    total_halaman = max(1, -(-len(df) // per_halaman))   # pembulatan ke atas
    halaman = min(max(1, halaman), total_halaman)
    awal = (halaman - 1) * per_halaman
    potongan = df.iloc[awal:awal + per_halaman]

    baris = [{"idx": idx, **row.to_dict()} for idx, row in potongan.iterrows()]
    return render_template("daftar.html", baris=baris, halaman=halaman,
                           total_halaman=total_halaman, total=len(df))


# ---------- Halaman 3: hasil predict ----------
@app.route("/predict/<int:idx>")
def predict(idx):
    df = baca_data()
    if df is None or idx not in df.index:
        flash("Data pelanggan tidak ditemukan.", "error")
        return redirect(url_for("index"))

    pelanggan = df.loc[idx]
    probabilitas = model.predict_proba(df.loc[[idx], FITUR])[0]

    # Urutkan produk dari peluang tertinggi
    hasil = sorted(zip(KELAS, probabilitas), key=lambda x: x[1], reverse=True)
    hasil = [{"produk": p, "persen": round(float(s) * 100, 1)} for p, s in hasil]
    nbo = hasil[0]["produk"]

    # Susun link WhatsApp (format: https://wa.me/<nomor>?text=<pesan>)
    pesan = f"Halo {pelanggan['nama']}, {PESAN[nbo]} Kalau tertarik balas pesan ini ya."
    link_wa = f"https://wa.me/{pelanggan['no_whatsapp']}?text={quote(pesan)}"

    return render_template("predict.html", p=pelanggan.to_dict(), hasil=hasil,
                           nbo=nbo, link_wa=link_wa, pesan=pesan)


if __name__ == "__main__":
    app.run(debug=True)
