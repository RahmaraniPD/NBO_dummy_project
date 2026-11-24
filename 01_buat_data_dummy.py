"""
MEMBUAT DATA DUMMY PELANGGAN
-----------------------------------------
Script ini akan membuat data pelanggan fiktif (3.000 baris) yang dirasa sesuai dengan case perusahaan pembiayaan, 
agar model memiliki data training yang representatif.

Run : python 01_buat_data_dummy.py
Hasil : data/data_pelanggan.csv
"""

import numpy as np
import pandas as pd
import random
import string

rng = np.random.default_rng(42)
JUMLAH = 3000

# ---------- 1. Membuat fitur pelanggan ----------
def random_nama():
    panjang = random.randint(5, 7)
    return ''.join(random.choices(string.ascii_lowercase, k=panjang)).capitalize()

df = pd.DataFrame({
    "customer_id": [f"CUST{str(i).zfill(5)}" for i in range(1, JUMLAH + 1)],
    "nama": [random_nama() for _ in range(JUMLAH)],
    "no_whatsapp": [f"62812{rng.integers(10000000, 99999999)}"[:-3] + "xxx" for _ in range(JUMLAH)],
    "usia": rng.integers(21, 60, JUMLAH),
    "jenis_kelamin": rng.choice(["Laki-laki", "Perempuan"], JUMLAH),
    "kota": rng.choice(["Jakarta", "Bandung", "Surabaya", "Medan", "Semarang", "Makassar"],
                       JUMLAH, p=[0.3, 0.15, 0.2, 0.1, 0.15, 0.1]),
    "pekerjaan": rng.choice(["Karyawan", "Wiraswasta", "Profesional", "Ibu Rumah Tangga"],
                            JUMLAH, p=[0.45, 0.30, 0.15, 0.10]),
    "penghasilan_bulanan_juta": np.round(rng.lognormal(mean=2.5, sigma=0.5, size=JUMLAH), 1),
    "produk_terakhir": rng.choice(["Motor Baru", "Mobil Bekas", "Elektronik", "Dana Tunai Agunan"],
                                  JUMLAH, p=[0.4, 0.2, 0.25, 0.15]),
    "jumlah_produk_dimiliki": rng.integers(1, 5, JUMLAH),
    "lama_jadi_customer_bulan": rng.integers(3, 96, JUMLAH),
    "riwayat_telat_bayar": rng.poisson(0.8, JUMLAH),
    "buka_app_30hari": rng.poisson(6, JUMLAH),
    "sisa_cicilan_bulan": rng.integers(0, 36, JUMLAH),
})
df["rasio_cicilan_penghasilan"] = np.round(rng.uniform(0.1, 0.6, JUMLAH), 2)

# ---------- 2. Menentukan produk berikutnya yang dibeli (TARGET) ----------
# Kita buat "skor" tiap produk berdasarkan aturan bisnis sederhana + noise acak.
# Model nanti akan mencoba MENEBAK aturan tersembunyi ini dari datanya.
skor = pd.DataFrame(index=df.index)

skor["Motor Baru"] = (
    1.0
    + (df["usia"] < 35) * 1.2
    + (df["produk_terakhir"] == "Elektronik") * 0.8
    + (df["penghasilan_bulanan_juta"] < 12) * 0.8
)
skor["Mobil Bekas"] = (
    0.5
    + (df["penghasilan_bulanan_juta"] > 15) * 1.8
    + (df["produk_terakhir"] == "Motor Baru") * 1.0
    + (df["jumlah_produk_dimiliki"] >= 3) * 0.7
)
skor["Elektronik"] = (
    0.8
    + (df["sisa_cicilan_bulan"] < 6) * 1.2
    + (df["buka_app_30hari"] > 8) * 1.0
    + (df["usia"] < 30) * 0.6
)
skor["Dana Tunai Agunan"] = (
    0.4
    + (df["lama_jadi_customer_bulan"] > 36) * 1.5
    + (df["pekerjaan"] == "Wiraswasta") * 1.3
    + (df["produk_terakhir"] == "Mobil Bekas") * 1.0
    - (df["riwayat_telat_bayar"] >= 3) * 1.0
)

# Tambah noise agar tidak 100% pasti
skor = skor + rng.normal(0, 0.9, skor.shape)

# Produk dengan skor tertinggi = produk berikutnya yang dibeli
df["produk_berikutnya"] = skor.idxmax(axis=1)

# ---------- 3. Simpan ----------
df.to_csv("data/data_pelanggan.csv", index=False)
print("Data dummy berhasil dibuat:", df.shape)
print("\nDistribusi target (produk_berikutnya):")
print(df["produk_berikutnya"].value_counts())
print("\n5 baris pertama:")
print(df.head())
