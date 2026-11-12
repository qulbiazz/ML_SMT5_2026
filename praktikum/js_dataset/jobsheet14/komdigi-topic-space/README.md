---
title: komdigi-topic-classifier-244107020050
emoji: 📰
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
license: mit
---

# Klasifikasi Topik Artikel Komdigi

Space Gradio untuk memprediksi topik artikel menggunakan judul dan isi artikel.
Model dilatih pada dataset klarifikasi Komdigi dengan enam label: Hoaks,
Kesehatan, Masyarakat Digital, Pejabat Publik, Penipuan, dan Program/Kebijakan.
Prediksi ini mengklasifikasikan topik; model bukan alat untuk memverifikasi
kebenaran sebuah artikel.

## Berkas model

- `komdigi_topic_model.h5`: model JST terlatih.
- `komdigi_preprocessor.pkl`: TF-IDF vectorizer dan label encoder.

Tidak ada `scaler.pkl` karena pipeline teks ini tidak memakai scaler numerik.
Vectorizer TF-IDF harus digunakan kembali persis seperti saat model dilatih.

## Melatih ulang dan mengekspor model

Dari folder ini, pastikan dataset tersedia di `../../../../data/komdigi_hoaks.csv`,
lalu jalankan:

```bash
pip install -r requirements.txt
python train_and_export.py
```

Skrip menghasilkan file model `.h5` dan preprocessor `.pkl` yang dipakai aplikasi.

## Membuat dan mengunggah ke Hugging Face Spaces

1. Buat Space baru dengan nama `komdigi-topic-classifier-244107020050`,
   SDK **Gradio**, lisensi **MIT**, dan visibilitas **Public**.
2. Clone repositori Space yang dibuat:

   ```bash
   git clone https://huggingface.co/spaces/USERNAME/komdigi-topic-classifier-244107020050
   ```

3. Salin seluruh isi folder ini ke folder hasil clone.
4. Commit dan push:

   ```bash
   git add app.py README.md requirements.txt train_and_export.py komdigi_topic_model.h5 komdigi_preprocessor.pkl
   git commit -m "Deploy Komdigi topic classifier with Gradio"
   git push
   ```

Saat autentikasi Git meminta kredensial, gunakan Hugging Face username dan
Access Token dengan izin write. Jangan masukkan token ke source code, README,
atau commit.
