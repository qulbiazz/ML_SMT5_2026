# Klasifikasi Berita Hoaks dan Arsip Klarifikasi

Aplikasi Flask untuk mengklasifikasikan teks berita sebagai **hoaks** atau **valid**, serta menelusuri dataset klarifikasi `komdigi_hoaks.csv`. Pengguna dapat mencari artikel berdasarkan judul, ringkasan, isi, tag, dan topik; memfilter kategori; melihat halaman detail; serta membuka artikel sumber.

Klasifikasi memakai fitur kata biner dan Bernoulli Naive Bayes yang dilatih saat aplikasi dijalankan dari `data/datasetUMPOHoax.csv`. Hanya baris dengan teks dan label `hoax` atau `valid` yang digunakan; baris tanpa label yang jelas diabaikan. Evaluasi menggunakan pemisahan data latih/uji terstratifikasi 80:20 dengan seed tetap. Hasil hanya memakai label **Hoaks** atau **Valid**. Prediksi hoaks hanya diterima jika keyakinan minimal 60% dan cakupan fitur teks minimal 35%; jika tidak, keluaran biner menjadi **Valid**. Label valid dengan aturan ini bukan verifikasi fakta. Karena dataset pelatihan berisi teks lama dari media sosial, hasil model merupakan prediksi awal dan bukan jaminan akurasi untuk berita baru.

Notebook K-Means di jobsheet ini hanya mengelompokkan topik artikel dalam arsip klarifikasi hoaks Komdigi; nomor cluster tidak menunjukkan apakah berita baru benar atau hoaks.

## Menjalankan lokal

Jalankan perintah dari root repository:

```powershell
python -m pip install -r praktikum/pbl_dataset/jobsheet15/requirements.txt
python praktikum/pbl_dataset/jobsheet15/app.py
```

Secara default aplikasi mencari `data/komdigi_hoaks.csv` dan `data/datasetUMPOHoax.csv` di root repository, atau file dengan nama yang sama di samping `app.py`. Atur `KOMDIGI_CSV_PATH` atau `HOAX_TRAINING_CSV_PATH` untuk memakai lokasi dataset lain.

Buka `http://127.0.0.1:5000/klasifikasi` untuk menguji model dan `http://127.0.0.1:5000` untuk membuka arsip klarifikasi. Atur `PORT` untuk memilih port lain, atau `FLASK_HOST=0.0.0.0` agar server dapat diakses dari lingkungan container.
