# Arsip Klarifikasi Hoaks Komdigi

Aplikasi Flask untuk menelusuri dataset `komdigi_hoaks.csv`. Pengguna dapat mencari artikel berdasarkan judul, ringkasan, isi, tag, dan topik; memfilter kategori; melihat halaman detail; serta membuka artikel sumber.

Dataset ini hanya berisi kategori **Klarifikasi Hoaks**, sehingga aplikasi menyajikan arsip dan tidak mengklaim dapat mengklasifikasikan berita baru sebagai hoaks atau bukan hoaks.

## Menjalankan lokal

Jalankan perintah dari root repository:

```powershell
python -m pip install -r praktikum/pbl_dataset/jobsheet15/requirements.txt
python praktikum/pbl_dataset/jobsheet15/app.py
```

Secara default aplikasi mencari `data/komdigi_hoaks.csv` di root repository atau `komdigi_hoaks.csv` di samping `app.py`. Untuk lokasi lain, atur `KOMDIGI_CSV_PATH`.

Buka `http://127.0.0.1:5000`. Atur `PORT` untuk memilih port lain, atau `FLASK_HOST=0.0.0.0` agar server dapat diakses dari lingkungan container.
