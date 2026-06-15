# Email Classification App

Aplikasi web berbasis Streamlit untuk mendeteksi apakah sebuah email termasuk kategori Spam atau Ham (biasa), menggunakan model Logistic Regression.

## Fitur
- Preprocessing otomatis menggunakan TF-IDF Vectorizer
- Model klasifikasi menggunakan Logistic Regression
- Threshold custom diatur pada 0.2936
- Deploy mudah melalui Streamlit Community Cloud

## Struktur File
- `app.py`: Source code utama untuk aplikasi Streamlit
- `lr_best.pkl`: Model Logistic Regression yang sudah dilatih (Tugas 6/7)
- `vec_terpilih.pkl`: TF-IDF Vectorizer yang sudah di-fit (Tugas 4)
- `label_encoder.pkl`: Encoder untuk mengonversi label kelas (Tugas 4)
- `threshold.txt`: Nilai threshold untuk prediksi (0.2936)
- `requirements.txt`: Daftar pustaka Python yang dibutuhkan

## Cara Menjalankan Lokal
1. Pastikan Anda sudah menginstal Python dan pip.
2. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```

## Deploy ke Streamlit Cloud
1. Upload folder/repository ini ke GitHub.
2. Buka [Streamlit Community Cloud](https://share.streamlit.io).
3. Buat aplikasi baru dan arahkan ke repository GitHub Anda, pilih file `app.py` sebagai Main file.
4. Klik Deploy.
