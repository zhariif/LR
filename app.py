import streamlit as st
import joblib
import numpy as np
import os

# Fungsi untuk memuat model dan objek terkait
@st.cache_resource
def load_objects():
    # Pastikan file ada di direktori yang sama
    base_dir = os.path.dirname(__file__)
    
    model = joblib.load(os.path.join(base_dir, 'lr_best.pkl'))
    vectorizer = joblib.load(os.path.join(base_dir, 'vec_terpilih.pkl'))
    le = joblib.load(os.path.join(base_dir, 'label_encoder.pkl'))
        
    with open(os.path.join(base_dir, 'threshold.txt'), 'r') as f:
        threshold_text = f.read().strip().replace(',', '.')
        threshold = float(threshold_text)
        
    return model, vectorizer, le, threshold


# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="Klasifikasi Email",
    page_icon="📧",
    layout="centered"
)

# Header aplikasi
st.title("📧 Aplikasi Klasifikasi Email")
st.write("Aplikasi ini menggunakan model Logistic Regression untuk memprediksi apakah sebuah teks merupakan Spam atau bukan, berdasarkan model yang sudah dilatih.")
st.markdown("---")

try:
    # Memuat objek
    model, vectorizer, le, threshold = load_objects()
    
    # Input dari pengguna
    st.subheader("Masukkan Teks")
    user_input = st.text_area("Ketik atau paste teks email di sini:", height=150)
    
    if st.button("Prediksi", type="primary"):
        if user_input.strip() == "":
            st.warning("Silakan masukkan teks terlebih dahulu!")
        else:
            with st.spinner("Sedang memproses..."):
                # 1. Transformasi teks menggunakan vectorizer
                vec_text = vectorizer.transform([user_input])
                
                # 2. Prediksi probabilitas
                # Asumsi model.predict_proba menghasilkan probabilitas untuk kelas 0 dan 1
                # Kita ambil probabilitas untuk kelas 1 (misalnya Spam)
                proba = model.predict_proba(vec_text)[0, 1]
                
                # 3. Terapkan custom threshold
                pred_class = 1 if proba >= threshold else 0
                
                # 4. Decode label menggunakan label encoder
                pred_label = le.inverse_transform([pred_class])[0]
                
            # Menampilkan hasil
            st.subheader("Hasil Prediksi:")
            
            if pred_class == 1:
                st.error(f"**{pred_label}**")
            else:
                st.success(f"**{pred_label}**")
                
            # Menampilkan detail probabilitas
            with st.expander("Lihat Detail Probabilitas"):
                st.write(f"Probabilitas Kelas 1: **{proba:.4f}**")
                st.write(f"Ambang Batas (Threshold): **{threshold:.4f}**")
                if proba >= threshold:
                    st.info("Karena probabilitas >= threshold, maka diklasifikasikan sebagai Kelas 1.")
                else:
                    st.info("Karena probabilitas < threshold, maka diklasifikasikan sebagai Kelas 0.")
                    
except FileNotFoundError as e:
    st.error(f"File tidak ditemukan: {e}. Pastikan semua file `.pkl` dan `.txt` sudah ada di folder yang sama.")
except Exception as e:
    st.error(f"Terjadi kesalahan: {e}")
