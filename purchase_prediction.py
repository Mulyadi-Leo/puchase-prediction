import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('model_rf.joblib')

st.title("Prediksi Customer Akan Melakukan Purchase")

st.markdown("Masukkan informasi interaksi customer:")

# Input

cart = st.number_input("Jumlah Cart", min_value=0, max_value=200, value=0)
view = st.number_input("Jumlah View", min_value=0, max_value=200, value=0)
day_type_weekday = st.number_input("Jumlah penggunaan di hari biasa (senin-jumat)", min_value=0, max_value=200, value=0)
day_type_weekend = st.number_input("Jumlah penggunaan di hari libur mingguan (sabtu-minggu)", min_value=0, max_value=200, value=0)
time_period_pagi = st.number_input("Interaksi di Pagi Hari", min_value=0, max_value=200, value=0)
time_period_siang = st.number_input("Interaksi di Siang Hari", min_value=0, max_value=200, value=0)
time_period_sore = st.number_input("Interaksi di Sore Hari", min_value=0, max_value=200, value=0)
time_period_malam = st.number_input("Interaksi di Malam Hari", min_value=0, max_value=200, value=0)

# # Pilihan untuk weekday atau weekend
# day_type = st.selectbox("Hari", ["weekday", "weekend"])

# # Konversi kategori ke numerik (misal weekday=0, weekend=1)
# day_type_val = 0 if day_type == "weekday" else 1

# Tombol prediksi
if st.button("Prediksi Purchase"):
    # Buat array input
    input_data = np.array([[cart, view, day_type_weekday, day_type_weekend, time_period_pagi, time_period_siang, time_period_sore, time_period_malam]])
    
    # Lakukan prediksi
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]  # Probabilitas purchase

    # Tampilkan hasil
    if prediction == 1:
        st.success(f"Customer **berpotensi melakukan purchase** (Probabilitas: {probability:.2%})")
    else:
        st.warning(f"Customer **tidak berpotensi purchase** (Probabilitas: {probability:.2%})")
