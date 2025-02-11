import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import pickle
import nltk
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import os
from PIL import Image

# Tambahkan CSS untuk menyelaraskan tombol
st.markdown("""
    <style>
    .stButton>button {
        height: 42px;
        margin-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title('Klasifikasi Jenis Pertanyaan Menggunakan Machine Learning')

# Layout untuk input dan tombol agar sejajar
col1, col2 = st.columns([5, 1])  # Lebih proporsional agar tombol sejajar

with col1:
    text = st.text_input("Masukkan Pertanyaan:", key="input1", label_visibility="collapsed")

with col2:
    submit = st.button("Enter")  # Tombol lebih sejajar dengan input

# Tabs
tab1, tab2, tab3 = st.tabs(["Prediksi", "Probabilitas Kelas", "Grafik Model"])

# Grafik langsung muncul di tab3
with tab3:
    st.subheader("Grafik Model")
    try:
        image = Image.open("Grafik.png")
        st.image(image, caption="Grafik Model", use_column_width=True)
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

# Jika tidak ada input, tampilkan pesan di tab2
if not text.strip():
    with tab2:
        st.write("Masukkan Pertanyaan Terlebih Dahulu!")

# Prediksi hanya dilakukan jika tombol ditekan
if submit and text.strip():
    try:
        # (Tambahkan kode preprocessing dan model prediksi di sini)
        st.success("Hasil Prediksi: Contoh Kelas")
    except Exception as e:
        st.error(f"Error: {str(e)}")
