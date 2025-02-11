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
from collections import defaultdict
import random

# Initialize NLTK data directory
nltk_data_dir = Path("./nltk_data")
nltk_data_dir.mkdir(exist_ok=True)
nltk.data.path.insert(0, str(nltk_data_dir))
nltk.data.path.append(str(nltk_data_dir))

@st.cache_resource
def initialize_nltk():
    """Initialize required NLTK resources"""
    resources = ['stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, download_dir=str(nltk_data_dir), quiet=True)

# Initialize NLTK resources
initialize_nltk()

# Load stopwords and lemmatizer
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', download_dir=str(nltk_data_dir), quiet=True)
    stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

@st.cache_resource
def load_model_files():
    """Load model and necessary files"""
    try:
        model = keras.models.load_model('sentimen_model.h5')
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open('label_encoder.pkl', 'rb') as handle:
            label_encoder = pickle.load(handle)
        with open('maxlen.pkl', 'rb') as handle:
            maxlen = pickle.load(handle)
        return model, tokenizer, label_encoder, maxlen
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None, None, None

def simple_tokenize(text):
    """Basic tokenization (remove URLs, split words)"""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    tokens = re.findall(r'\w+', text.lower())
    return tokens

def preprocessing_text(text):
    """Clean and preprocess input text"""
    try:
        text = text.lower()
        words = simple_tokenize(text)
        words = [word for word in words if word not in stop_words or word in ['not', 'no', "n't"]]
        words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
        return text

# Load model and related files
model_prediksi, tokenizer, label_encoder, maxlen = load_model_files()

# Load dataset and extract 50 questions per category
file_path = "dataset.txt"
selected_classes = {"loc", "num", "hum", "desc", "abbr", "enty"}
categories = defaultdict(list)
try:
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split(" ", 1)
            if len(parts) > 1:
                category, question = parts
                if category in selected_classes:
                    all_questions.append(question)

    # Shuffle questions randomly
    random.shuffle(all_questions)
    sampled_questions = all_questions[:50]  # Display 50 random questions

    # Prepare content for tab4
    tab4_content = "### Contoh-Contoh Pertanyaan dari Dataset\n\n"
    for q in sampled_questions:
        tab4_content += f"- {q}\n"
    tab4_content += "\n"

except Exception as e:
    tab4_content = f"Error loading dataset: {str(e)}"
# Streamlit UI
st.title('Klasifikasi Jenis Pertanyaan Menggunakan Machine Learning')

# Input text
text = st.text_input("Masukkan Pertanyaan:", key="input1")

# Tabs for different outputs
tab1, tab2, tab3, tab4 = st.tabs(['Prediksi', 'Probabilitas Kelas', 'Grafik Model', 'Contoh-Contoh Pertanyaan'])

# Langsung tampilkan grafik di tab3
with tab3:
    st.subheader("Grafik Model")
    try:
        image = Image.open("Grafik.png")
        st.image(image, caption="Grafik Model", use_container_width=True)
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

# Jika pengguna belum memasukkan teks
if not text.strip():
    with tab2:
        st.write("Masukkan Pertanyaan Terlebih Dahulu!")
else:
    try:
        # Preprocess text
        text_prepared = preprocessing_text(text)

        if text_prepared and tokenizer and model_prediksi and label_encoder:
            sequence_testing = tokenizer.texts_to_sequences([text_prepared])
            padded_testing = pad_sequences(sequence_testing, maxlen=maxlen, padding='post')

            with st.spinner('Melakukan prediksi...'):
                prediksi = model_prediksi.predict(padded_testing, verbose=0)
                predicted_class = np.argmax(prediksi, axis=1)[0]
                predicted_label = label_encoder.inverse_transform([predicted_class])[0]

            # Display prediction in tab1
            with tab1:
                st.success(f"Hasil Prediksi (Class): {predicted_label}")

            # Display class probabilities in tab2
            with tab2:
                st.subheader("Probabilitas Kelas:")
                classes = label_encoder.classes_
                predictions_with_classes = {cls: f"{prob * 100:.2f}%" for cls, prob in zip(classes, prediksi[0])}

                for cls, prob in predictions_with_classes.items():
                    st.write(f"{cls}: {prob}")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.info("Pastikan semua file model dan resources sudah tersedia.")

# Display sample questions in Tab 4
with tab4:
    st.subheader("Contoh-Contoh Pertanyaan dari Dataset")
    st.markdown(tab4_content)

