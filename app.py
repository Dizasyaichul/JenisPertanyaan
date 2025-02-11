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
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import os

# Create nltk_data directory if it doesn't exist
nltk_data_dir = Path("./nltk_data")
nltk_data_dir.mkdir(exist_ok=True)

# Add the nltk_data directory to NLTK's search path
nltk.data.path.insert(0, str(nltk_data_dir))

def download_nltk_resources():
    """Download required NLTK resources"""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}' if resource == 'punkt' 
                          else f'corpora/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, download_dir=str(nltk_data_dir), quiet=True)
            
# Download resources at startup
download_nltk_resources()

# Make sure stopwords are available
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', download_dir=str(nltk_data_dir), quiet=True)
    stop_words = set(stopwords.words('english'))

# Load model and related files
@st.cache_resource
def load_model_files():
    model = keras.models.load_model('sentimen_model.h5')
    
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    try:
        with open('label_encoder.pkl', 'rb') as handle:
            label_encoder = pickle.load(handle)
    except Exception as e:
        st.error(f"Error loading label_encoder: {str(e)}")
        return None
        
    with open('maxlen.pkl', 'rb') as handle:
        maxlen = pickle.load(handle)
        
    return model, tokenizer, label_encoder, maxlen

# Preprocessing function
def preprocessing_text(text):
    """Clean and preprocess input text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\?!.,\'"]', '', text)
    
    # Tokenize
    words = word_tokenize(text)
    
    # Remove stopwords but keep negation words
    words = [word for word in words if word not in stop_words 
            or word in ['not', 'no', "n't"]]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Load model and related files
model_prediksi, tokenizer, label_encoder, maxlen = load_model_files()

# Streamlit UI
st.title('Klasifikasi Jenis Pertanyaan Menggunakan Machine Learning')

# Input text
text = st.text_input("Masukkan Pertanyaan:", key="input1")

# Process input when available
if text.strip():
    try:
        # Preprocess text
        text_prepared = preprocessing_text(text)
        
        # Convert to sequence and pad
        sequence_testing = tokenizer.texts_to_sequences([text_prepared])
        padded_testing = pad_sequences(sequence_testing, maxlen=maxlen, padding='post')
        
        # Make prediction
        prediksi = model_prediksi.predict(padded_testing, verbose=0)
        predicted_class = np.argmax(prediksi, axis=1)[0]
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
        
        # Show results
        st.write("Hasil Prediksi (Class):", predicted_label)
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
