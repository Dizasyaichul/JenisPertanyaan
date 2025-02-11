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

# Initialize NLTK data directory
nltk_data_dir = Path("./nltk_data")
nltk_data_dir.mkdir(exist_ok=True)

# Add the nltk_data directory to NLTK's search path
nltk.data.path.insert(0, str(nltk_data_dir))

@st.cache_resource
def initialize_nltk():
    """Initialize all required NLTK resources"""
    resources = ['stopwords', 'wordnet']
    for resource in resources:
        try:
            if resource == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif resource == 'wordnet':
                nltk.data.find('corpora/wordnet')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, download_dir=str(nltk_data_dir), quiet=True)

# Initialize NLTK resources
initialize_nltk()

# Initialize stopwords
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', download_dir=str(nltk_data_dir), quiet=True)
    stop_words = set(stopwords.words('english'))

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load model and related files
@st.cache_resource
def load_model_files():
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
    """Simple tokenization function that splits on whitespace and punctuation"""
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Split on punctuation and whitespace
    tokens = re.findall(r'\w+', text.lower())
    return tokens

def preprocessing_text(text):
    """Clean and preprocess input text"""
    try:
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize using simple tokenization
        words = simple_tokenize(text)
        
        # Remove stopwords but keep negation words
        words = [word for word in words if word not in stop_words 
                or word in ['not', 'no', "n't"]]
        
        # Lemmatize
        words = [lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    except Exception as e:
        st.error(f"Error in text preprocessing: {str(e)}")
        return text

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
        
        if text_prepared:
            # Convert to sequence and pad
            sequence_testing = tokenizer.texts_to_sequences([text_prepared])
            padded_testing = pad_sequences(sequence_testing, maxlen=maxlen, padding='post')
            
            # Make prediction
            with st.spinner('Melakukan prediksi...'):
                prediksi = model_prediksi.predict(padded_testing, verbose=0)
                predicted_class = np.argmax(prediksi, axis=1)[0]
                predicted_label = label_encoder.inverse_transform([predicted_class])[0]
                
                # Show results
                st.success("Hasil Prediksi (Class): " + predicted_label)
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.info("Pastikan semua file model dan resources sudah tersedia.")
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.info("Pastikan semua file model dan resources sudah tersedia.")
