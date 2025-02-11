#!/bin/bash
# Create nltk_data directory if it doesn't exist
mkdir -p nltk_data

# Download NLTK data to nltk_data directory
python3 -m nltk.downloader -d ./nltk_data punkt
python3 -m nltk.downloader -d ./nltk_data wordnet
python3 -m nltk.downloader -d ./nltk_data averaged_perceptron_tagger
