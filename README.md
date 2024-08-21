# Song Genre Classification - Rock vs Hip-Hop

## Overview

This project involves classifying songs into two genres—Rock and Hip-Hop—using machine learning techniques. The dataset used for this project contains track information derived from Echonest (now part of Spotify), and features such as danceability, energy, acousticness, and tempo are utilized to build classifiers.

## Project Details

**Objective**: Develop a machine learning model to classify songs as either Rock or Hip-Hop based on track information.

**Technologies Used**:
- Python
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib

## Tasks and Methodology

1. **Data Preprocessing**:
   - Subset the dataset into Rock and Hip-Hop tracks.
   - Balance the dataset by ensuring an equal number of samples for each genre.

2. **Feature Engineering**:
   - Extract features and labels from the dataset.
   - Normalize the features using StandardScaler.
   - Apply Principal Component Analysis (PCA) to reduce dimensionality.

3. **Model Training**:
   - Train a Decision Tree Classifier.
   - Train a Logistic Regression Classifier.

4. **Model Evaluation**:
   - Compare models using classification reports.
   - Use K-Fold Cross-Validation to assess the robustness of the models.

## Key Results

- **Decision Tree Classifier**:
  - Precision, Recall, and F1-Score metrics.
- **Logistic Regression Classifier**:
  - Precision, Recall, and F1-Score metrics.
  
  After balancing the dataset, both models showed improved performance, with Logistic Regression performing slightly better in classifying the less prevalent Hip-Hop genre.

## Code
- Data Loading and Preprocessing
- Feature Extraction and Normalization
- Model Training and Evaluation
- Cross-Validation

import joblib

# Train the Logistic Regression model
logreg_model = LogisticRegression(random_state=10)
logreg_model.fit(train_pca, train_labels)

# Save the trained model, scaler, and PCA
joblib.dump(logreg_model, 'logistic_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')


import requests
import numpy as np
import librosa

# Load the trained model, scaler, and PCA
logreg_model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

def download_audio(url):
    response = requests.get(url)
    with open('temp_audio.mp3', 'wb') as f:
        f.write(response.content)

def extract_features_from_audio_librosa(file_path):
    # Load audio file
    y, sr = librosa.load(file_path, sr=None)
    # Extract features (e.g., MFCCs)
    features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=6).mean(axis=1)
    return features

# URL of the public audio file (Hip-Hop or Rock)
audio_url = 'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3'
download_audio(audio_url)

# Extract features from the new audio file using librosa
audio_features = extract_features_from_audio_librosa('temp_audio.mp3')

# Preprocess the features
scaled_features = scaler.transform([audio_features])
pca_features = pca.transform(scaled_features)

# Predict the genre
predicted_genre = logreg_model.predict(pca_features)

print('The predicted genre is: {}'.format(predicted_genre[0]))


