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

The code for this project is organized into the following key sections:

- Data Loading and Preprocessing
- Feature Extraction and Normalization
- Model Training and Evaluation
- Cross-Validation
