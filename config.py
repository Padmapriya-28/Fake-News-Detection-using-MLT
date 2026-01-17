"""
Configuration file for Fake News Detection System
Contains all project constants and settings
"""

import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# Model Parameters
MODELS = {
    'Naive Bayes': 'naive_bayes',
    'Logistic Regression': 'logistic_regression',
    'Support Vector Machine': 'svm'
}

# Text Processing Parameters
MAX_FEATURES = 5000  # Maximum number of features for TF-IDF vectorization
TEST_SIZE = 0.2  # Train-test split ratio
RANDOM_STATE = 42  # For reproducibility

# UI Configuration
PAGE_TITLE = "🔍 Fake News Detection System"
PAGE_ICON = "🔍"
LAYOUT = "wide"
