"""
Data Preprocessing Module
Handles text cleaning, tokenization, and vectorization for fake news detection
"""

import re
import string
import nltk
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class TextPreprocessor:
    """
    Class for preprocessing text data for fake news detection
    """
    
    def __init__(self, max_features=5000):
        """
        Initialize the preprocessor
        
        Args:
            max_features (int): Maximum number of features for TF-IDF vectorization
        """
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
    def clean_text(self, text):
        """
        Clean and preprocess a single text string
        
        Steps:
        1. Convert to lowercase
        2. Remove URLs
        3. Remove mentions (@username)
        4. Remove hashtags
        5. Remove special characters and digits
        6. Remove extra whitespaces
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text):
        """
        Remove stopwords from text
        
        Args:
            text (str): Cleaned text
            
        Returns:
            str: Text without stopwords
        """
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        return ' '.join(filtered_words)
    
    def stem_text(self, text):
        """
        Apply stemming to reduce words to their root form
        
        Args:
            text (str): Text without stopwords
            
        Returns:
            str: Stemmed text
        """
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    
    def preprocess_text(self, text):
        """
        Complete preprocessing pipeline for a single text
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Fully preprocessed text
        """
        # Step 1: Clean text
        text = self.clean_text(text)
        
        # Step 2: Remove stopwords
        text = self.remove_stopwords(text)
        
        # Step 3: Apply stemming
        text = self.stem_text(text)
        
        return text
    
    def fit_transform(self, texts):
        """
        Fit the vectorizer and transform texts to TF-IDF features
        
        Args:
            texts (list): List of preprocessed texts
            
        Returns:
            array: TF-IDF feature matrix
        """
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """
        Transform texts to TF-IDF features using fitted vectorizer
        
        Args:
            texts (list): List of preprocessed texts
            
        Returns:
            array: TF-IDF feature matrix
        """
        return self.vectorizer.transform(texts)


def load_and_preprocess_data(file_path):
    """
    Load data from CSV and preprocess it
    
    Expected CSV format:
    - 'text' or 'content' column: News article text
    - 'label' column: 0 for Real, 1 for Fake
    
    Args:
        file_path (str): Path to CSV file
        
    Returns:
        tuple: (preprocessed_texts, labels, raw_texts)
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Identify text column
    text_column = None
    for col in ['text', 'content', 'article', 'news']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        raise ValueError("No text column found in CSV. Expected columns: 'text', 'content', 'article', or 'news'")
    
    # Identify label column
    label_column = None
    for col in ['label', 'class', 'category', 'target']:
        if col in df.columns:
            label_column = col
            break
    
    if label_column is None:
        raise ValueError("No label column found in CSV. Expected columns: 'label', 'class', 'category', or 'target'")
    
    # Extract texts and labels
    raw_texts = df[text_column].values
    labels = df[label_column].values
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Preprocess all texts
    print("Preprocessing texts...")
    preprocessed_texts = [preprocessor.preprocess_text(text) for text in raw_texts]
    
    return preprocessed_texts, labels, raw_texts, preprocessor


if __name__ == "__main__":
    # Test preprocessing
    sample_text = "Breaking News: @RealNews reports SHOCKING discovery! #FakeNews https://example.com"
    
    preprocessor = TextPreprocessor()
    cleaned = preprocessor.preprocess_text(sample_text)
    
    print("Original:", sample_text)
    print("Preprocessed:", cleaned)
