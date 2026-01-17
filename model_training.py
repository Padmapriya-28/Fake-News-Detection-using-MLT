"""
Model Training Module
Implements and trains multiple ML algorithms for fake news detection
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
from config import MODELS_DIR, TEST_SIZE, RANDOM_STATE
import warnings
warnings.filterwarnings('ignore')


class FakeNewsDetector:
    """
    Main class for training and evaluating fake news detection models
    """
    
    def __init__(self):
        """
        Initialize the detector with three ML models
        """
        self.models = {
            'Naive Bayes': MultinomialNB(alpha=1.0),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            'Support Vector Machine': SVC(kernel='linear', probability=True, random_state=RANDOM_STATE)
        }
        self.trained_models = {}
        self.results = {}
        
    def split_data(self, X, y, test_size=TEST_SIZE):
        """
        Split data into training and testing sets
        
        Args:
            X: Feature matrix
            y: Labels
            test_size: Proportion of test set
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    
    def train_model(self, model_name, X_train, y_train):
        """
        Train a specific model
        
        Args:
            model_name (str): Name of the model
            X_train: Training features
            y_train: Training labels
            
        Returns:
            model: Trained model
        """
        print(f"\nTraining {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        print(f"{model_name} training completed!")
        return model
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """
        Evaluate a trained model
        
        Args:
            model_name (str): Name of the model
            model: Trained model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
        
        self.results[model_name] = results
        
        # Print results
        print(f"\n{'='*50}")
        print(f"{model_name} - Evaluation Results")
        print(f"{'='*50}")
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        
        return results
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate all models
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            dict: Results for all models
        """
        print("\n" + "="*60)
        print("TRAINING FAKE NEWS DETECTION MODELS")
        print("="*60)
        
        for model_name in self.models.keys():
            # Train model
            model = self.train_model(model_name, X_train, y_train)
            
            # Evaluate model
            self.evaluate_model(model_name, model, X_test, y_test)
        
        print("\n" + "="*60)
        print("ALL MODELS TRAINED SUCCESSFULLY!")
        print("="*60)
        
        return self.results
    
    def predict(self, model_name, X, return_proba=False):
        """
        Make predictions using a trained model
        
        Args:
            model_name (str): Name of the model
            X: Feature matrix
            return_proba (bool): Whether to return probabilities
            
        Returns:
            array: Predictions or probabilities
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} has not been trained yet!")
        
        model = self.trained_models[model_name]
        
        if return_proba and hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            return model.predict(X)
    
    def save_models(self, preprocessor=None):
        """
        Save all trained models and preprocessor to disk
        
        Args:
            preprocessor: TextPreprocessor object to save
        """
        print("\nSaving models...")
        
        for model_name, model in self.trained_models.items():
            # Create safe filename
            filename = model_name.lower().replace(' ', '_') + '.pkl'
            filepath = os.path.join(MODELS_DIR, filename)
            
            # Save model
            joblib.dump(model, filepath)
            print(f"✓ {model_name} saved to {filepath}")
        
        # Save preprocessor
        if preprocessor is not None:
            preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
            joblib.dump(preprocessor, preprocessor_path)
            print(f"✓ Preprocessor saved to {preprocessor_path}")
        
        # Save results
        results_path = os.path.join(MODELS_DIR, 'results.pkl')
        joblib.dump(self.results, results_path)
        print(f"✓ Results saved to {results_path}")
        
        print("\nAll models saved successfully!")
    
    def load_models(self):
        """
        Load all trained models from disk
        
        Returns:
            tuple: (trained_models, preprocessor, results)
        """
        print("\nLoading models...")
        
        # Load each model
        for model_name in self.models.keys():
            filename = model_name.lower().replace(' ', '_') + '.pkl'
            filepath = os.path.join(MODELS_DIR, filename)
            
            if os.path.exists(filepath):
                self.trained_models[model_name] = joblib.load(filepath)
                print(f"✓ {model_name} loaded")
        
        # Load preprocessor
        preprocessor_path = os.path.join(MODELS_DIR, 'preprocessor.pkl')
        preprocessor = None
        if os.path.exists(preprocessor_path):
            preprocessor = joblib.load(preprocessor_path)
            print(f"✓ Preprocessor loaded")
        
        # Load results
        results_path = os.path.join(MODELS_DIR, 'results.pkl')
        if os.path.exists(results_path):
            self.results = joblib.load(results_path)
            print(f"✓ Results loaded")
        
        print("\nAll models loaded successfully!")
        
        return self.trained_models, preprocessor, self.results
    
    def get_best_model(self):
        """
        Get the best performing model based on F1-score
        
        Returns:
            tuple: (best_model_name, best_results)
        """
        if not self.results:
            raise ValueError("No results available. Train models first!")
        
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1_score'])
        best_results = self.results[best_model_name]
        
        return best_model_name, best_results


def create_comparison_dataframe(results):
    """
    Create a DataFrame comparing all models
    
    Args:
        results (dict): Results from all models
        
    Returns:
        DataFrame: Comparison table
    """
    comparison_data = []
    
    for model_name, metrics in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'Precision': f"{metrics['precision']:.4f}",
            'Recall': f"{metrics['recall']:.4f}",
            'F1-Score': f"{metrics['f1_score']:.4f}"
        })
    
    df = pd.DataFrame(comparison_data)
    return df


if __name__ == "__main__":
    print("Model Training Module")
    print("This module should be imported and used with preprocessed data")
