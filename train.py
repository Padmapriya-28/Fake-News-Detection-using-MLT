"""
Main Training Script
Run this script to train all models from command line
"""

import os
import sys
from preprocessing import load_and_preprocess_data
from model_training import FakeNewsDetector
from evaluation import generate_all_visualizations
from dataset_generator import save_sample_data
from config import DATA_DIR, MODELS_DIR, ASSETS_DIR


def main():
    """
    Main training pipeline
    """
    print("="*70)
    print("FAKE NEWS DETECTION SYSTEM - MODEL TRAINING")
    print("="*70)
    
    # Step 1: Generate or load dataset
    print("\n[STEP 1/6] Loading Dataset...")
    data_path = os.path.join(DATA_DIR, 'fake_news_sample.csv')
    
    if not os.path.exists(data_path):
        print("   Dataset not found. Generating sample dataset...")
        data_path = save_sample_data()
    else:
        print(f"   Dataset found: {data_path}")
    
    # Step 2: Preprocess data
    print("\n[STEP 2/6] Preprocessing Data...")
    preprocessed_texts, labels, raw_texts, preprocessor = load_and_preprocess_data(data_path)
    print(f"   ✓ Preprocessed {len(preprocessed_texts)} texts")
    
    # Step 3: Vectorize texts
    print("\n[STEP 3/6] Vectorizing Texts...")
    X = preprocessor.fit_transform(preprocessed_texts)
    y = labels
    print(f"   ✓ Feature matrix shape: {X.shape}")
    
    # Step 4: Train models
    print("\n[STEP 4/6] Training Machine Learning Models...")
    detector = FakeNewsDetector()
    
    # Split data
    X_train, X_test, y_train, y_test = detector.split_data(X, y)
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples: {X_test.shape[0]}")
    
    # Train all models
    results = detector.train_all_models(X_train, y_train, X_test, y_test)
    
    # Step 5: Save models
    print("\n[STEP 5/6] Saving Models...")
    detector.save_models(preprocessor)
    
    # Step 6: Generate visualizations
    print("\n[STEP 6/6] Generating Visualizations...")
    generate_all_visualizations(results, ASSETS_DIR)
    
    # Summary
    print("\n" + "="*70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📊 Results Summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"   Accuracy:  {metrics['accuracy']*100:.2f}%")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    
    # Best model
    best_model, best_results = detector.get_best_model()
    print(f"\n🏆 Best Model: {best_model}")
    print(f"   F1-Score: {best_results['f1_score']:.4f}")
    
    print("\n✅ Models saved to:", MODELS_DIR)
    print("✅ Visualizations saved to:", ASSETS_DIR)
    
    print("\n🚀 Next Steps:")
    print("   1. Run 'streamlit run app.py' to start the web application")
    print("   2. Navigate to http://localhost:8501")
    print("   3. Start detecting fake news!")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPlease check the error message and try again.")
        sys.exit(1)
