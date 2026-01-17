"""
Fake News Detection System - Enterprise-Grade Web Application
Advanced AI-powered interface for detecting fake news using Machine Learning
Author: AI & Data Science Student
Version: 2.0 (Enterprise Edition)
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import warnings
from preprocessing import TextPreprocessor, load_and_preprocess_data
from model_training import FakeNewsDetector, create_comparison_dataframe
from evaluation import ModelEvaluator, generate_all_visualizations
from dataset_generator import save_sample_data, get_sample_texts_for_testing
from config import DATA_DIR, MODELS_DIR, ASSETS_DIR
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import time
from sklearn.metrics import roc_curve, auc, roc_auc_score
import hashlib

warnings.filterwarnings('ignore')

# Page configuration - Advanced
st.set_page_config(
    page_title="Fake News Detection - Enterprise Edition",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### Fake News Detection System v2.0\nEnterprise ML Solution for News Authenticity Analysis"
    }
)

# Advanced Enterprise CSS with Professional Dark Theme
st.markdown("""
<style>
    /* Main background - Professional gradient */
    .stApp {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #1a3a52 100%);
        color: #e0e0e0;
    }
    
    /* Optimize sidebar width */
    [data-testid="stSidebar"] {
        width: 280px !important;
        min-width: 280px !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        width: 280px !important;
    }
    
    /* Optimize main content area */
    .main .block-container {
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Adjust layout for proper sidebar fit */
    .stContainer {
        max-width: 100% !important;
    }
    
    /* Enterprise header styling */
    .main-header {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 15px 35px rgba(0, 212, 255, 0.3);
        margin-bottom: 40px;
        border: none;
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        font-size: 3.5em;
        margin: 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.4);
        font-weight: 900;
        letter-spacing: 1px;
    }
    
    .main-header p {
        font-size: 1.3em;
        margin-top: 15px;
        opacity: 0.98;
        font-weight: 500;
    }
    
    /* Premium card styling */
    .content-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.15);
        margin: 20px 0;
        border: 1px solid rgba(0, 212, 255, 0.3);
        backdrop-filter: blur(10px);
        color: #e0e0e0;
    }
    
    /* Advanced result boxes with animations */
    .result-box-fake {
        background: linear-gradient(135deg, #ff4757 0%, #d63031 100%);
        color: white;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        font-size: 2em;
        font-weight: 900;
        box-shadow: 0 15px 40px rgba(255, 71, 87, 0.5);
        margin: 30px 0;
        border: 2px solid #ff6348;
        animation: pulse-danger 2s infinite;
    }
    
    .result-box-real {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        font-size: 2em;
        font-weight: 900;
        box-shadow: 0 15px 40px rgba(46, 204, 113, 0.5);
        margin: 30px 0;
        border: 2px solid #16a085;
        animation: pulse-success 2s infinite;
    }
    
    @keyframes pulse-danger {
        0%, 100% { transform: scale(1); box-shadow: 0 15px 40px rgba(255, 71, 87, 0.5); }
        50% { transform: scale(1.02); box-shadow: 0 20px 50px rgba(255, 71, 87, 0.7); }
    }
    
    @keyframes pulse-success {
        0%, 100% { transform: scale(1); box-shadow: 0 15px 40px rgba(46, 204, 113, 0.5); }
        50% { transform: scale(1.02); box-shadow: 0 20px 50px rgba(46, 204, 113, 0.7); }
    }
    
    /* Metric boxes - Professional */
    .metric-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 153, 204, 0.1) 100%);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.2);
        margin: 15px 0;
        border: 1px solid rgba(0, 212, 255, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .metric-value {
        font-size: 3em;
        font-weight: 900;
        color: #00d4ff;
        text-shadow: 0 2px 10px rgba(0, 212, 255, 0.3);
    }
    
    .metric-label {
        font-size: 1.2em;
        color: #a0a0a0;
        margin-top: 8px;
        font-weight: 600;
    }
    
    /* Button styling - Premium */
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        color: white;
        border: none;
        padding: 15px 40px;
        font-size: 1.1em;
        font-weight: 700;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
        transition: all 0.4s ease;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.6);
    }
    
    /* Info/Warning boxes */
    .info-box {
        background: rgba(51, 154, 240, 0.1);
        border-left: 5px solid #339af0;
        padding: 18px;
        border-radius: 12px;
        margin: 15px 0;
        backdrop-filter: blur(10px);
    }
    
    .warning-box {
        background: rgba(255, 193, 7, 0.1);
        border-left: 5px solid #ffc107;
        padding: 18px;
        border-radius: 12px;
        margin: 15px 0;
        backdrop-filter: blur(10px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(0, 212, 255, 0.1);
        border-radius: 12px 12px 0 0;
        padding: 18px 30px;
        font-weight: 700;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    /* Text Colors - Enhanced Visibility */
    h1 {
        color: #00d4ff !important;
        text-shadow: 0 2px 10px rgba(0, 212, 255, 0.2);
    }
    
    h2 {
        color: #00d4ff !important;
        text-shadow: 0 2px 8px rgba(0, 212, 255, 0.2);
    }
    
    h3 {
        color: #00d4ff !important;
        text-shadow: 0 1px 6px rgba(0, 212, 255, 0.15);
    }
    
    h4, h5, h6 {
        color: #0099cc !important;
    }
    
    /* Label and text colors */
    label {
        color: #e0e0e0 !important;
        font-weight: 600;
    }
    
    .stMarkdown {
        color: #e0e0e0 !important;
    }
    
    p {
        color: #d0d0d0 !important;
    }
    
    /* Sidebar text colors */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #16213e 100%);
    }
    
    [data-testid="stSidebar"] h3 {
        color: #00d4ff !important;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #e0e0e0 !important;
    }
    
    /* Input field text colors */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        color: #e0e0e0 !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Metric text colors */
    .stMetric {
        color: #e0e0e0 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #00d4ff !important;
        font-size: 2em !important;
    }
    
    .stMetric [data-testid="stMetricLabel"] {
        color: #a0a0a0 !important;
    }
    
    /* Alert and info text */
    .stAlert {
        background: rgba(0, 212, 255, 0.1) !important;
        color: #e0e0e0 !important;
    }
    
    /* DataFrame text colors */
    .stDataFrame {
        color: #e0e0e0 !important;
    }
    
    /* Code and pre-formatted text */
    code {
        color: #00ff88 !important;
        background: rgba(0, 255, 136, 0.1) !important;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    pre {
        color: #00ff88 !important;
        background: rgba(0, 30, 60, 0.6) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
    }
    
    /* Radio and checkbox labels */
    .stRadio label, .stCheckbox label {
        color: #e0e0e0 !important;
    }
    
    /* Select and multiselect text */
    .stSelectbox label, .stMultiSelect label {
        color: #e0e0e0 !important;
    }
    
    /* Divider colors */
    hr {
        border-color: rgba(0, 212, 255, 0.3) !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 30px;
        color: #a0a0a0;
        margin-top: 60px;
        background: rgba(0, 212, 255, 0.08);
        border-radius: 15px;
        border-top: 2px solid rgba(0, 212, 255, 0.3);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 212, 255, 0.1);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def load_models():
    """Load trained models and preprocessor"""
    detector = FakeNewsDetector()
    
    try:
        trained_models, preprocessor, results = detector.load_models()
        return detector, preprocessor, results, True
    except Exception as e:
        return None, None, None, False


def train_models_pipeline():
    """Train all models from scratch"""
    
    with st.spinner("🚀 Initializing training pipeline..."):
        # Check if sample data exists, if not create it
        data_path = os.path.join(DATA_DIR, 'fake_news_sample.csv')
        
        if not os.path.exists(data_path):
            st.info("📊 Generating sample dataset...")
            data_path = save_sample_data()
        
        # Load and preprocess data
        st.info("🔄 Loading and preprocessing data...")
        preprocessed_texts, labels, raw_texts, preprocessor = load_and_preprocess_data(data_path)
        
        # Vectorize texts
        st.info("🔢 Vectorizing texts with TF-IDF...")
        X = preprocessor.fit_transform(preprocessed_texts)
        y = labels
        
        # Initialize detector
        detector = FakeNewsDetector()
        
        # Split data
        st.info("✂️ Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = detector.split_data(X, y)
        
        # Train all models
        st.info("🤖 Training machine learning models...")
        progress_bar = st.progress(0)
        
        models_list = list(detector.models.keys())
        for i, model_name in enumerate(models_list):
            st.write(f"   Training {model_name}...")
            model = detector.train_model(model_name, X_train, y_train)
            detector.evaluate_model(model_name, model, X_test, y_test)
            progress_bar.progress((i + 1) / len(models_list))
        
        # Save models
        st.info("💾 Saving trained models...")
        detector.save_models(preprocessor)
        
        # Generate visualizations
        st.info("📊 Generating evaluation visualizations...")
        generate_all_visualizations(detector.results, ASSETS_DIR)
        
        st.success("✅ Training completed successfully!")
        
        return detector, preprocessor, detector.results


def predict_news(text, detector, preprocessor, model_name):
    """Predict if news is fake or real with advanced confidence analysis"""
    
    # Preprocess text with analysis
    preprocessed_text = preprocessor.preprocess_text(text)
    
    # Text statistics
    original_words = len(text.split())
    preprocessed_words = len(preprocessed_text.split())
    
    # Vectorize
    X = preprocessor.transform([preprocessed_text])
    
    # Predict with probabilities
    prediction = detector.predict(model_name, X)[0]
    probabilities = detector.predict(model_name, X, return_proba=True)[0]
    
    # Get result
    label = "Fake" if prediction == 1 else "Real"
    confidence = probabilities[prediction] * 100
    
    # Calculate confidence level
    confidence_level = "Very High" if confidence >= 90 else "High" if confidence >= 75 else "Medium" if confidence >= 60 else "Low"
    
    return label, confidence, probabilities, confidence_level, {
        'original_words': original_words,
        'preprocessed_words': preprocessed_words,
        'preprocessing_ratio': f"{((1 - preprocessed_words/original_words) * 100):.1f}%"
    }


def display_header():
    """Display enterprise-grade header"""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 FAKE NEWS DETECTION SYSTEM</h1>
        <p>Enterprise-Grade AI-Powered News Authenticity Analyzer</p>
        <p style="font-size: 0.95em; margin-top: 10px; opacity: 0.9;">
            Advanced ML Classification | Real-time Analysis | Production-Ready
        </p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application function"""
    
    # Display header
    display_header()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        
        page = st.radio(
            "Select Page:",
            ["🏠 Home - Detect News", "📊 Model Performance", "🔬 Train Models"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.markdown("### ⚙️ Settings")
        
        model_choice = st.selectbox(
            "Choose ML Model:",
            ["Naive Bayes", "Logistic Regression", "Support Vector Machine"],
            label_visibility="collapsed"
        )
    
    # Load models
    detector, preprocessor, results, models_loaded = load_models()
    
    # Main content based on page selection
    if page == "🏠 Home - Detect News":
        home_page(detector, preprocessor, model_choice, models_loaded)
    
    elif page == "📊 Model Performance":
        performance_page(detector, results, models_loaded)
    
    elif page == "🔬 Train Models":
        train_page()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2026 Fake News Detection System | Developed for Academic Excellence</p>
        <p style="font-size: 0.9em; margin-top: 5px;">
            Powered by Machine Learning • Built with ❤️ using Python & Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)


def home_page(detector, preprocessor, model_choice, models_loaded):
    """Home page for news detection"""
    
    if not models_loaded:
        st.warning("⚠️ Models not found! Please train the models first.")
        st.info("👉 Go to '🔬 Train Models' page to train the models.")
        
        # Show example of what the page will look like
        st.markdown("### 📝 How It Works (Demo Preview)")
        st.text_area("Enter news text here:", "Your news text will be analyzed here...", height=150, disabled=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.button("🔍 Analyze News", disabled=True)
        
        return
    
    # Main detection interface
    st.markdown("### 📰 Enter News Text for Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        
        # Text input methods
        input_method = st.radio(
            "Choose input method:",
            ["✍️ Type Text", "📝 Use Sample"],
            horizontal=True
        )
        
        if input_method == "✍️ Type Text":
            news_text = st.text_area(
                "Paste or type news article:",
                height=200,
                placeholder="Enter the news article text you want to verify..."
            )
        else:
            # Sample texts
            samples = get_sample_texts_for_testing()
            sample_options = [f"{s['category']}: {s['text'][:50]}..." for s in samples]
            selected_sample = st.selectbox("Select a sample:", sample_options)
            sample_index = sample_options.index(selected_sample)
            news_text = samples[sample_index]['text']
            
            st.text_area("Selected text:", news_text, height=150, disabled=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Analyze button
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            analyze_button = st.button("🔍 Analyze News", use_container_width=True, type="primary")
    
    with col2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("### ℹ️ Information")
        
        st.markdown("""
        <div class="info-box">
        <strong>How to use:</strong>
        <ol>
            <li>Enter or select news text</li>
            <li>Choose ML model (sidebar)</li>
            <li>Click Analyze button</li>
            <li>View prediction result</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        **Current Model:**  
        🤖 {model_choice}
        """)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis results
    if analyze_button and news_text:
        with st.spinner("⚙️ Analyzing with advanced ML models..."):
            time.sleep(0.5)  # Processing indicator
            
            # Predict
            label, confidence, probabilities, confidence_level, text_stats = predict_news(
                news_text, detector, preprocessor, model_choice
            )
            
            # Display result with advanced styling
            st.markdown("## 🎯 ANALYSIS RESULT")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if label == "Fake":
                    st.markdown(f"""
                    <div class="result-box-fake">
                        ⚠️ FAKE NEWS DETECTED ⚠️<br>
                        <span style="font-size: 0.6em; opacity: 0.9;">Confidence: {confidence:.1f}% ({confidence_level})</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box-real">
                        ✅ LIKELY AUTHENTIC ✅<br>
                        <span style="font-size: 0.6em; opacity: 0.9;">Confidence: {confidence:.1f}% ({confidence_level})</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Advanced Metrics Section
            st.markdown("### 📊 ADVANCED ANALYSIS METRICS")
            
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{probabilities[0]*100:.1f}%</div>
                    <div class="metric-label">Real News<br>Probability</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_col2:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{probabilities[1]*100:.1f}%</div>
                    <div class="metric-label">Fake News<br>Probability</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_col3:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{confidence:.1f}%</div>
                    <div class="metric-label">Model<br>Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metrics_col4:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-value">{model_choice.split()[0]}</div>
                    <div class="metric-label">Algorithm<br>Used</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence meter with gauge
            st.markdown("### 🎯 CONFIDENCE GAUGE")
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=confidence,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "<b>PREDICTION CONFIDENCE</b>", 'font': {'size': 24, 'color': '#00d4ff'}},
                delta={'reference': 50, 'increasing': {'color': "#2ecc71" if label == "Real" else "#ff4757"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#00d4ff"},
                    'bar': {'color': "#00d4ff"},
                    'bgcolor': "rgba(0, 212, 255, 0.1)",
                    'borderwidth': 3,
                    'bordercolor': "#00d4ff",
                    'steps': [
                        {'range': [0, 33], 'color': 'rgba(255, 71, 87, 0.2)'},
                        {'range': [33, 66], 'color': 'rgba(255, 193, 7, 0.2)'},
                        {'range': [66, 100], 'color': 'rgba(46, 204, 113, 0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': "#00d4ff", 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ))
            
            fig_gauge.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=80, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "#00d4ff", 'family': "Arial", 'size': 12},
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Text Analysis Statistics
            st.markdown("### 📈 TEXT ANALYSIS STATISTICS")
            
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            
            with stats_col1:
                st.info(f"**Original Words**: {text_stats['original_words']}")
            
            with stats_col2:
                st.info(f"**After Preprocessing**: {text_stats['preprocessed_words']}")
            
            with stats_col3:
                st.info(f"**Removed**: {text_stats['preprocessing_ratio']}")
            
            # Decision Logic
            st.markdown("### 🔬 PREDICTION ANALYSIS")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if label == "Fake":
                    st.markdown(f"""
                    <div class="warning-box">
                    <strong>⚠️ Alert Level: {'CRITICAL' if confidence >= 85 else 'HIGH' if confidence >= 70 else 'MEDIUM'}</strong><br>
                    This content exhibits characteristics typical of misinformation. 
                    The {model_choice} algorithm detected patterns commonly associated with 
                    fabricated or misleading content with {confidence:.1f}% certainty.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="info-box">
                    <strong>✅ Authenticity: {'VERY HIGH' if confidence >= 85 else 'HIGH' if confidence >= 70 else 'MODERATE'}</strong><br>
                    This content appears to be genuine. The {model_choice} algorithm 
                    did not detect typical misinformation patterns with {confidence:.1f}% confidence.
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="content-card">
                <strong>🤖 Model: {model_choice}</strong><br>
                Probability Distribution:<br>
                • Real: {probabilities[0]*100:.2f}%<br>
                • Fake: {probabilities[1]*100:.2f}%<br><br>
                <strong>Confidence Level: {confidence_level}</strong>
                </div>
                """, unsafe_allow_html=True)


def performance_page(detector, results, models_loaded):
    """Advanced model performance comparison page"""
    
    st.markdown("## 📊 ADVANCED PERFORMANCE ANALYTICS")
    
    if not models_loaded or results is None:
        st.warning("⚠️ No model results available. Please train the models first.")
        return
    
    # Summary table with advanced metrics
    st.markdown("### 📈 COMPREHENSIVE METRICS COMPARISON")
    
    evaluator = ModelEvaluator(results)
    summary_df = evaluator.create_summary_table()
    
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Advanced tabs for detailed analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Metrics", "🎯 Radar", "🔲 Confusion Matrix", "📉 Performance", "📋 Report"])
    
    with tab1:
        st.markdown("#### Model Metrics Comparison")
        metrics_path = os.path.join(ASSETS_DIR, 'metrics_comparison.png')
        if os.path.exists(metrics_path):
            st.image(metrics_path, use_container_width=True)
        else:
            fig = evaluator.plot_metrics_comparison()
            st.pyplot(fig)
    
    with tab2:
        st.markdown("#### Multi-Dimensional Performance Radar")
        radar_fig = evaluator.plot_model_comparison_radar()
        st.plotly_chart(radar_fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### Confusion Matrix Analysis")
        cols = st.columns(3)
        model_names = list(results.keys())
        
        for i, model_name in enumerate(model_names):
            with cols[i]:
                st.markdown(f"**{model_name}**")
                fig = evaluator.plot_confusion_matrix(model_name)
                st.pyplot(fig)
                plt.close(fig)
    
    with tab4:
        st.markdown("#### Detailed Performance Metrics")
        
        selected_model = st.selectbox(
            "Select Model for Detailed Analysis:",
            list(results.keys())
        )
        
        if selected_model:
            metrics = results[selected_model]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%", 
                         delta=f"{metrics['accuracy']*100 - 50:.1f}%")
            
            with col2:
                st.metric("Precision", f"{metrics['precision']:.4f}")
            
            with col3:
                st.metric("Recall", f"{metrics['recall']:.4f}")
            
            with col4:
                st.metric("F1-Score", f"{metrics['f1_score']:.4f}")

            # Performance line chart across models
            st.markdown("#### Performance Line Chart")
            line_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
            line_df_long = line_df.melt(
                id_vars=['Model'],
                value_vars=['accuracy', 'precision', 'recall', 'f1_score'],
                var_name='Metric',
                value_name='Score'
            )
            line_df_long['Score'] = line_df_long['Score'] * 100  # convert to percentage for readability
            line_df_long['Metric'] = line_df_long['Metric'].map({
                'accuracy': 'Accuracy',
                'precision': 'Precision',
                'recall': 'Recall',
                'f1_score': 'F1-Score'
            })
            line_fig = px.line(
                line_df_long,
                x='Model',
                y='Score',
                color='Metric',
                markers=True,
                title='Model Performance Over Key Metrics (%)'
            )
            line_fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend_title_text='Metric',
                font={'color': '#e0e0e0'},
                yaxis_title='Score (%)'
            )
            st.plotly_chart(line_fig, use_container_width=True)
    
    with tab5:
        st.markdown("#### Model Comparison Report")
        
        best_model, best_results = detector.get_best_model()
        
        st.success(f"🏆 **Best Performing Model**: {best_model}")
        
        report_text = f"""
        ### Model Performance Summary
        
        **Champion Model**: {best_model}
        - **Accuracy**: {best_results['accuracy']*100:.2f}%
        - **Precision**: {best_results['precision']:.4f}
        - **Recall**: {best_results['recall']:.4f}
        - **F1-Score**: {best_results['f1_score']:.4f}
        
        ### Detailed Comparison
        """
        
        for model_name, metrics in results.items():
            report_text += f"""
        **{model_name}**:
        - Accuracy: {metrics['accuracy']*100:.2f}% {'✅' if metrics['accuracy'] == best_results['accuracy'] else ''}
        - Precision: {metrics['precision']:.4f}
        - Recall: {metrics['recall']:.4f}
        - F1-Score: {metrics['f1_score']:.4f}
        """
        
        st.markdown(report_text)


def about_page():
    """About project page"""
    
    st.markdown("## 📚 About This Project")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="content-card">
        <h3>🎯 Project Overview</h3>
        <p>
        This Fake News Detection System is an AI-powered application designed to identify 
        and classify fake news from social media content using Machine Learning techniques.
        </p>
        
        <h3>🎓 Academic Context</h3>
        <ul>
            <li><strong>Course:</strong> Artificial Intelligence & Data Science (AIDS)</li>
            <li><strong>Year:</strong> 3rd Year Mini Project</li>
            <li><strong>Focus:</strong> Classical Machine Learning (not Deep Learning)</li>
            <li><strong>Type:</strong> Text-based Analysis</li>
        </ul>
        
        <h3>🤖 Machine Learning Algorithms Used</h3>
        <ol>
            <li><strong>Naive Bayes:</strong> Probabilistic classifier based on Bayes' theorem</li>
            <li><strong>Logistic Regression:</strong> Linear model for binary classification</li>
            <li><strong>Support Vector Machine (SVM):</strong> Finds optimal hyperplane for classification</li>
        </ol>
        
        <h3>🔄 System Architecture</h3>
        <ol>
            <li><strong>Input:</strong> Social media text/news article</li>
            <li><strong>Preprocessing:</strong>
                <ul>
                    <li>Text cleaning (remove URLs, mentions, hashtags)</li>
                    <li>Tokenization</li>
                    <li>Stop-word removal</li>
                    <li>Stemming</li>
                </ul>
            </li>
            <li><strong>Feature Extraction:</strong> TF-IDF Vectorization</li>
            <li><strong>ML Model:</strong> Classification using trained algorithms</li>
            <li><strong>Output:</strong> Prediction (Fake/Real) with confidence score</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-card">
        <h3>🛠️ Tech Stack</h3>
        <ul>
            <li>🐍 Python 3.x</li>
            <li>📊 scikit-learn</li>
            <li>🔢 NumPy, Pandas</li>
            <li>📈 Matplotlib, Seaborn</li>
            <li>🌐 Streamlit</li>
            <li>💬 NLTK</li>
        </ul>
        
        <h3>📊 Evaluation Metrics</h3>
        <ul>
            <li>✅ Accuracy</li>
            <li>🎯 Precision</li>
            <li>📍 Recall</li>
            <li>⚖️ F1-Score</li>
            <li>📉 Confusion Matrix</li>
        </ul>
        
        <h3>💡 Key Features</h3>
        <ul>
            <li>Real-time prediction</li>
            <li>Multiple ML models</li>
            <li>Interactive UI</li>
            <li>Performance comparison</li>
            <li>Confidence scores</li>
            <li>Visual analytics</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="content-card">
    <h3>🎯 Project Objectives Achieved</h3>
    <ul>
        <li>✅ Clear problem statement and real-world motivation</li>
        <li>✅ Complete system architecture implementation</li>
        <li>✅ Comprehensive data preprocessing pipeline</li>
        <li>✅ Three ML algorithms trained and compared</li>
        <li>✅ Detailed evaluation metrics and visualizations</li>
        <li>✅ Professional web-based UI with social media theme</li>
        <li>✅ Well-documented code with comments</li>
        <li>✅ Easy-to-follow installation and usage guide</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


def train_page():
    """Training page"""
    
    st.markdown("## 🔬 Model Training Center")
    
    st.markdown("""
    <div class="content-card">
    <h3>📋 Training Process</h3>
    <p>
    This page allows you to train all three machine learning models from scratch. 
    The training process includes:
    </p>
    <ol>
        <li>Loading or generating sample dataset</li>
        <li>Text preprocessing and cleaning</li>
        <li>TF-IDF vectorization</li>
        <li>Training Naive Bayes, Logistic Regression, and SVM</li>
        <li>Model evaluation and metrics calculation</li>
        <li>Generating visualizations</li>
        <li>Saving trained models to disk</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if models already exist
    models_exist = os.path.exists(os.path.join(MODELS_DIR, 'naive_bayes.pkl'))
    
    if models_exist:
        st.warning("⚠️ Trained models already exist. Training again will overwrite them.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Start Training", use_container_width=True, type="primary"):
            with st.expander("📊 Training Progress", expanded=True):
                try:
                    detector, preprocessor, results = train_models_pipeline()
                    
                    st.balloons()
                    st.success("🎉 All models trained successfully!")
                    
                    # Show results
                    st.markdown("### 📊 Training Results")
                    evaluator = ModelEvaluator(results)
                    summary_df = evaluator.create_summary_table()
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
    
    # Display dataset info
    st.markdown("### 📊 Dataset Information")
    
    data_path = os.path.join(DATA_DIR, 'fake_news_sample.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Samples", len(df))
        
        with col2:
            st.metric("Real News", sum(df['label'] == 0))
        
        with col3:
            st.metric("Fake News", sum(df['label'] == 1))
        
        st.markdown("#### Sample Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.info("📝 Sample dataset will be generated during training.")


if __name__ == "__main__":
    main()
