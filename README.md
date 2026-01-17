# 🔍 Fake News Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange.svg)
![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)
> An AI-powered system to detect fake news on social media using Machine Learning techniques

---


## 🎯 Project Overview

### Problem Statement

In the digital age, social media has become a primary source of news and information. However, the rapid spread of fake news poses a significant threat to society, influencing public opinion and decision-making. This project aims to develop an automated system to detect and classify fake news using Machine Learning algorithms.

### Real-World Motivation

- **Misinformation Crisis**: Fake news spreads faster than real news on social media
- **Social Impact**: False information affects elections, public health, and social harmony
- **Need for Automation**: Manual verification is time-consuming and impractical at scale
- **Educational Purpose**: Understanding ML applications in NLP and text classification

### Project Scope

This is an **academic mini project** focusing on:
- Classical Machine Learning techniques (not deep learning)
- Text-based analysis (no images or videos)
- Comparative study of multiple ML algorithms
- Emphasis on explainability and methodology

---

## ✨ Features

### Core Features
- ✅ **Real-time Fake News Detection** - Instant analysis of news articles
- ✅ **Multiple ML Algorithms** - Compare 3 different models
- ✅ **Confidence Scoring** - Probability-based predictions
- ✅ **Interactive Web UI** - User-friendly Streamlit interface
- ✅ **Visual Analytics** - Comprehensive performance visualizations
- ✅ **Sample Texts** - Pre-loaded examples for testing

### Technical Features
- 📊 TF-IDF Vectorization for text features
- 🔄 Complete preprocessing pipeline
- 📈 Multiple evaluation metrics
- 💾 Model persistence (save/load trained models)
- 🎨 Professional UI with social media theme
- 📉 Confusion matrices and comparison charts

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                          │
│              Social Media Text / News Article               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   PREPROCESSING LAYER                       │
│  ┌──────────────┬──────────────┬──────────────┬──────────┐ │
│  │Text Cleaning │ Tokenization │Stop-word     │ Stemming │ │
│  │(URLs, etc.)  │              │Removal       │          │ │
│  └──────────────┴──────────────┴──────────────┴──────────┘ │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                 FEATURE EXTRACTION LAYER                    │
│              TF-IDF Vectorization (5000 features)           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   MACHINE LEARNING LAYER                    │
│  ┌──────────────┬──────────────────┬───────────────────┐   │
│  │ Naive Bayes  │Logistic Regression│ Support Vector   │   │
│  │  Classifier  │   (max_iter=1000) │  Machine (Linear)│   │
│  └──────────────┴──────────────────┴───────────────────┘   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                      OUTPUT LAYER                           │
│      Classification: FAKE (1) or REAL (0)                   │
│      Confidence Score: Probability Distribution             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

### Programming Language
- **Python 3.8+**

### Core Libraries

#### Machine Learning & Data Processing
- `scikit-learn` - ML algorithms and metrics
- `numpy` - Numerical computations
- `pandas` - Data manipulation and analysis
- `nltk` - Natural Language Processing

#### Visualization
- `matplotlib` - Static plots and charts
- `seaborn` - Statistical visualizations
- `plotly` - Interactive charts

#### Web Framework
- `streamlit` - Web application interface

#### Utilities
- `joblib` - Model persistence
- `tqdm` - Progress bars

---

## 📥 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 4GB RAM minimum
- Internet connection (for first-time NLTK data download)

### Step-by-Step Installation

#### 1. Clone or Download the Project

```bash
# If you have git
git clone <repository-url>
cd "Fake News Detection using MLT"

# Or simply extract the ZIP file
```

#### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- numpy==1.24.3
- pandas==2.0.3
- scikit-learn==1.3.0
- nltk==3.8.1
- streamlit==1.28.0
- matplotlib==3.7.2
- seaborn==0.12.2
- plotly==5.17.0
- tqdm==4.66.1
- Pillow==10.0.0
- joblib==1.3.2

#### 4. Download NLTK Data (Automatic)

The application will automatically download required NLTK data on first run:
- Stopwords corpus
- Punkt tokenizer

---

## 🚀 Usage

### Method 1: Run the Web Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Method 2: Train Models First (If Needed)

If models are not trained yet, you can train them through the web interface:

1. Run the application: `streamlit run app.py`
2. Navigate to **"🔬 Train Models"** page
3. Click **"🚀 Start Training"** button
4. Wait for training to complete (2-5 minutes)
5. Models will be saved automatically

### Using the Application

#### Home Page - Detect News
1. **Select Input Method**:
   - Type/paste your own news text
   - OR select from sample texts

2. **Choose ML Model** (from sidebar):
   - Naive Bayes
   - Logistic Regression
   - Support Vector Machine

3. **Click "Analyze News"** button

4. **View Results**:
   - Prediction: Fake or Real
   - Confidence scores
   - Visual confidence meter

#### Model Performance Page
- View comparison of all three models
- See accuracy, precision, recall, F1-score
- Explore confusion matrices
- Interactive charts


## 🤖 Machine Learning Models

### 1. Naive Bayes Classifier

**Algorithm Type**: Probabilistic Classifier

**How It Works**:
- Based on Bayes' theorem with independence assumption
- Calculates probability of text being fake/real
- Effective for text classification tasks

**Parameters**:
```python
MultinomialNB(alpha=1.0)
```

**Advantages**:
- Fast training and prediction
- Works well with small datasets
- Good baseline model

### 2. Logistic Regression

**Algorithm Type**: Linear Classification Model

**How It Works**:
- Uses logistic function to model binary outcomes
- Finds decision boundary between classes
- Provides probability estimates

**Parameters**:
```python
LogisticRegression(max_iter=1000, random_state=42)
```

**Advantages**:
- Interpretable coefficients
- Good generalization
- Probability calibration

### 3. Support Vector Machine (SVM)

**Algorithm Type**: Maximum Margin Classifier

**How It Works**:
- Finds optimal hyperplane separating classes
- Maximizes margin between support vectors
- Effective in high-dimensional spaces

**Parameters**:
```python
SVC(kernel='linear', probability=True, random_state=42)
```

**Advantages**:
- Effective with TF-IDF features
- Robust to overfitting
- Good performance on text data

---

## 📊 Dataset

### Sample Dataset Structure

The project includes a sample dataset with 60 news articles:
- **30 Real News** - Legitimate news from credible sources
- **30 Fake News** - Fabricated or misleading content

### CSV Format

```csv
text,label
"Scientists announce breakthrough in renewable energy...",0
"SHOCKING: Aliens spotted in city center!!!",1
```

- **text**: News article content
- **label**: 0 = Real, 1 = Fake

### Data Split
- **Training Set**: 80% (48 samples)
- **Testing Set**: 20% (12 samples)
- **Stratified Split**: Maintains class distribution

### Using Your Own Dataset

To use your own dataset:

1. Create a CSV file with columns: `text` and `label`
2. Place it in the `data/` folder
3. Update the filename in training pipeline
4. Ensure labels are: 0 (Real) and 1 (Fake)

**Minimum Recommendations**:
- At least 100+ samples per class
- Balanced distribution (equal Real/Fake)
- Diverse content from various sources

---

## 📈 Evaluation Metrics

### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
- Overall correctness of the model
- Percentage of correct predictions

### Precision
```
Precision = TP / (TP + FP)
```
- Of all predicted fake news, how many are actually fake?
- Important to minimize false alarms

### Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```
- Of all actual fake news, how many did we catch?
- Important to catch maximum fake news

### F1-Score
```
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```
- Harmonic mean of precision and recall
- Balanced metric for model comparison

### Confusion Matrix

```
                Predicted
                Real  Fake
Actual  Real    [TN]  [FP]
        Fake    [FN]  [TP]
```

Where:
- **TP** (True Positive): Correctly identified fake news
- **TN** (True Negative): Correctly identified real news
- **FP** (False Positive): Real news wrongly marked as fake
- **FN** (False Negative): Fake news wrongly marked as real

---

## 📁 Project Structure

```
Fake News Detection using MLT/
│
├── app.py                      # Main Streamlit web application
├── config.py                   # Project configuration and constants
├── preprocessing.py            # Text preprocessing module
├── model_training.py           # ML model training pipeline
├── evaluation.py               # Model evaluation and visualization
├── dataset_generator.py        # Sample data generator
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)
│
├── data/                       # Dataset folder
│   └── fake_news_sample.csv   # Sample dataset (generated)
│
├── models/                     # Trained models folder
│   ├── naive_bayes.pkl        # Saved Naive Bayes model
│   ├── logistic_regression.pkl # Saved Logistic Regression model
│   ├── svm.pkl                # Saved SVM model
│   ├── preprocessor.pkl       # Saved text preprocessor
│   └── results.pkl            # Saved evaluation results
│
└── assets/                     # Generated visualizations
    ├── metrics_comparison.png
    ├── accuracy_comparison.png
    ├── cm_naive_bayes.png
    ├── cm_logistic_regression.png
    ├── cm_svm.png
    ├── radar_comparison.html
    └── results_summary.csv
```

---

## 📸 Screenshots

### Home Page - News Detection
![Home Page](assets/screenshot_home.png)
*Main interface for entering and analyzing news articles*

### Prediction Results
![Results](assets/screenshot_results.png)
*Detailed prediction with confidence scores and visual meter*

### Model Performance
![Performance](assets/screenshot_performance.png)
*Comparison of all three ML models with metrics*

### Training Page
![Training](assets/screenshot_training.png)
*Model training interface with progress tracking*

---


**Step 3: Feature Extraction**
- TF-IDF (Term Frequency-Inverse Document Frequency)
- Converts text to numerical features
- Max 5000 features used

**Step 4: Model Training**
- Train three models: Naive Bayes, Logistic Regression, SVM
- 80-20 train-test split
- Stratified sampling for balanced classes

**Step 5: Evaluation**
- Calculate accuracy, precision, recall, F1-score
- Generate confusion matrices
- Compare model performance

#### 3. Technical Questions & Answers

**Q: Why TF-IDF instead of Bag of Words?**
- TF-IDF considers both term frequency and document importance
- Reduces weight of common words
- Better feature representation

**Q: Why these three algorithms?**
- Naive Bayes: Fast, probabilistic, good baseline
- Logistic Regression: Interpretable, linear decision boundary
- SVM: Robust, effective in high dimensions

**Q: How do you handle overfitting?**
- Train-test split
- Cross-validation (mentioned in code)
- Regularization in Logistic Regression and SVM

**Q: What are limitations?**
- Limited to text analysis (no images/videos)
- Requires labeled training data
- May not catch sophisticated fake news
- Language-specific (English)

#### 4. Results Discussion

**Expected Performance**:
- Accuracy: 85-95% (on sample data)
- All three models perform comparably
- SVM typically highest, Naive Bayes fastest

**Real-world Considerations**:
- Model needs regular updates
- New fake news patterns emerge
- Requires larger, diverse datasets

#### 5. Future Enhancements
- Deep Learning (LSTM, BERT)
- Multi-language support
- Image/video analysis
- Real-time social media integration
- Fact-checking database integration

### Presentation Flow

1. **Introduction** (2 min)
   - Problem statement
   - Motivation
   - Objectives

2. **Literature Review** (2 min)
   - Brief overview of fake news research
   - ML applications in text classification

3. **Methodology** (5 min)
   - System architecture diagram
   - Preprocessing steps
   - ML algorithms explanation
   - Evaluation metrics

4. **Implementation** (3 min)
   - Tech stack
   - Code structure
   - Key modules

5. **Results** (3 min)
   - Model comparison
   - Accuracy metrics
   - Visualizations

6. **Demo** (3 min)
   - Live application walkthrough
   - Test with sample texts

7. **Conclusion** (2 min)
   - Achievements
   - Limitations
   - Future work

---

## 🚀 Future Enhancements

### Short-term
- [ ] Add more diverse dataset
- [ ] Implement cross-validation
- [ ] Add feature importance visualization
- [ ] Export detailed PDF reports

### Medium-term
- [ ] Deep Learning models (LSTM, BERT)
- [ ] Multi-language support
- [ ] Browser extension for real-time detection
- [ ] API for integration with other apps

### Long-term
- [ ] Image and video analysis
- [ ] Source credibility scoring
- [ ] Fact-checking database integration
- [ ] Mobile application

---


---



---

## 📚 References

1. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). "Fake news detection on social media: A data mining perspective." *ACM SIGKDD explorations newsletter*.

2. Pérez-Rosas, V., Kleinberg, B., Lefevre, A., & Mihalcea, R. (2017). "Automatic detection of fake news." *arXiv preprint*.

3. Scikit-learn Documentation: https://scikit-learn.org/
4. NLTK Documentation: https://www.nltk.org/
5. Streamlit Documentation: https://docs.streamlit.io/

---

<div align="center">


