# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train Models (Optional - First Time)
```bash
python train.py
```
This will:
- Generate sample dataset
- Train all three ML models
- Save models to disk
- Generate visualizations

### Step 3: Run Web Application
```bash
streamlit run app.py
```

The app will open at: http://localhost:8501

---

## 📱 Using the Application

### Detect Fake News
1. Go to "🏠 Home - Detect News" page
2. Enter or select news text
3. Choose ML model from sidebar
4. Click "Analyze News"
5. View prediction and confidence score

### View Performance
1. Go to "📊 Model Performance" page
2. Compare all three models
3. View accuracy metrics
4. Explore confusion matrices

---

## 💡 Tips

- **First time users**: Train models first or use the "🔬 Train Models" page in the web app
- **Sample texts**: Use pre-loaded samples to test the system quickly
- **Model comparison**: Try different models to see performance differences
- **Confidence scores**: Higher confidence means more certain prediction

---

## 🆘 Troubleshooting

### Issue: NLTK Data Not Found
**Solution**: Run this in Python:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Issue: Models Not Found
**Solution**: Run training script:
```bash
python train.py
```
OR use "🔬 Train Models" page in web app

### Issue: Port Already in Use
**Solution**: Use different port:
```bash
streamlit run app.py --server.port 8502
```

---

## 📊 Project Files

- `app.py` - Web application (main file)
- `train.py` - Training script
- `preprocessing.py` - Text preprocessing
- `model_training.py` - ML models
- `evaluation.py` - Metrics and visualizations
- `dataset_generator.py` - Sample data
- `config.py` - Configuration

---

## 🎓 For Presentation

Key features to demonstrate:
1. **Real-time detection** - Enter news and get instant results
2. **Multiple models** - Switch between algorithms
3. **Confidence scores** - See prediction probability
4. **Performance metrics** - Show model comparison
5. **Interactive UI** - Professional and user-friendly

---

## 📞 Need Help?

Check the full README.md for detailed documentation.

---

**Happy Fake News Hunting! 🔍**
