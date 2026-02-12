# Binary TabPFN-based Sentiment Analysis Training Script (Positive/Negative Only)
import os
import sys

# CRITICAL FIX: Permanent Mock TensorFlow for einops/transformers
# 1. Mock TF so einops can import it and definitions exist
import types
import sys
from unittest.mock import MagicMock

dummy_tf = types.ModuleType('tensorflow')
class DummyTensor: pass
dummy_tf.Tensor = DummyTensor
dummy_tf.Variable = DummyTensor

# 2. Add a dummy spec so transformers importlib checks pass
dummy_spec = MagicMock()
dummy_spec.name = "tensorflow"
dummy_spec.loader = None
dummy_spec.origin = None
dummy_spec.submodule_search_locations = []
dummy_spec.has_location = False
dummy_tf.__spec__ = dummy_spec

# 3. Register in sys.modules
sys.modules['tensorflow'] = dummy_tf

import pandas as pd
import numpy as np
import glob
import re
import warnings
import torch
import gc
import time
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Disable multiprocessing to avoid segmentation faults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# TabPFN Import
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
    print("✅ Using TabPFN Classifier")
except ImportError:
    try:
        from tabpfn.models import TabPFNClassifier
        TABPFN_AVAILABLE = True
        print("✅ Using TabPFN Classifier from models module")
    except ImportError:
        TABPFN_AVAILABLE = False
        print("⚠️  TabPFN not available. Install with: pip install tabpfn")

# Transformers for GT
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("❌ Transformers not available. Install with: pip install transformers")

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================
class Config:
    DATA_DIR = '../dataset'
    OUTPUT_DIR = 'outputs'
    CONFUSION_MATRIX_DIR = os.path.join(OUTPUT_DIR, 'binary_tabpfn_confusion_matrix')
    METRICS_FILE = os.path.join(OUTPUT_DIR, 'binary_sentiment_metrics.csv')

# ============================================================================
# ENHANCED SENTIMENT KEYWORDS (from train.py)
# ============================================================================
POSITIVE_KEYWORDS = {
    'good': 2, 'best': 3, 'fast': 2, 'great': 3, 'good service': 4, 
    'friendly': 2, 'user friendly': 4, 'excellent': 4, 'love': 3, 
    'fast delivery': 4, 'recommend': 3, 'service good': 4, 'best food': 4,
    'amazing': 4, 'satisfied': 3, 'awesome': 4, 'good food': 4, 
    'recommended': 3, 'good app': 4, 'quick': 2, 'delicious': 3,
    'app good': 4, 'helpful': 2, 'highly recommend': 5, 'delivery fast': 4,
    'faster': 2, 'good experience': 4, 'happy': 3, 'excellent service': 5,
    'quick easy': 4, 'super fast': 4, 'fresh': 2, 'delivery good': 4,
    'great service': 4, 'really good': 4, 'good delivery': 4, 'best service': 5,
    'loved': 3, 'love app': 4, 'food good': 4, 'great app': 4,
    'highly recommended': 5, 'best wishes': 2, 'best app': 4, 'best luck': 2,
    'friendly interface': 4, 'quickly': 2, 'satisfied service': 4,
    'easy delicious': 4, 'best delivery': 4, 'perfect': 4, 'outstanding': 4,
    'fantastic': 4, 'wonderful': 4, 'smooth': 3, 'reliable': 3,
    'convenient': 3, 'efficient': 3, 'responsive': 3, 'professional': 3
}

NEGATIVE_KEYWORDS = {
    'worst': 4, 'bad': 3, 'worst app': 5, 'poor': 3, 'late': 3,
    'worst food': 5, 'wrong': 3, 'bad experience': 4, 'worst experience': 5,
    'problem': 3, 'bad service': 4, 'worst service': 5, 'poor service': 4,
    'scam': 5, 'disappointed': 3, 'service bad': 4, 'fake': 4,
    'bad app': 4, 'cold': 3, 'late delivery': 4, 'delay': 3,
    'worst customer': 5, 'worst delivery': 5, 'later': 2, 'missing': 3,
    'fraud': 5, 'bad food': 4, 'late night': 2, 'error': 3,
    'app worst': 5, 'service worst': 5, 'problems': 3, 'bad delivery': 4,
    'crashes': 4, 'app bad': 4, 'problem problem': 4, 'service poor': 4,
    'poor app': 4, 'bad customer': 4, 'delayed': 3, 'rude': 3,
    'wrong order': 4, 'went wrong': 4, 'cold food': 4, 'poor delivery': 4,
    'poor customer': 4, 'minutes later': 2, 'late hours': 2, 'time late': 2,
    'wrong delivery': 4, 'terrible': 4, 'horrible': 4, 'awful': 4,
    'disgusting': 4, 'useless': 4, 'waste': 4, 'never again': 5,
    'avoid': 4, 'unprofessional': 4, 'slow': 3, 'not working': 4
}

# ============================================================================
# Text Preprocessing
# ============================================================================
def clean_text_advanced(text):
    """Advanced text cleaning with better handling"""
    if pd.isna(text) or not text:
        return ""
    
    text = str(text).strip()
    if len(text) == 0:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters but keep Bengali and basic punctuation
    text = re.sub(r'[^\w\s\u0980-\u09FF@#!?]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text if len(text) > 0 else " "

# ============================================================================
# ADVANCED SENTIMENT DETECTION (Ground Truth Fallback)
# ============================================================================
def detect_sentiment_advanced(text):
    """Advanced sentiment detection with contextual understanding"""
    if not text or pd.isna(text):
        return 1  # Neutral
    
    text_lower = str(text).lower()
    text_original = str(text)
    
    negative_score = 0
    positive_score = 0
    
    for keyword, weight in NEGATIVE_KEYWORDS.items():
        if ' ' in keyword:
            if keyword in text_lower:
                negative_score += weight * 2.0
        else:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                negative_score += weight
    
    for keyword, weight in POSITIVE_KEYWORDS.items():
        if ' ' in keyword:
            if keyword in text_lower:
                positive_score += weight * 2.0
        else:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text_lower):
                positive_score += weight
    
    negative_emojis = ['😡', '🤬', '💩', '😠', '😤', '😞', '😢', '😭', '👎', '💔', '😒', '😩']
    positive_emojis = ['😍', '🥰', '🤩', '😊', '👍', '❤️', '🔥', '⭐', '🌟', '✅', '🎉', '😎', '🙌', '💫']
    
    for emoji in negative_emojis:
        if emoji in text_original:
            negative_score += 4
    
    for emoji in positive_emojis:
        if emoji in text_original:
            positive_score += 4
    
    text_length = len(text_lower.split())
    if text_length > 20:
        if negative_score > positive_score:
            negative_score += text_length * 0.1
        elif positive_score > negative_score:
            positive_score += text_length * 0.1
    
    if negative_score > 0 and positive_score > 0:
        ratio = negative_score / positive_score if positive_score > 0 else float('inf')
        if ratio > 2.0:
            return 0  # Negative
        elif ratio < 0.5:
            return 2  # Positive
        else:
            return 1  # Neutral - when scores are balanced
    
    elif negative_score >= 5:
        return 0  # Negative
    
    elif positive_score >= 5:
        return 2  # Positive
    
    elif negative_score > 0:
        if negative_score >= 3:
            return 0  # Negative
        else:
            return 1  # Neutral
    
    elif positive_score > 0:
        if positive_score >= 3:
            return 2  # Positive
        else:
            return 1  # Neutral
    
    else:
        return 1  # Neutral

# ============================================================================
# Ground Truth Generation (Using RoBERTa)
# ============================================================================
def get_sentiment_model(model_name):
    """Load sentiment model with error handling"""
    if not TRANSFORMERS_AVAILABLE:
        return None, None
    try:
        gc.collect()
        time.sleep(0.2)
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            local_files_only=False
        )
        
        time.sleep(0.1)
        
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                local_files_only=False
            )
        except (TypeError, ValueError):
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                local_files_only=False
            )
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        model = model.cpu()
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None, None

def predict_sentiment_batch(texts, tokenizer, model, model_name, device="cpu"):
    """Predict sentiment for a batch of texts"""
    if tokenizer is None or model is None:
        return [-1] * len(texts)
    
    try:
        model_label_map = {
            "nlptown/bert-base-multilingual-uncased-sentiment": {0: "negative", 1: "negative", 2: "neutral", 3: "positive", 4: "positive"},
            "distilbert-base-uncased-finetuned-sst-2-english": {0: "negative", 1: "positive"},
            "cardiffnlp/twitter-roberta-base-sentiment-latest": {0: "negative", 1: "neutral", 2: "positive"},
            "microsoft/deberta-base": {0: "negative", 1: "positive"},
            "microsoft/deberta-v3-base": {0: "negative", 1: "positive"}
        }
        
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
            add_special_tokens=True
        )
        
        model = model.cpu()
        
        with torch.no_grad():
            logits = model(**inputs).logits
        
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        
        label_map = model_label_map.get(model_name, {0: "negative", 1: "positive"})
        sentiments = []
        for pred in predictions:
            if pred in label_map:
                sentiment = label_map[pred]
                if sentiment == "negative":
                    sentiments.append(0)
                elif sentiment == "neutral":
                    sentiments.append(1)
                else:
                    sentiments.append(2)
            else:
                sentiments.append(-1)
        
        return sentiments
    except Exception as e:
        print(f"Error in batch sentiment prediction for {model_name}: {e}")
        return [-1] * len(texts)

# ============================================================================
# Feature Extraction (TF-IDF)
# ============================================================================
def create_features(texts_train, texts_test, max_features=100):
    """Create TF-IDF features from text
    Note: TabPFN is restricted to max 100 features, so we limit to 100
    """
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
            stop_words='english'  # Remove English stop words
        )
        
        X_train_features = vectorizer.fit_transform(texts_train)
        X_test_features = vectorizer.transform(texts_test)
        
        # Convert to dense array (TabPFN works better with dense arrays)
        X_train_features = X_train_features.toarray()
        X_test_features = X_test_features.toarray()
        
        return X_train_features, X_test_features, vectorizer
    except Exception as e:
        print(f"   ❌ Error creating TF-IDF features: {e}")
        return None, None, None

# ============================================================================
# Data Loading
# ============================================================================
def load_data_enhanced():
    """Load data with enhanced preprocessing"""
    data_dir = Config.DATA_DIR
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not all_files:
        print(f"❌ No CSV files found in {data_dir}")
        return None
    
    print(f"📂 Found {len(all_files)} CSV files")
    
    all_data = []
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
            print(f"   📄 Loading {os.path.basename(file_path)}: {len(df)} rows")
            
            if 'content' not in df.columns:
                for col in df.columns:
                    if 'review' in col.lower() or 'text' in col.lower() or 'comment' in col.lower():
                        df['content'] = df[col]
                        break
            
            if 'content' not in df.columns:
                continue
                
            df = df.dropna(subset=["content"])
            df['content'] = df['content'].astype(str)
            df['cleaned_text'] = df['content'].apply(clean_text_advanced)
            all_data.append(df)
            
        except Exception as e:
            print(f"   ❌ Error loading {os.path.basename(file_path)}: {e}")
    
    if not all_data:
        print("❌ No valid data found")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Combined dataset: {len(combined_df)} rows")
    return combined_df

# ============================================================================
# Metrics Saving
# ============================================================================
def save_metrics(metrics_list, output_file):
    """Save metrics to a CSV file"""
    try:
        df = pd.DataFrame(metrics_list)
        
        # Check if file exists to append or create new
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            # Append new metrics
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            # Remove duplicates if any (keeping latest)
            combined_df = combined_df.drop_duplicates(subset=['Model'], keep='last')
            combined_df.to_csv(output_file, index=False)
        else:
            df.to_csv(output_file, index=False)
            
        print(f"   💾 Saved metrics to {output_file}")
        return True
    except Exception as e:
        print(f"   ❌ Error saving metrics: {e}")
        return False

# ============================================================================
# Confusion Matrix Generation
# ============================================================================
def create_confusion_matrix(y_true, y_pred, model_name, output_dir, accuracy=None):
    """Create and save confusion matrix for a model with accuracy displayed"""
    try:
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {model_name}\nAccuracy: {accuracy:.2f}%')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save confusion matrix
        save_path = os.path.join(output_dir, f'{model_name.replace("/", "_")}_confusion_matrix.png')
        plt.savefig(save_path)
        plt.close()
        print(f"   📊 Saved confusion matrix to {save_path}")
        return True
    
    except Exception as e:
        print(f"   ❌ Error creating confusion matrix: {e}")
        return False

# ============================================================================
# Main Training Function
# ============================================================================
def main():
    print("="*80)
    print("🎯 BINARY TABPFN SENTIMENT ANALYSIS (Positive vs Negative)")
    print("="*80)
    
    if not TABPFN_AVAILABLE:
        print("❌ TabPFN not available.")
        return
    
    # Force CPU
    device = "cpu"
    print(f"Using device: {device}\n")
    
    # Create output directory
    os.makedirs(Config.CONFUSION_MATRIX_DIR, exist_ok=True)
    print(f"📁 Confusion matrices will be saved in: {Config.CONFUSION_MATRIX_DIR}\n")
    
    # Load data
    print("📂 Loading and preprocessing data...")
    df = load_data_enhanced()
    if df is None:
        return
    
    # Use RoBERTa as ground truth (same as bert_train.py and ml_train.py)
    print("🔍 Using RoBERTa as ground truth for high accuracy...")
    
    roberta_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    print("   📥 Loading RoBERTa for ground truth...")
    roberta_tokenizer, roberta_model = get_sentiment_model(roberta_model_name)
    
    if roberta_tokenizer and roberta_model:
        texts = df['cleaned_text'].tolist()
        ground_truth_predictions = []
        
        print("   🔮 Generating ground truth labels with RoBERTa...")
        for i in tqdm(range(0, len(texts), 8), desc="      RoBERTa GT", leave=False):
            try:
                batch_texts = texts[i:i+8]
                batch_preds = predict_sentiment_batch(batch_texts, roberta_tokenizer, roberta_model, roberta_model_name, device)
                ground_truth_predictions.extend(batch_preds)
                gc.collect()
            except Exception as e:
                print(f"\n         ⚠️  Error in batch {i//8 + 1}: {e}")
                ground_truth_predictions.extend([-1] * len(batch_texts))
                gc.collect()
                continue
        
        while len(ground_truth_predictions) < len(texts):
            ground_truth_predictions.append(-1)
        
        df['ground_truth'] = ground_truth_predictions
        
        invalid_mask = df['ground_truth'] == -1
        if invalid_mask.sum() > 0:
            print(f"   ⚠️  {invalid_mask.sum()} invalid RoBERTa predictions, using keyword-based fallback...")
            df.loc[invalid_mask, 'ground_truth'] = df.loc[invalid_mask, 'cleaned_text'].apply(detect_sentiment_advanced)
        
        del roberta_model
        del roberta_tokenizer
        gc.collect()
        time.sleep(1.0)
    else:
        print("   ⚠️  Failed to load RoBERTa, using keyword-based ground truth...")
        df['ground_truth'] = df['cleaned_text'].apply(detect_sentiment_advanced)
    
    # ============================================================================
    # BINARY FILTERING
    # ============================================================================
    print("\n⚖️  Filtering for BINARY sentiment (Negative vs Positive)...")
    original_len = len(df)
    binary_df = df[df['ground_truth'] != 1].copy()
    
    # Map Positive (2) to 1, keeping Negative (0) as 0
    binary_df['binary_label'] = binary_df['ground_truth'].apply(lambda x: 1 if x == 2 else 0)
    
    print(f"   📉 Filtered {original_len} -> {len(binary_df)} samples (removed {original_len - len(binary_df)} neutral)")
    print(f"   📊 Binary distribution: {binary_df['binary_label'].value_counts().to_dict()}")
    
    # Get texts and labels for binary data
    texts = binary_df['cleaned_text'].tolist()
    y_true = binary_df['binary_label'].tolist()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y_true, test_size=0.2, random_state=42, stratify=y_true
    )
    
    print(f"📊 Dataset split:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Testing: {len(X_test)} samples")
    
    # TabPFN Size Warning - limit if excessively large, but try to use all
    X_train_sampled, y_train_sampled = X_train, y_train
    if len(X_train) > 1000:
        print(f"\n⚠️  Training set size ({len(X_train)}) exceeds TabPFN's recommended limit (1000)")
        print(f"   Using all samples for better accuracy (ignore_pretraining_limits=True)")
    
    # Create TF-IDF features
    print(f"\n📥 Creating TF-IDF features...")
    X_train_features, X_test_features, vectorizer = create_features(X_train_sampled, X_test)
    
    if X_train_features is None or X_test_features is None:
        print(f"❌ Failed to create features")
        return
    
    print(f"✅ Features created: Train shape {X_train_features.shape}, Test shape {X_test_features.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_test_scaled = scaler.transform(X_test_features)
    
    # Train TabPFN
    print(f"\n🤖 Training TabPFN...")
    
    try:
        model = None
        # Try TabPFN Init
        try:
            model = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
        except TypeError:
            try:
                model = TabPFNClassifier(device='cpu')
            except Exception as e:
                print(f"TabPFN Init Error: {e}")
                
        if model is None:
            raise Exception("TabPFN initialization failed")
            
        # Fit model - Patch validation if needed (reusing patch logic if relevant, but simplified here)
        print("   Fitting TabPFN model...")
        model.fit(X_train_scaled, y_train_sampled)
        print("   ✅ TabPFN fitted successfully")
        
        # Predict
        print("   🔮 Predicting on test set...")
        y_pred = model.predict(X_test_scaled)
        
        # Score
        accuracy = accuracy_score(y_test, y_pred) * 100
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100
        
        print(f"   ✅ Accuracy: {accuracy:.2f}%, F1: {f1:.2f}%")
        
        # Save Confusion Matrix
        create_confusion_matrix(y_test, y_pred, "TabPFN", Config.CONFUSION_MATRIX_DIR, accuracy=accuracy)
        
        # Append to metrics
        metrics = [{
            'Model': 'TabPFN',
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }]
        save_metrics(metrics, Config.METRICS_FILE)
        
    except Exception as e:
        print(f"   ❌ Error training TabPFN: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
