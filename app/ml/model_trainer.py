"""
Machine Learning Model Trainer for Fake News Detection
Trains a Logistic Regression model using the WELFake dataset
"""

import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import joblib
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FakeNewsModelTrainer:
    """Train and evaluate fake news detection model using Logistic Regression"""
    
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path or "datasets/WELFake_Dataset.csv"
        self.preprocessed_path = "datasets/WELFake_Dataset_preprocessed.csv"
        self.models_dir = Path("app/ml/models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize NLTK components
        self._download_nltk_data()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Single model configuration - Logistic Regression only
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        
        # Best model will be stored here
        self.best_pipeline = None
        self.best_score = 0
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return ' '.join(tokens)
    
    def load_and_preprocess_data(self, force_reprocess=False):
        """Load and preprocess the dataset with caching"""
        
        # Check if preprocessed file exists and is newer than original
        if (not force_reprocess and 
            Path(self.preprocessed_path).exists() and 
            Path(self.preprocessed_path).stat().st_mtime > Path(self.dataset_path).stat().st_mtime):
            
            logger.info(f"Loading preprocessed dataset from {self.preprocessed_path}")
            df = pd.read_csv(self.preprocessed_path)
            logger.info(f"Preprocessed dataset shape: {df.shape}")
            return df
        
        logger.info(f"Loading and preprocessing dataset from {self.dataset_path}")
        
        # Load original dataset
        df = pd.read_csv(self.dataset_path)
        logger.info(f"Original dataset shape: {df.shape}")
        
        # Check for missing values
        logger.info(f"Missing values:\n{df.isnull().sum()}")
        
        # Combine title and text
        df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        
        # Preprocess combined text
        logger.info("Preprocessing text... This may take a few minutes.")
        df['processed_text'] = df['combined_text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        # Keep only necessary columns
        df_clean = df[['processed_text', 'label']].copy()
        
        # Save preprocessed data
        logger.info(f"Saving preprocessed dataset to {self.preprocessed_path}")
        df_clean.to_csv(self.preprocessed_path, index=False)
        
        # Check label distribution
        logger.info(f"Label distribution:\n{df_clean['label'].value_counts()}")
        logger.info(f"Label distribution (%):\n{df_clean['label'].value_counts(normalize=True) * 100}")
        
        return df_clean
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train the Logistic Regression model"""
        
        # Load and preprocess data
        df = self.load_and_preprocess_data()
        
        # Prepare features and labels
        X = df['processed_text']
        y = df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # TF-IDF Vectorizer with optimized parameters
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        logger.info("Training Logistic Regression model...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('classifier', self.model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
        
        result = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pipeline': pipeline,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        logger.info(f"Logistic Regression - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Set as best model
        self.best_score = accuracy
        self.best_pipeline = pipeline
        
        # Save model
        self.save_model()
        
        return result
    def train_model_quick(self, test_size=0.2, random_state=42, sample_size=10000):
        """Train model quickly with a smaller sample for testing"""
        
        # Load and preprocess data
        df = self.load_and_preprocess_data()
        
        # Sample data for quick training
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
            logger.info(f"Using sample of {sample_size} records for quick training")
        
        # Prepare features and labels
        X = df['processed_text']
        y = df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Quick training set size: {len(X_train)}")
        logger.info(f"Quick test set size: {len(X_test)}")
        
        # Simplified TF-IDF Vectorizer for speed
        vectorizer = TfidfVectorizer(
            max_features=5000,  # Reduced features
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        logger.info("Quick training Logistic Regression model...")
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', vectorizer),
            ('classifier', self.model)
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        # Simple cross-validation with fewer folds
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring='accuracy')
        
        result = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'pipeline': pipeline,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        logger.info(f"Quick Logistic Regression - Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Set as best model
        self.best_score = accuracy
        self.best_pipeline = pipeline
        
        # Save model
        self.save_model()
        
        return result
    
    def save_model(self):
        """Save the trained model"""
        if self.best_pipeline:
            model_path = self.models_dir / "fake_news_detector.joblib"
            joblib.dump(self.best_pipeline, model_path)
            logger.info(f"Logistic Regression model saved to {model_path}")
            
            # Save model metadata
            metadata = {
                'model_type': 'logistic_regression',
                'accuracy': self.best_score,
                'features': 'title + text (preprocessed)',
                'vectorizer': 'TfidfVectorizer',
                'preprocessed_data_path': self.preprocessed_path
            }
            
            metadata_path = self.models_dir / "model_metadata.joblib"
            joblib.dump(metadata, metadata_path)
            logger.info(f"Model metadata saved to {metadata_path}")
    
    def evaluate_model_details(self):
        """Generate detailed evaluation report"""
        if not self.best_pipeline:
            logger.error("No trained model found. Please run train_model() first.")
            return None
            
        # Load data
        df = self.load_and_preprocess_data()
        X = df['processed_text']
        y = df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Predictions
        y_pred = self.best_pipeline.predict(X_test)
        y_pred_proba = self.best_pipeline.predict_proba(X_test)
        
        # Detailed metrics
        report = {
            'model_type': 'logistic_regression',
            'accuracy': accuracy_score(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'sample_predictions': []
        }
        
        # Add sample predictions
        for i in range(min(10, len(X_test))):
            report['sample_predictions'].append({
                'text': X_test.iloc[i][:200] + "..." if len(X_test.iloc[i]) > 200 else X_test.iloc[i],
                'actual': int(y_test.iloc[i]),
                'predicted': int(y_pred[i]),
                'confidence': float(max(y_pred_proba[i]))
            })
        
        return report

def main(quick=False):
    """Main training function"""
    trainer = FakeNewsModelTrainer()
    
    # Train model
    if quick:
        logger.info("Starting quick Logistic Regression training (sample data)...")
        result = trainer.train_model_quick()
    else:
        logger.info("Starting full Logistic Regression training...")
        result = trainer.train_model()
    
    # Print results
    print("\n" + "="*60)
    print("📊 TRAINING RESULTS")
    if quick:
        print("⚡ (Quick Mode - 10K sample)")
    print("="*60)
    
    print(f"\n🔍 LOGISTIC REGRESSION:")
    print(f"   Accuracy: {result['accuracy']:.4f}")
    print(f"   Cross-validation: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")
    print("\nClassification Report:")
    print(result['classification_report'])
    
    print(f"\n🎯 Model Accuracy: {trainer.best_score:.4f}")
    
    # Detailed evaluation
    print("\n" + "="*60)
    print("📈 DETAILED EVALUATION")
    print("="*60)
    
    detailed_report = trainer.evaluate_model_details()
    if detailed_report:
        print(f"🤖 Final Model: Logistic Regression")
        print(f"🎯 Final Accuracy: {detailed_report['accuracy']:.4f}")
        
        real_metrics = detailed_report['classification_report']['1']
        print(f"📰 Real News Metrics:")
        print(f"   Precision: {real_metrics['precision']:.4f}")
        print(f"   Recall: {real_metrics['recall']:.4f}")
        print(f"   F1-Score: {real_metrics['f1-score']:.4f}")
        
        fake_metrics = detailed_report['classification_report']['0']
        print(f"🚫 Fake News Metrics:")
        print(f"   Precision: {fake_metrics['precision']:.4f}")
        print(f"   Recall: {fake_metrics['recall']:.4f}")
        print(f"   F1-Score: {fake_metrics['f1-score']:.4f}")
        
        print(f"\n📄 Sample Predictions:")
        for i, sample in enumerate(detailed_report['sample_predictions'][:3]):
            print(f"\nSample {i+1}:")
            print(f"   Text: {sample['text']}")
            print(f"   Actual: {'Real' if sample['actual'] == 1 else 'Fake'}")
            print(f"   Predicted: {'Real' if sample['predicted'] == 1 else 'Fake'}")
            print(f"   Confidence: {sample['confidence']:.3f}")

if __name__ == "__main__":
    import sys
    
    # Check for quick training argument
    quick_mode = '--quick' in sys.argv or '-q' in sys.argv
    
    if quick_mode:
        print("=" * 60)
        print("🚀 QUICK TRAINING MODE")
        print("=" * 60)
        print("• Using 10K sample for fast training")
        print("• Training Logistic Regression model only")
        print("• For full training: python train_model.py")
        print("-" * 60)
        print()
    else:
        print("=" * 60)
        print("🤖 FULL TRAINING MODE")
        print("=" * 60)
        print("• Using complete dataset (72K+ records)")
        print("• Training Logistic Regression model")
        print("• Includes comprehensive evaluation")
        print("• For quick training: python train_model.py --quick")
        print("-" * 60)
        print()
    
    sys.exit(main(quick=quick_mode))
