"""
Hybrid Fake News Prediction Service
DistilBERT as primary model with Logistic Regression fallback
Optimized for NVIDIA GTX 1050 (2GB VRAM)
"""

import joblib
import re
import string
from pathlib import Path
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# DistilBERT imports (with graceful fallback)
try:
    import torch
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    DISTILBERT_AVAILABLE = True
except ImportError:
    DISTILBERT_AVAILABLE = False
    logger.warning("DistilBERT dependencies not found. Using Logistic Regression only.")

logger = logging.getLogger(__name__)

class FakeNewsPredictor:
    """Hybrid predictor using DistilBERT (primary) and Logistic Regression (fallback)"""
    
    def __init__(self, model_path=None, use_distilbert=True):
        # Model paths
        self.lr_model_path = model_path or Path("app/ml/models/fake_news_detector.joblib")
        self.lr_metadata_path = Path("app/ml/models/model_metadata.joblib")
        self.distilbert_model_path = Path("app/ml/models/distilbert_fake_news_detector.pt")
        self.distilbert_tokenizer_path = Path("app/ml/models/distilbert_tokenizer")
        self.distilbert_metadata_path = Path("app/ml/models/distilbert_metadata.joblib")
        
        # Model preference
        self.use_distilbert = use_distilbert and DISTILBERT_AVAILABLE
        
        # Device setup
        self.device = self._setup_device()
        
        # Initialize NLTK components
        self._download_nltk_data()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Model containers
        self.lr_model = None
        self.lr_metadata = None
        self.distilbert_model = None
        self.distilbert_tokenizer = None
        self.distilbert_metadata = None
        
        # Load models
        self.load_models()
        
        # Determine active model
        self.active_model = self._determine_active_model()
        logger.info(f"Active model: {self.active_model}")
    
    def _setup_device(self):
        """Setup device for DistilBERT inference"""
        if not DISTILBERT_AVAILABLE:
            return None
            
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.cuda.empty_cache()
            logger.info(f"GPU available for inference: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU for DistilBERT inference")
        
        return device
    
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
    
    def load_models(self):
        """Load all available models"""
        # Load Logistic Regression model (fallback)
        self._load_lr_model()
        
        # Load DistilBERT model (primary)
        if self.use_distilbert:
            self._load_distilbert_model()
    
    def _load_lr_model(self):
        """Load Logistic Regression model"""
        try:
            if self.lr_model_path.exists():
                self.lr_model = joblib.load(self.lr_model_path)
                logger.info(f"Logistic Regression model loaded from {self.lr_model_path}")
                
                if self.lr_metadata_path.exists():
                    self.lr_metadata = joblib.load(self.lr_metadata_path)
                    logger.info(f"LR model metadata loaded")
            else:
                logger.warning(f"Logistic Regression model not found at {self.lr_model_path}")
        except Exception as e:
            logger.error(f"Error loading Logistic Regression model: {e}")
    
    def _load_distilbert_model(self):
        """Load DistilBERT model"""
        if not DISTILBERT_AVAILABLE:
            logger.warning("DistilBERT dependencies not available")
            return
            
        try:
            # Load tokenizer
            if self.distilbert_tokenizer_path.exists():
                self.distilbert_tokenizer = DistilBertTokenizer.from_pretrained(
                    str(self.distilbert_tokenizer_path)
                )
                logger.info(f"DistilBERT tokenizer loaded")
            
            # Load model
            if self.distilbert_model_path.exists():
                checkpoint = torch.load(self.distilbert_model_path, map_location=self.device)
                
                # Initialize model
                self.distilbert_model = DistilBertForSequenceClassification.from_pretrained(
                    checkpoint['model_name'],
                    num_labels=2
                )
                
                # Load trained weights
                self.distilbert_model.load_state_dict(checkpoint['model_state_dict'])
                self.distilbert_model.to(self.device)
                self.distilbert_model.eval()
                
                logger.info(f"DistilBERT model loaded with accuracy: {checkpoint['accuracy']:.4f}")
                
                # Load metadata
                if self.distilbert_metadata_path.exists():
                    self.distilbert_metadata = joblib.load(self.distilbert_metadata_path)
                    logger.info(f"DistilBERT metadata loaded")
                    
            else:
                logger.warning(f"DistilBERT model not found at {self.distilbert_model_path}")
                
        except Exception as e:
            logger.error(f"Error loading DistilBERT model: {e}")
            self.distilbert_model = None
            self.distilbert_tokenizer = None
    
    def _determine_active_model(self):
        """Determine which model to use as primary"""
        if self.distilbert_model is not None and self.distilbert_tokenizer is not None:
            return "distilbert"
        elif self.lr_model is not None:
            return "logistic_regression"
        else:
            return "none"
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data (same as training)"""
        if pd.isna(text) or not text:
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
    
    def extract_content_from_url(self, url: str) -> Dict[str, str]:
        """Extract title and content from a URL"""
        try:
            # Set headers to mimic a real browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else ""
            
            # Also try meta og:title
            if not title:
                og_title = soup.find('meta', property='og:title')
                title = og_title.get('content', '').strip() if og_title else ""
            
            # Extract main content
            content = ""
            
            # Try different content selectors
            content_selectors = [
                'article',
                '.article-content',
                '.content',
                '.post-content',
                '.entry-content',
                '.story-body',
                '.article-body',
                'main',
                '.main-content'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    # Remove script and style elements
                    for script in content_elem(["script", "style"]):
                        script.decompose()
                    content = content_elem.get_text()
                    break
            
            # If no specific content found, try paragraphs
            if not content:
                paragraphs = soup.find_all('p')
                content = ' '.join([p.get_text() for p in paragraphs])
            
            # Clean up content
            content = ' '.join(content.split())
            
            return {
                'title': title,
                'content': content,
                'url': url
            }
            
        except requests.RequestException as e:
            logger.error(f"Error fetching URL {url}: {e}")
            raise ValueError(f"Could not fetch content from URL: {e}")
        except Exception as e:
            logger.error(f"Error parsing content from URL {url}: {e}")
            raise ValueError(f"Could not parse content from URL: {e}")
    
    def predict_text(self, title: str = "", content: str = "") -> Dict[str, Any]:
        """Predict if given text is fake or real news using the best available model"""
        
        # Combine title and content
        combined_text = f"{title} {content}".strip()
        
        if not combined_text:
            raise ValueError("No text provided for prediction")
        
        # Try DistilBERT first (if available)
        if self.active_model == "distilbert":
            try:
                result = self._predict_with_distilbert(combined_text)
                result['model_used'] = 'distilbert'
                result['fallback_used'] = False
                return result
            except Exception as e:
                logger.warning(f"DistilBERT prediction failed: {e}. Falling back to Logistic Regression.")
                
        # Fallback to Logistic Regression
        if self.lr_model is not None:
            try:
                result = self._predict_with_lr(combined_text)
                result['model_used'] = 'logistic_regression'
                result['fallback_used'] = (self.active_model == "distilbert")
                return result
            except Exception as e:
                logger.error(f"Logistic Regression prediction also failed: {e}")
                
        raise RuntimeError("No working models available for prediction")
    
    def _predict_with_distilbert(self, text: str) -> Dict[str, Any]:
        """Predict using DistilBERT model"""
        if self.distilbert_model is None or self.distilbert_tokenizer is None:
            raise RuntimeError("DistilBERT model not loaded")
        
        # Get max length from metadata or use default
        max_length = 256
        if self.distilbert_metadata:
            max_length = self.distilbert_metadata.get('max_length', 256)
        
        # Tokenize text
        encoding = self.distilbert_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.distilbert_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1).item()
            
            # Convert to numpy for JSON serialization
            fake_prob = float(probabilities[0][0].cpu())
            real_prob = float(probabilities[0][1].cpu())
        
        # Calculate confidence and create result
        confidence = max(fake_prob, real_prob)
        label = "Real" if prediction == 1 else "Fake"
        
        result = {
            'prediction': label,
            'confidence': confidence,
            'fake_probability': fake_prob,
            'real_probability': real_prob,
            'reliability_score': confidence,
            'text_length': len(text),
            'model_info': self.distilbert_metadata or {}
        }
        
        # Add interpretation
        if confidence > 0.9:
            result['interpretation'] = f"Very high confidence: This appears to be {label.upper()} news"
        elif confidence > 0.8:
            result['interpretation'] = f"High confidence: This appears to be {label.upper()} news"
        elif confidence > 0.6:
            result['interpretation'] = f"Moderate confidence: This likely appears to be {label.upper()} news"
        else:
            result['interpretation'] = "Low confidence: Uncertain classification - manual review recommended"
        
        return result
    
    def _predict_with_lr(self, text: str) -> Dict[str, Any]:
        """Predict using Logistic Regression model (fallback)"""
        if self.lr_model is None:
            raise RuntimeError("Logistic Regression model not loaded")
        
        # Preprocess text (same as training)
        processed_text = self.preprocess_text(text)
        
        if not processed_text:
            raise ValueError("Text contains no meaningful content after preprocessing")
        
        # Make prediction
        prediction = self.lr_model.predict([processed_text])[0]
        prediction_proba = self.lr_model.predict_proba([processed_text])[0]
        
        # Calculate confidence
        confidence = float(max(prediction_proba))
        
        # Determine label
        label = "Real" if prediction == 1 else "Fake"
        
        # Calculate reliability score
        fake_prob = float(prediction_proba[0])
        real_prob = float(prediction_proba[1])
        
        result = {
            'prediction': label,
            'confidence': confidence,
            'fake_probability': fake_prob,
            'real_probability': real_prob,
            'reliability_score': confidence,
            'text_length': len(text),
            'processed_text_length': len(processed_text),
            'model_info': self.lr_metadata or {}
        }
        
        # Add interpretation
        if confidence > 0.8:
            result['interpretation'] = f"High confidence: This appears to be {label.upper()} news"
        elif confidence > 0.6:
            result['interpretation'] = f"Moderate confidence: This likely appears to be {label.upper()} news"
        else:
            result['interpretation'] = "Low confidence: Uncertain classification - manual review recommended"
        
        return result
    
    def predict_url(self, url: str) -> Dict[str, Any]:
        """Predict if content from a URL is fake or real news"""
        # Extract content from URL
        extracted_content = self.extract_content_from_url(url)
        
        # Make prediction
        result = self.predict_text(
            title=extracted_content['title'],
            content=extracted_content['content']
        )
        
        # Add URL info to result
        result['url'] = url
        result['extracted_title'] = extracted_content['title']
        result['extracted_content_preview'] = extracted_content['content'][:500] + "..." if len(extracted_content['content']) > 500 else extracted_content['content']
        
        return result
    
    def batch_predict(self, texts: list) -> list:
        """Predict multiple texts at once"""
        if not self.model:
            raise RuntimeError("Model not loaded. Please train the model first.")
        
        results = []
        for i, text_data in enumerate(texts):
            try:
                if isinstance(text_data, dict):
                    # If it's a dict with title and content
                    result = self.predict_text(
                        title=text_data.get('title', ''),
                        content=text_data.get('content', '')
                    )
                else:
                    # If it's just text
                    result = self.predict_text(content=str(text_data))
                
                result['index'] = i
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error predicting text {i}: {e}")
                results.append({
                    'index': i,
                    'error': str(e),
                    'prediction': None
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about all loaded models"""
        info = {
            'active_model': self.active_model,
            'models_available': [],
            'distilbert_available': DISTILBERT_AVAILABLE,
            'device': str(self.device) if self.device else 'N/A'
        }
        
        # DistilBERT info
        if self.distilbert_model is not None:
            distilbert_info = {
                'name': 'DistilBERT',
                'type': 'transformer',
                'status': 'loaded',
                'metadata': self.distilbert_metadata or {}
            }
            
            if self.distilbert_metadata:
                distilbert_info.update({
                    'accuracy': self.distilbert_metadata.get('accuracy', 'Unknown'),
                    'device_trained': self.distilbert_metadata.get('device_used', 'Unknown'),
                    'parameters': '~66M (DistilBERT-base)',
                    'max_length': self.distilbert_metadata.get('max_length', 256)
                })
            
            info['models_available'].append(distilbert_info)
            info['primary_model'] = distilbert_info
        
        # Logistic Regression info
        if self.lr_model is not None:
            lr_info = {
                'name': 'Logistic Regression',
                'type': 'traditional_ml',
                'status': 'loaded',
                'metadata': self.lr_metadata or {}
            }
            
            # Get pipeline info
            try:
                if hasattr(self.lr_model, 'steps'):
                    lr_info['pipeline_steps'] = [step[0] for step in self.lr_model.steps]
                    
                    # Get vectorizer info
                    if 'tfidf' in dict(self.lr_model.steps):
                        vectorizer = self.lr_model.named_steps['tfidf']
                        lr_info['vectorizer_features'] = getattr(vectorizer, 'max_features', None)
                        lr_info['vectorizer_ngram_range'] = getattr(vectorizer, 'ngram_range', None)
                    
                    # Get classifier info
                    if 'classifier' in dict(self.lr_model.steps):
                        classifier = self.lr_model.named_steps['classifier']
                        lr_info['classifier_type'] = type(classifier).__name__
            except Exception as e:
                logger.warning(f"Could not extract LR pipeline info: {e}")
            
            info['models_available'].append(lr_info)
            
            # Set as fallback or primary
            if self.active_model == 'logistic_regression':
                info['primary_model'] = lr_info
            else:
                info['fallback_model'] = lr_info
        
        # Overall status
        if not info['models_available']:
            info['status'] = 'No models loaded'
            info['error'] = 'No working models found'
        elif self.active_model == 'distilbert':
            info['status'] = 'DistilBERT active with LR fallback'
        elif self.active_model == 'logistic_regression':
            info['status'] = 'Logistic Regression only'
        else:
            info['status'] = 'No working models'
            info['error'] = 'Models loaded but not functional'
        
        return info

# Convenience function for quick predictions
def predict_fake_news(text: str = "", title: str = "", url: str = "") -> Dict[str, Any]:
    """Quick function to predict fake news"""
    predictor = FakeNewsPredictor()
    
    if url:
        return predictor.predict_url(url)
    else:
        return predictor.predict_text(title=title, content=text)
