"""
Fake News Prediction Service
Provides functionality to predict if news content is fake or real
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

logger = logging.getLogger(__name__)

class FakeNewsPredictor:
    """Predict if news content is fake or real"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path or Path("app/ml/models/fake_news_detector.joblib")
        self.metadata_path = Path("app/ml/models/model_metadata.joblib")
        
        # Initialize NLTK components
        self._download_nltk_data()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load model
        self.model = None
        self.metadata = None
        self.load_model()
    
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
    
    def load_model(self):
        """Load the trained model"""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
                
                if self.metadata_path.exists():
                    self.metadata = joblib.load(self.metadata_path)
                    logger.info(f"Model metadata loaded from {self.metadata_path}")
                else:
                    logger.warning("Model metadata not found")
            else:
                logger.error(f"Model file not found at {self.model_path}")
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
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
        """Predict if given text is fake or real news"""
        if not self.model:
            raise RuntimeError("Model not loaded. Please train the model first.")
        
        # Combine title and content
        combined_text = f"{title} {content}".strip()
        
        if not combined_text:
            raise ValueError("No text provided for prediction")
        
        # Preprocess text
        processed_text = self.preprocess_text(combined_text)
        
        if not processed_text:
            raise ValueError("Text contains no meaningful content after preprocessing")
        
        try:
            # Make prediction
            prediction = self.model.predict([processed_text])[0]
            prediction_proba = self.model.predict_proba([processed_text])[0]
            
            # Calculate confidence
            confidence = float(max(prediction_proba))
            
            # Determine label
            label = "Real" if prediction == 1 else "Fake"
            
            # Calculate reliability score (confidence adjusted for class balance)
            fake_prob = float(prediction_proba[0])
            real_prob = float(prediction_proba[1])
            
            result = {
                'prediction': label,
                'confidence': confidence,
                'fake_probability': fake_prob,
                'real_probability': real_prob,
                'reliability_score': confidence,
                'text_length': len(combined_text),
                'processed_text_length': len(processed_text),
                'model_info': self.metadata if self.metadata else {}
            }
            
            # Add interpretation
            if confidence > 0.8:
                if label == "Fake":
                    result['interpretation'] = "High confidence: This appears to be FAKE news"
                else:
                    result['interpretation'] = "High confidence: This appears to be REAL news"
            elif confidence > 0.6:
                if label == "Fake":
                    result['interpretation'] = "Moderate confidence: This likely appears to be FAKE news"
                else:
                    result['interpretation'] = "Moderate confidence: This likely appears to be REAL news"
            else:
                result['interpretation'] = "Low confidence: Uncertain classification - manual review recommended"
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    
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
        """Get information about the loaded model"""
        if not self.model:
            return {'error': 'No model loaded'}
        
        info = {
            'model_loaded': True,
            'model_path': str(self.model_path),
            'metadata': self.metadata if self.metadata else {}
        }
        
        # Try to get pipeline info
        try:
            if hasattr(self.model, 'steps'):
                info['pipeline_steps'] = [step[0] for step in self.model.steps]
                
                # Get vectorizer info
                if 'tfidf' in dict(self.model.steps):
                    vectorizer = self.model.named_steps['tfidf']
                    info['vectorizer_features'] = getattr(vectorizer, 'max_features', None)
                    info['vectorizer_ngram_range'] = getattr(vectorizer, 'ngram_range', None)
                
                # Get classifier info
                if 'classifier' in dict(self.model.steps):
                    classifier = self.model.named_steps['classifier']
                    info['classifier_type'] = type(classifier).__name__
        except Exception as e:
            logger.warning(f"Could not extract pipeline info: {e}")
        
        return info

# Convenience function for quick predictions
def predict_fake_news(text: str = "", title: str = "", url: str = "") -> Dict[str, Any]:
    """Quick function to predict fake news"""
    predictor = FakeNewsPredictor()
    
    if url:
        return predictor.predict_url(url)
    else:
        return predictor.predict_text(title=title, content=text)
