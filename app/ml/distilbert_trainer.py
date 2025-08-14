"""
DistilBERT Model Trainer for Fake News Detection
Optimized for NVIDIA GTX 1050 (2GB VRAM) with fallback to CPU
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging
import os
from pathlib import Path
import joblib
from tqdm import tqdm
import gc
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FakeNewsDataset(Dataset):
    """Dataset for fake news classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class DistilBertTrainer:
    """DistilBERT trainer optimized for GTX 1050 (2GB VRAM)"""
    
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path or "datasets/WELFake_Dataset.csv"
        self.preprocessed_path = "datasets/WELFake_Dataset_preprocessed_distilbert.csv"
        self.models_dir = Path("app/ml/models")
        self.models_dir.mkdir(exist_ok=True)
        
        # GPU Configuration for GTX 1050
        self.device = self._setup_device()
        logger.info(f"Using device: {self.device}")
        
        # Model configuration optimized for 2GB VRAM
        self.model_name = 'distilbert-base-uncased'
        self.max_length = 256  # Reduced from 512 to save memory
        self.batch_size = 8    # Small batch size for GTX 1050
        self.gradient_accumulation_steps = 4  # Simulate larger batch size
        self.learning_rate = 2e-5
        self.epochs = 3
        self.warmup_ratio = 0.1
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.best_model = None
        self.best_accuracy = 0.0
        
    def _setup_device(self):
        """Setup device with preference for CUDA if available"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Check GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {gpu_memory_gb:.1f} GB")
            
            # Optimize for GTX 1050 (2GB)
            if gpu_memory_gb < 3:
                logger.info("Optimizing for low VRAM GPU (GTX 1050)")
                self.max_length = 128  # Further reduce for very low VRAM
                self.batch_size = 4    # Even smaller batch size
                
        else:
            device = torch.device('cpu')
            logger.info("CUDA not available, using CPU")
            # Increase batch size for CPU training
            self.batch_size = 16
            
        return device
    
    def preprocess_data(self, force_reprocess=False):
        """Load and preprocess data with caching"""
        
        # Check if preprocessed file exists
        if (not force_reprocess and 
            Path(self.preprocessed_path).exists() and 
            Path(self.preprocessed_path).stat().st_mtime > Path(self.dataset_path).stat().st_mtime):
            
            logger.info(f"Loading preprocessed dataset from {self.preprocessed_path}")
            return pd.read_csv(self.preprocessed_path)
        
        logger.info(f"Loading and preprocessing dataset from {self.dataset_path}")
        
        # Load original dataset
        df = pd.read_csv(self.dataset_path)
        logger.info(f"Original dataset shape: {df.shape}")
        
        # Combine title and text
        df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
        
        # Clean data
        df = df[df['combined_text'].str.len() > 10]  # Remove very short texts
        df = df.dropna(subset=['label'])
        
        # Keep only necessary columns
        df_clean = df[['combined_text', 'label']].copy()
        
        # Truncate very long texts to save memory
        df_clean['combined_text'] = df_clean['combined_text'].str[:2000]
        
        # Save preprocessed data
        logger.info(f"Saving preprocessed dataset to {self.preprocessed_path}")
        df_clean.to_csv(self.preprocessed_path, index=False)
        
        logger.info(f"Label distribution:\n{df_clean['label'].value_counts()}")
        return df_clean
    
    def setup_model(self):
        """Initialize tokenizer and model"""
        logger.info(f"Loading tokenizer and model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        
        # Load model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,  # Fake (0) or Real (1)
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        logger.info("Model setup complete")
    
    def create_data_loaders(self, df, test_size=0.2, sample_size=None):
        """Create train and validation data loaders"""
        
        # Sample data if specified (for quick training)
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Using sample of {sample_size} records for training")
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['combined_text'].values,
            df['label'].values,
            test_size=test_size,
            random_state=42,
            stratify=df['label']
        )
        
        logger.info(f"Train size: {len(train_texts)}, Validation size: {len(val_texts)}")
        
        # Create datasets
        train_dataset = FakeNewsDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        val_dataset = FakeNewsDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader
    
    def train_model(self, quick_mode=False):
        """Train the DistilBERT model"""
        
        # Preprocess data
        df = self.preprocess_data()
        
        # Setup model
        self.setup_model()
        
        # Create data loaders
        sample_size = 5000 if quick_mode else None
        train_loader, val_loader = self.create_data_loaders(df, sample_size=sample_size)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=1e-8
        )
        
        total_steps = len(train_loader) * self.epochs // self.gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self.warmup_ratio),
            num_training_steps=total_steps
        )
        
        logger.info(f"Training for {self.epochs} epochs with {total_steps} steps")
        
        # Training loop
        train_losses = []
        val_accuracies = []
        
        for epoch in range(self.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.epochs}")
            
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            train_losses.append(train_loss)
            
            # Validation phase
            val_accuracy, val_report = self._validate_epoch(val_loader)
            val_accuracies.append(val_accuracy)
            
            # Save best model
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_model = self.model.state_dict().copy()
                logger.info(f"New best accuracy: {val_accuracy:.4f}")
            
            # Clear GPU cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save the best model
        self.save_model()
        
        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_accuracy': self.best_accuracy,
            'final_report': val_report
        }
    
    def _train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / self.gradient_accumulation_steps
            loss.backward()
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Update weights every gradient_accumulation_steps
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            progress_bar.set_postfix({'loss': loss.item() * self.gradient_accumulation_steps})
            
            # Clear cache periodically
            if step % 50 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        predictions = []
        true_labels = []
        
        progress_bar = tqdm(val_loader, desc='Validation', leave=False)
        
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        
        return accuracy, report
    
    def save_model(self):
        """Save the trained model"""
        if self.best_model is None:
            logger.error("No trained model to save")
            return
        
        # Save the model state dict
        model_path = self.models_dir / "distilbert_fake_news_detector.pt"
        torch.save({
            'model_state_dict': self.best_model,
            'model_name': self.model_name,
            'max_length': self.max_length,
            'accuracy': self.best_accuracy
        }, model_path)
        
        # Save tokenizer
        tokenizer_path = self.models_dir / "distilbert_tokenizer"
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save metadata
        metadata = {
            'model_type': 'distilbert',
            'model_name': self.model_name,
            'accuracy': self.best_accuracy,
            'max_length': self.max_length,
            'device_used': str(self.device),
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate
        }
        
        metadata_path = self.models_dir / "distilbert_metadata.joblib"
        joblib.dump(metadata, metadata_path)
        
        logger.info(f"DistilBERT model saved to {model_path}")
        logger.info(f"Tokenizer saved to {tokenizer_path}")
        logger.info(f"Metadata saved to {metadata_path}")
    
    def evaluate_model_details(self):
        """Generate detailed evaluation report"""
        if self.best_model is None:
            logger.error("No trained model found. Please run train_model() first.")
            return None
        
        # Load data for evaluation
        df = self.preprocess_data()
        _, val_loader = self.create_data_loaders(df, test_size=0.2)
        
        # Load best model
        self.model.load_state_dict(self.best_model)
        
        # Get detailed predictions
        accuracy, report = self._validate_epoch(val_loader)
        
        detailed_report = {
            'model_type': 'distilbert',
            'accuracy': accuracy,
            'classification_report': report,
            'device_used': str(self.device),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'max_length': self.max_length,
            'batch_size': self.batch_size
        }
        
        return detailed_report

def main(quick=False):
    """Main training function"""
    
    # Check GPU availability
    if torch.cuda.is_available():
        print("🚀 GPU-Accelerated DistilBERT Training")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("💻 CPU DistilBERT Training")
    
    print("="*60)
    
    trainer = DistilBertTrainer()
    
    try:
        # Train model
        if quick:
            logger.info("Starting quick DistilBERT training (5K sample)...")
            result = trainer.train_model(quick_mode=True)
        else:
            logger.info("Starting full DistilBERT training...")
            result = trainer.train_model(quick_mode=False)
        
        # Print results
        print("\n" + "="*60)
        print("📊 DISTILBERT TRAINING RESULTS")
        if quick:
            print("⚡ (Quick Mode - 5K sample)")
        print("="*60)
        
        print(f"\n🎯 Best Accuracy: {result['best_accuracy']:.4f}")
        print(f"📈 Training completed successfully!")
        
        # Detailed evaluation
        detailed_report = trainer.evaluate_model_details()
        if detailed_report:
            print(f"\n🤖 Model Parameters: {detailed_report['model_parameters']:,}")
            print(f"💾 Device Used: {detailed_report['device_used']}")
            print(f"📏 Max Length: {detailed_report['max_length']}")
            print(f"📦 Batch Size: {detailed_report['batch_size']}")
            
            real_metrics = detailed_report['classification_report']['1']
            print(f"\n📰 Real News Metrics:")
            print(f"   Precision: {real_metrics['precision']:.4f}")
            print(f"   Recall: {real_metrics['recall']:.4f}")
            print(f"   F1-Score: {real_metrics['f1-score']:.4f}")
            
            fake_metrics = detailed_report['classification_report']['0']
            print(f"\n🚫 Fake News Metrics:")
            print(f"   Precision: {fake_metrics['precision']:.4f}")
            print(f"   Recall: {fake_metrics['recall']:.4f}")
            print(f"   F1-Score: {fake_metrics['f1-score']:.4f}")
        
        print(f"\n✅ DistilBERT model training complete!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    import sys
    
    # Check for quick training argument
    quick_mode = '--quick' in sys.argv or '-q' in sys.argv
    
    if quick_mode:
        print("=" * 60)
        print("🚀 QUICK DISTILBERT TRAINING")
        print("=" * 60)
        print("• Using 5K sample for fast training")
        print("• Optimized for GTX 1050 (2GB VRAM)")
        print("• For full training: python distilbert_trainer.py")
        print("-" * 60)
        print()
    else:
        print("=" * 60)
        print("🤖 FULL DISTILBERT TRAINING")
        print("=" * 60)
        print("• Using complete dataset (72K+ records)")
        print("• GPU-accelerated with CUDA")
        print("• Optimized for GTX 1050 (2GB VRAM)")
        print("• For quick training: python distilbert_trainer.py --quick")
        print("-" * 60)
        print()
    
    success = main(quick=quick_mode)
    sys.exit(0 if success else 1)
