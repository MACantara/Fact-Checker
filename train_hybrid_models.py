"""
Hybrid Training Script for Both DistilBERT and Logistic Regression Models
Trains both models and compares performance
"""

import sys
import os
from pathlib import Path
import logging
import argparse

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.ml.distilbert_trainer import DistilBertTrainer
from app.ml.model_trainer import FakeNewsModelTrainer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_both_models(quick_mode=False):
    """Train both DistilBERT and Logistic Regression models"""
    
    print("=" * 80)
    print("🚀 HYBRID FAKE NEWS DETECTION TRAINING")
    print("=" * 80)
    print("Training both DistilBERT and Logistic Regression models")
    print(f"Mode: {'Quick (samples)' if quick_mode else 'Full dataset'}")
    print("-" * 80)
    print()
    
    results = {}
    
    # Train DistilBERT Model
    print("🤖 TRAINING DISTILBERT MODEL")
    print("=" * 50)
    try:
        distilbert_trainer = DistilBertTrainer()
        distilbert_result = distilbert_trainer.train_model(quick_mode=quick_mode)
        results['distilbert'] = {
            'success': True,
            'accuracy': distilbert_result['best_accuracy'],
            'trainer': distilbert_trainer
        }
        print(f"✅ DistilBERT training completed! Accuracy: {distilbert_result['best_accuracy']:.4f}")
    except Exception as e:
        logger.error(f"DistilBERT training failed: {e}")
        results['distilbert'] = {
            'success': False,
            'error': str(e)
        }
        print(f"❌ DistilBERT training failed: {e}")
    
    print("\n" + "=" * 50)
    
    # Train Logistic Regression Model
    print("📊 TRAINING LOGISTIC REGRESSION MODEL")
    print("=" * 50)
    try:
        lr_trainer = FakeNewsModelTrainer()
        if quick_mode:
            lr_result = lr_trainer.train_model_quick()
        else:
            lr_result = lr_trainer.train_model()
        
        results['logistic_regression'] = {
            'success': True,
            'accuracy': lr_result['accuracy'],
            'trainer': lr_trainer
        }
        print(f"✅ Logistic Regression training completed! Accuracy: {lr_result['accuracy']:.4f}")
    except Exception as e:
        logger.error(f"Logistic Regression training failed: {e}")
        results['logistic_regression'] = {
            'success': False,
            'error': str(e)
        }
        print(f"❌ Logistic Regression training failed: {e}")
    
    # Results Summary
    print("\n" + "=" * 80)
    print("📊 TRAINING RESULTS SUMMARY")
    print("=" * 80)
    
    successful_models = []
    
    # DistilBERT Results
    if results['distilbert']['success']:
        print(f"🤖 DistilBERT:")
        print(f"   Status: ✅ Success")
        print(f"   Accuracy: {results['distilbert']['accuracy']:.4f} ({results['distilbert']['accuracy']*100:.1f}%)")
        print(f"   Model Type: Transformer (66M parameters)")
        print(f"   Device: GPU optimized")
        successful_models.append(('DistilBERT', results['distilbert']['accuracy']))
    else:
        print(f"🤖 DistilBERT:")
        print(f"   Status: ❌ Failed")
        print(f"   Error: {results['distilbert']['error']}")
    
    print()
    
    # Logistic Regression Results
    if results['logistic_regression']['success']:
        print(f"📊 Logistic Regression:")
        print(f"   Status: ✅ Success")
        print(f"   Accuracy: {results['logistic_regression']['accuracy']:.4f} ({results['logistic_regression']['accuracy']*100:.1f}%)")
        print(f"   Model Type: Statistical (TF-IDF + LogReg)")
        print(f"   Device: CPU optimized")
        successful_models.append(('Logistic Regression', results['logistic_regression']['accuracy']))
    else:
        print(f"📊 Logistic Regression:")
        print(f"   Status: ❌ Failed")
        print(f"   Error: {results['logistic_regression']['error']}")
    
    # Performance Comparison
    if len(successful_models) > 1:
        print("\n" + "-" * 50)
        print("🏆 PERFORMANCE COMPARISON")
        print("-" * 50)
        
        # Sort by accuracy
        successful_models.sort(key=lambda x: x[1], reverse=True)
        
        best_model, best_accuracy = successful_models[0]
        print(f"🥇 Best Model: {best_model} ({best_accuracy:.4f})")
        
        for i, (model_name, accuracy) in enumerate(successful_models[1:], 1):
            diff = best_accuracy - accuracy
            print(f"🥈 #{i+1}: {model_name} ({accuracy:.4f}) [-{diff:.4f}]")
        
        print(f"\n💡 Recommendation:")
        if best_model == 'DistilBERT':
            print("   Use DistilBERT as primary with Logistic Regression as fallback")
        else:
            print("   Use Logistic Regression as primary (DistilBERT may need more training)")
    
    elif len(successful_models) == 1:
        model_name, accuracy = successful_models[0]
        print(f"\n🎯 Single successful model: {model_name} ({accuracy:.4f})")
        print("   Consider this as your primary model")
    
    else:
        print(f"\n❌ No models trained successfully!")
        print("   Check the error messages above and resolve issues")
        return False
    
    # Final Instructions
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)
    print("Next steps:")
    print("1. Test the models using the web interface")
    print("2. Run predictions to verify both models work")
    print("3. Check the model info in the 'About' page")
    print("4. The system will automatically use the best available model")
    print("\nTo test the models:")
    print("   python run.py")
    print("   Then visit: http://localhost:5000/fake-news/check")
    print()
    
    return len(successful_models) > 0

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Train hybrid fake news detection models')
    parser.add_argument('--quick', '-q', action='store_true', 
                       help='Quick training mode with sample data')
    parser.add_argument('--distilbert-only', action='store_true',
                       help='Train only DistilBERT model')
    parser.add_argument('--lr-only', action='store_true',
                       help='Train only Logistic Regression model')
    
    args = parser.parse_args()
    
    if args.distilbert_only:
        print("Training DistilBERT only...")
        try:
            trainer = DistilBertTrainer()
            result = trainer.train_model(quick_mode=args.quick)
            print(f"DistilBERT training completed! Accuracy: {result['best_accuracy']:.4f}")
            return True
        except Exception as e:
            print(f"DistilBERT training failed: {e}")
            return False
    
    elif args.lr_only:
        print("Training Logistic Regression only...")
        try:
            trainer = FakeNewsModelTrainer()
            if args.quick:
                result = trainer.train_model_quick()
            else:
                result = trainer.train_model()
            print(f"Logistic Regression training completed! Accuracy: {result['accuracy']:.4f}")
            return True
        except Exception as e:
            print(f"Logistic Regression training failed: {e}")
            return False
    
    else:
        # Train both models
        return train_both_models(quick_mode=args.quick)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
