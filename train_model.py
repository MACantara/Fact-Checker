"""
Standalone script to train the fake news detection model
Run this script to train the ML model using the WELFake dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.ml.model_trainer import FakeNewsModelTrainer
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main(quick=False):
    """Train the fake news detection model"""
    if quick:
        print("🚀 Starting QUICK Fake News Detection Model Training")
        print("⚡ Training with 10K sample for fast testing")
    else:
        print("🤖 Starting Fake News Detection Model Training")
        print("📊 Training with full dataset (72K+ records)")
    
    print("=" * 60)
    
    try:
        # Initialize trainer
        trainer = FakeNewsModelTrainer()
        
        # Train the model
        print("📚 Training Logistic Regression model...")
        if quick:
            result = trainer.train_model_quick()
        else:
            result = trainer.train_model()
        
        # Display results
        print("\n" + "=" * 60)
        print("📊 TRAINING RESULTS")
        if quick:
            print("⚡ (Quick Mode - 10K sample)")
        print("=" * 60)
        
        print(f"\n🔍 LOGISTIC REGRESSION:")
        print(f"   Accuracy: {result['accuracy']:.4f}")
        print(f"   Cross-validation: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")
        
        print(f"\n� Model Accuracy: {trainer.best_score:.4f}")
        
        # Final evaluation
        print("\n📈 Performing final evaluation...")
        detailed_report = trainer.evaluate_model_details()
        if detailed_report:
            print(f"🤖 Final Model: Logistic Regression")
            print(f"🎯 Final Accuracy: {detailed_report['accuracy']:.4f}")
            
            real_metrics = detailed_report['classification_report']['1']
            fake_metrics = detailed_report['classification_report']['0']
            
            print(f"📰 Real News - Precision: {real_metrics['precision']:.3f}, Recall: {real_metrics['recall']:.3f}, F1: {real_metrics['f1-score']:.3f}")
            print(f"🚫 Fake News - Precision: {fake_metrics['precision']:.3f}, Recall: {fake_metrics['recall']:.3f}, F1: {fake_metrics['f1-score']:.3f}")
        
        print("\n" + "=" * 60)
        if quick:
            print("✅ QUICK MODEL TRAINING COMPLETED!")
            print("⚡ Model is functional but may have reduced accuracy")
            print("🎯 For production use, run: python train_model.py")
        else:
            print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        # Detailed evaluation
        print("\n" + "=" * 60)
        print("📈 DETAILED EVALUATION")
        print("=" * 60)
        
        detailed_report = trainer.evaluate_model_details()
        if detailed_report:
            print(f"🤖 Final Model: Logistic Regression")
            print(f"🎯 Final Accuracy: {detailed_report['accuracy']:.4f}")
            
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
            
            print(f"\n📄 Sample Predictions:")
            for i, sample in enumerate(detailed_report['sample_predictions'][:3]):
                print(f"\nSample {i+1}:")
                print(f"   Text: {sample['text']}")
                print(f"   Actual: {'Real' if sample['actual'] == 1 else 'Fake'}")
                print(f"   Predicted: {'Real' if sample['predicted'] == 1 else 'Fake'}")
                print(f"   Confidence: {sample['confidence']:.3f}")
        
        print("\n" + "=" * 60)
        if quick:
            print("✅ QUICK MODEL TRAINING COMPLETED!")
            print("⚡ Model is functional but may have reduced accuracy")
            print("🎯 For production use, run: python train_model.py")
        else:
            print("✅ MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📁 Model saved to: app/ml/models/fake_news_detector.joblib")
        print("📁 Metadata saved to: app/ml/models/model_metadata.joblib")
        print("\n🚀 You can now use the model for predictions!")
        print("💡 Try: flask ml predict 'Your news text here'")
        print("💡 Or visit: http://localhost:5000/fake-news")
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        print("💡 Make sure the dataset file exists at: datasets/WELFake_Dataset.csv")
        print("💡 Install required packages: pip install scikit-learn pandas nltk")
        return 1
    
    return 0

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
