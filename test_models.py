"""
Quick test script for the hybrid fake news detection system
Tests both DistilBERT and Logistic Regression models
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from app.ml.predictor import FakeNewsPredictor

def test_sample_texts():
    """Test with sample fake and real news texts"""
    
    # Sample texts for testing
    test_cases = [
        {
            "title": "Scientists Discover Cure for All Diseases Using Magic Crystals",
            "content": "A team of researchers claims they have found a universal cure for all diseases using magical healing crystals found in a secret cave. The crystals reportedly emit special energy that can heal any illness instantly.",
            "expected": "Fake",
            "description": "Obviously fake medical claim"
        },
        {
            "title": "Local School District Announces New STEM Program",
            "content": "The Springfield School District announced today that they will be implementing a new Science, Technology, Engineering, and Mathematics (STEM) program starting next fall. The program will provide students with hands-on learning opportunities in robotics, coding, and environmental science.",
            "expected": "Real",
            "description": "Realistic local news"
        },
        {
            "title": "Aliens Land in Times Square, Demand Pizza",
            "content": "Extraterrestrial visitors reportedly landed their spaceship in the middle of Times Square yesterday evening, causing massive traffic delays. According to witnesses, the aliens' first words were 'Take us to your pizza', leading to an immediate diplomatic crisis over the best pizza toppings.",
            "expected": "Fake",
            "description": "Absurd claim"
        },
        {
            "title": "Federal Reserve Raises Interest Rates by 0.25%",
            "content": "The Federal Reserve announced today that it is raising the federal funds rate by 0.25 percentage points to combat inflation. The decision was unanimous among voting members and brings the target rate to 5.25-5.50%. Fed Chair Jerome Powell cited continued strong labor market conditions and persistent inflation concerns as reasons for the increase.",
            "expected": "Real",
            "description": "Standard economic news"
        }
    ]
    
    return test_cases

def run_tests():
    """Run comprehensive tests"""
    print("🧪 HYBRID FAKE NEWS DETECTION SYSTEM TEST")
    print("=" * 60)
    
    # Initialize predictor
    try:
        predictor = FakeNewsPredictor()
        print("✅ Predictor initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        return False
    
    # Get model info
    print("\n📊 MODEL INFORMATION")
    print("-" * 30)
    model_info = predictor.get_model_info()
    
    print(f"Active Model: {model_info.get('active_model', 'Unknown')}")
    print(f"Device: {model_info.get('device', 'Unknown')}")
    print(f"Status: {model_info.get('status', 'Unknown')}")
    print(f"DistilBERT Available: {model_info.get('distilbert_available', False)}")
    print(f"Models Loaded: {len(model_info.get('models_available', []))}")
    
    if model_info.get('models_available'):
        for model in model_info['models_available']:
            print(f"  - {model['name']}: {model['status']}")
            if model.get('metadata', {}).get('accuracy'):
                print(f"    Accuracy: {model['metadata']['accuracy']:.1%}")
    
    # Test predictions
    print(f"\n🔍 PREDICTION TESTS")
    print("-" * 30)
    
    test_cases = test_sample_texts()
    correct_predictions = 0
    total_tests = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}/{total_tests}: {test_case['description']}")
        print(f"Expected: {test_case['expected']}")
        
        try:
            result = predictor.predict_text(
                title=test_case['title'],
                content=test_case['content']
            )
            
            prediction = result['prediction']
            confidence = result['confidence']
            model_used = result.get('model_used', 'Unknown')
            fallback_used = result.get('fallback_used', False)
            
            print(f"Predicted: {prediction}")
            print(f"Confidence: {confidence:.1%}")
            print(f"Model Used: {model_used}")
            
            if fallback_used:
                print("⚠️ Fallback model was used")
            
            # Check if prediction matches expected
            if prediction == test_case['expected']:
                print("✅ Correct prediction!")
                correct_predictions += 1
            else:
                print("❌ Incorrect prediction")
                
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
    
    # Results summary
    print(f"\n📈 TEST RESULTS SUMMARY")
    print("=" * 40)
    accuracy = correct_predictions / total_tests if total_tests > 0 else 0
    print(f"Correct Predictions: {correct_predictions}/{total_tests}")
    print(f"Test Accuracy: {accuracy:.1%}")
    
    if accuracy >= 0.75:
        print("🎉 Great! The model is performing well on test cases")
    elif accuracy >= 0.5:
        print("⚠️ Moderate performance. Model may need more training or different examples")
    else:
        print("❌ Poor performance. Check model training and data quality")
    
    # Performance analysis
    print(f"\n💡 PERFORMANCE INSIGHTS")
    print("-" * 30)
    
    if model_info.get('active_model') == 'distilbert':
        print("🤖 Using DistilBERT (Advanced AI):")
        print("  ✅ Better context understanding")
        print("  ✅ Higher accuracy potential")
        print("  ⚡ GPU acceleration (if available)")
    elif model_info.get('active_model') == 'logistic_regression':
        print("📊 Using Logistic Regression (Traditional ML):")
        print("  ✅ Fast predictions")
        print("  ✅ Reliable performance") 
        print("  ⚡ CPU efficient")
    else:
        print("❌ No active model detected")
        return False
    
    return accuracy >= 0.5  # Consider success if >50% accuracy

def test_urls():
    """Test URL analysis (optional)"""
    print(f"\n🌐 URL ANALYSIS TEST (Optional)")
    print("-" * 30)
    
    # Note: URL testing requires internet connection and may be slow
    sample_urls = [
        "https://www.bbc.com/news",  # Legitimate news source
    ]
    
    try:
        predictor = FakeNewsPredictor()
        print("URL testing available. Skipping for quick test.")
        print("To test URLs, use the web interface at /fake-news/check")
        return True
    except:
        print("URL testing not available")
        return False

def main():
    """Main test function"""
    print("Starting hybrid fake news detection test...\n")
    
    try:
        # Run basic tests
        success = run_tests()
        
        # Optional URL test info
        test_urls()
        
        # Final recommendations
        print(f"\n🎯 RECOMMENDATIONS")
        print("=" * 30)
        print("1. If tests pass: The system is ready for use!")
        print("2. If accuracy is low: Consider retraining with more data")
        print("3. Test the web interface: python run.py")
        print("4. Visit: http://localhost:5000/fake-news/check")
        print("5. For detailed model info: /fake-news/about")
        
        if success:
            print(f"\n✅ All tests completed successfully!")
            print("🚀 Your hybrid fake news detection system is ready!")
        else:
            print(f"\n⚠️ Some tests failed. Check the model training.")
            print("💡 Try retraining: python train_hybrid_models.py --quick")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
