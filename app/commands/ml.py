"""
Flask CLI commands for machine learning operations
"""

import click
from flask.cli import with_appcontext
from pathlib import Path
import json
from app.ml.model_trainer import FakeNewsModelTrainer
from app.ml.predictor import FakeNewsPredictor

@click.group()
def ml():
    """Machine learning commands for fake news detection."""
    pass

@ml.command()
@click.option('--dataset', default='datasets/WELFake_Dataset.csv', help='Path to the dataset CSV file')
@click.option('--test-size', default=0.2, help='Test set size (0.0-1.0)')
@click.option('--quick', is_flag=True, help='Quick training with sample data (10K records)')
@with_appcontext
def train(dataset, test_size, quick):
    """Train the fake news detection model."""
    if quick:
        click.echo("🚀 Starting QUICK fake news detection model training (10K sample)...")
        click.echo("💡 For full training, use: flask ml train")
    else:
        click.echo("🤖 Starting fake news detection model training...")
    
    try:
        trainer = FakeNewsModelTrainer(dataset_path=dataset)
        
        # Train models
        if quick:
            click.echo("📚 Quick training with Logistic Regression...")
            result = trainer.train_model_quick(test_size=test_size)
        else:
            click.echo("📚 Training Logistic Regression model...")
            result = trainer.train_model(test_size=test_size)
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("📊 TRAINING RESULTS")
        if quick:
            click.echo("⚡ (Quick Mode - 10K sample)")
        click.echo("="*60)
        
        click.echo(f"\n🔍 LOGISTIC REGRESSION:")
        click.echo(f"   Accuracy: {result['accuracy']:.4f}")
        click.echo(f"   Cross-validation: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")
        
        click.echo(f"\n� Model Accuracy: {trainer.best_score:.4f}")
        
        if quick:
            click.echo("\n💡 For full training with all data:")
            click.echo("   flask ml train")
        
        # Detailed evaluation
        click.echo("\n" + "="*60)
        click.echo("📈 DETAILED EVALUATION")
        click.echo("="*60)
        
        detailed_report = trainer.evaluate_model_details()
        if detailed_report:
            click.echo(f"🤖 Final Model: Logistic Regression")
            click.echo(f"🎯 Final Accuracy: {detailed_report['accuracy']:.4f}")
            
            real_metrics = detailed_report['classification_report']['1']
            click.echo(f"📰 Real News Metrics:")
            click.echo(f"   Precision: {real_metrics['precision']:.4f}")
            click.echo(f"   Recall: {real_metrics['recall']:.4f}")
            click.echo(f"   F1-Score: {real_metrics['f1-score']:.4f}")
            
            fake_metrics = detailed_report['classification_report']['0']
            click.echo(f"🚫 Fake News Metrics:")
            click.echo(f"   Precision: {fake_metrics['precision']:.4f}")
            click.echo(f"   Recall: {fake_metrics['recall']:.4f}")
            click.echo(f"   F1-Score: {fake_metrics['f1-score']:.4f}")
        
        click.echo("\n✅ Model training completed successfully!")
        click.echo("📁 Model saved to: app/ml/models/fake_news_detector.joblib")
        
        if quick:
            click.echo("\n⚡ Quick training completed. Model is functional but may have lower accuracy.")
            click.echo("🎯 For production use, run full training: flask ml train")
        
    except Exception as e:
        click.echo(f"❌ Error during training: {e}", err=True)
        raise click.ClickException(str(e))

@ml.command()
@click.argument('text')
@click.option('--title', default='', help='Title of the news article')
@with_appcontext
def predict(text, title):
    """Predict if given text is fake or real news."""
    try:
        predictor = FakeNewsPredictor()
        result = predictor.predict_text(title=title, content=text)
        
        click.echo("\n" + "="*60)
        click.echo("🔍 FAKE NEWS PREDICTION")
        click.echo("="*60)
        
        # Display prediction
        if result['prediction'] == 'Real':
            click.echo(f"✅ Prediction: {result['prediction']}")
        else:
            click.echo(f"🚫 Prediction: {result['prediction']}")
        
        click.echo(f"🎯 Confidence: {result['confidence']:.4f}")
        click.echo(f"📊 Real Probability: {result['real_probability']:.4f}")
        click.echo(f"📊 Fake Probability: {result['fake_probability']:.4f}")
        click.echo(f"💬 Interpretation: {result['interpretation']}")
        
        if title:
            click.echo(f"\n📰 Title: {title}")
        click.echo(f"📝 Text Preview: {text[:200]}{'...' if len(text) > 200 else ''}")
        
    except Exception as e:
        click.echo(f"❌ Error during prediction: {e}", err=True)
        raise click.ClickException(str(e))

@ml.command()
@click.argument('url')
@with_appcontext
def predict_url(url):
    """Predict if content from a URL is fake or real news."""
    try:
        click.echo(f"🌐 Extracting content from: {url}")
        
        predictor = FakeNewsPredictor()
        result = predictor.predict_url(url)
        
        click.echo("\n" + "="*60)
        click.echo("🔍 FAKE NEWS PREDICTION (URL)")
        click.echo("="*60)
        
        # Display prediction
        if result['prediction'] == 'Real':
            click.echo(f"✅ Prediction: {result['prediction']}")
        else:
            click.echo(f"🚫 Prediction: {result['prediction']}")
        
        click.echo(f"🎯 Confidence: {result['confidence']:.4f}")
        click.echo(f"📊 Real Probability: {result['real_probability']:.4f}")
        click.echo(f"📊 Fake Probability: {result['fake_probability']:.4f}")
        click.echo(f"💬 Interpretation: {result['interpretation']}")
        
        click.echo(f"\n🌐 URL: {result['url']}")
        click.echo(f"📰 Extracted Title: {result['extracted_title']}")
        click.echo(f"📝 Content Preview: {result['extracted_content_preview']}")
        
    except Exception as e:
        click.echo(f"❌ Error during URL prediction: {e}", err=True)
        raise click.ClickException(str(e))

@ml.command()
@with_appcontext
def model_info():
    """Display information about the loaded model."""
    try:
        predictor = FakeNewsPredictor()
        info = predictor.get_model_info()
        
        click.echo("\n" + "="*60)
        click.echo("🤖 MODEL INFORMATION")
        click.echo("="*60)
        
        if info.get('model_loaded'):
            click.echo("✅ Model Status: Loaded")
            click.echo(f"📁 Model Path: {info['model_path']}")
            
            if 'metadata' in info and info['metadata']:
                metadata = info['metadata']
                click.echo(f"🏷️ Model Type: {metadata.get('model_type', 'Unknown')}")
                click.echo(f"🎯 Model Accuracy: {metadata.get('accuracy', 'Unknown')}")
                click.echo(f"📊 Features: {metadata.get('features', 'Unknown')}")
                click.echo(f"🔤 Vectorizer: {metadata.get('vectorizer', 'Unknown')}")
            
            if 'pipeline_steps' in info:
                click.echo(f"🔧 Pipeline Steps: {', '.join(info['pipeline_steps'])}")
            
            if 'classifier_type' in info:
                click.echo(f"🤖 Classifier: {info['classifier_type']}")
            
            if 'vectorizer_features' in info:
                click.echo(f"📈 Max Features: {info['vectorizer_features']}")
            
            if 'vectorizer_ngram_range' in info:
                click.echo(f"📝 N-gram Range: {info['vectorizer_ngram_range']}")
        else:
            click.echo("❌ Model Status: Not Loaded")
            click.echo("💡 Run 'flask ml train' to train a model first")
        
    except Exception as e:
        click.echo(f"❌ Error getting model info: {e}", err=True)
        raise click.ClickException(str(e))

@ml.command()
@click.option('--format', 'output_format', type=click.Choice(['text', 'json']), default='text', help='Output format')
@with_appcontext
def test_samples(output_format):
    """Test the model with sample news texts."""
    try:
        predictor = FakeNewsPredictor()
        
        # Sample test cases
        test_cases = [
            {
                'title': 'Scientists Discover New Planet in Solar System',
                'content': 'Researchers at NASA have announced the discovery of a new planet in our solar system, located beyond Pluto. The planet, temporarily named Planet X, is estimated to be twice the size of Earth.',
                'expected': 'This is likely fake - no new planets have been discovered in our solar system recently'
            },
            {
                'title': 'Philippine President Signs New Economic Reform Bill',
                'content': 'President Ferdinand Marcos Jr. signed into law a comprehensive economic reform bill aimed at boosting foreign investment and creating jobs. The legislation reduces bureaucratic red tape and provides tax incentives for new businesses.',
                'expected': 'This could be real - typical government policy news'
            },
            {
                'title': 'Local Man Wins Lottery Three Times in One Week',
                'content': 'A 45-year-old man from Quezon City has reportedly won the lottery three times in one week, earning a total of 500 million pesos. Mathematical experts are calling it impossible.',
                'expected': 'This is likely fake - statistically impossible'
            },
            {
                'title': 'Climate Change Report Shows Rising Sea Levels',
                'content': 'A new scientific report published in Nature journal shows that sea levels are rising faster than previously predicted due to climate change. The study analyzed data from satellite measurements over the past 30 years.',
                'expected': 'This could be real - typical scientific reporting'
            }
        ]
        
        results = []
        
        click.echo("\n" + "="*60)
        click.echo("🧪 TESTING MODEL WITH SAMPLE CASES")
        click.echo("="*60)
        
        for i, case in enumerate(test_cases, 1):
            try:
                result = predictor.predict_text(title=case['title'], content=case['content'])
                
                if output_format == 'json':
                    results.append({
                        'test_case': i,
                        'title': case['title'],
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'expected': case['expected']
                    })
                else:
                    click.echo(f"\n📝 Test Case {i}:")
                    click.echo(f"Title: {case['title']}")
                    if result['prediction'] == 'Real':
                        click.echo(f"✅ Prediction: {result['prediction']} (Confidence: {result['confidence']:.3f})")
                    else:
                        click.echo(f"🚫 Prediction: {result['prediction']} (Confidence: {result['confidence']:.3f})")
                    click.echo(f"💭 Expected: {case['expected']}")
                    click.echo(f"🎯 Interpretation: {result['interpretation']}")
                
            except Exception as e:
                click.echo(f"❌ Error testing case {i}: {e}")
                if output_format == 'json':
                    results.append({
                        'test_case': i,
                        'error': str(e)
                    })
        
        if output_format == 'json':
            click.echo(json.dumps(results, indent=2))
        
    except Exception as e:
        click.echo(f"❌ Error during testing: {e}", err=True)
        raise click.ClickException(str(e))

def init_app(app):
    """Initialize ML commands with Flask app"""
    app.cli.add_command(ml)
