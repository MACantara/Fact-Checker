"""
Fake News Detection API Blueprint
Provides REST API endpoints for fake news detection
"""

from flask import Blueprint, request, jsonify, current_app
from marshmallow import Schema, fields, ValidationError
from app.ml.predictor import FakeNewsPredictor
import logging

logger = logging.getLogger(__name__)

# Create blueprint
ml_bp = Blueprint('ml', __name__, url_prefix='/api/ml')

# Prediction schemas
class TextPredictionSchema(Schema):
    """Schema for text prediction requests"""
    title = fields.Str(missing='', allow_none=True)
    content = fields.Str(required=True, validate=lambda x: len(x.strip()) > 0)

class URLPredictionSchema(Schema):
    """Schema for URL prediction requests"""
    url = fields.Url(required=True)

class BatchPredictionSchema(Schema):
    """Schema for batch prediction requests"""
    texts = fields.List(fields.Dict(), required=True, validate=lambda x: len(x) > 0 and len(x) <= 100)

# Global predictor instance
predictor = None

def get_predictor():
    """Get or create predictor instance"""
    global predictor
    if predictor is None:
        try:
            predictor = FakeNewsPredictor()
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            raise
    return predictor

@ml_bp.route('/predict/text', methods=['POST'])
def predict_text():
    """
    Predict if given text is fake or real news
    
    POST /api/ml/predict/text
    {
        "title": "Optional news title",
        "content": "News content to analyze"
    }
    """
    try:
        # Validate request data
        schema = TextPredictionSchema()
        data = schema.load(request.get_json() or {})
        
        # Get predictor
        pred = get_predictor()
        
        # Make prediction
        result = pred.predict_text(
            title=data.get('title', ''),
            content=data['content']
        )
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': 'Validation error',
            'details': e.messages
        }), 400
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': 'Invalid input',
            'details': str(e)
        }), 400
        
    except RuntimeError as e:
        return jsonify({
            'success': False,
            'error': 'Model error',
            'details': str(e)
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in predict_text: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@ml_bp.route('/predict/url', methods=['POST'])
def predict_url():
    """
    Predict if content from a URL is fake or real news
    
    POST /api/ml/predict/url
    {
        "url": "https://example.com/news-article"
    }
    """
    try:
        # Validate request data
        schema = URLPredictionSchema()
        data = schema.load(request.get_json() or {})
        
        # Get predictor
        pred = get_predictor()
        
        # Make prediction
        result = pred.predict_url(data['url'])
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': 'Validation error',
            'details': e.messages
        }), 400
        
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': 'Invalid input or URL error',
            'details': str(e)
        }), 400
        
    except RuntimeError as e:
        return jsonify({
            'success': False,
            'error': 'Model error',
            'details': str(e)
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in predict_url: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@ml_bp.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict multiple texts at once
    
    POST /api/ml/predict/batch
    {
        "texts": [
            {"title": "Title 1", "content": "Content 1"},
            {"title": "Title 2", "content": "Content 2"},
            "Just content string",
            ...
        ]
    }
    """
    try:
        # Validate request data
        schema = BatchPredictionSchema()
        data = schema.load(request.get_json() or {})
        
        # Get predictor
        pred = get_predictor()
        
        # Make predictions
        results = pred.batch_predict(data['texts'])
        
        # Calculate summary statistics
        successful_predictions = [r for r in results if 'prediction' in r and r['prediction'] is not None]
        fake_count = sum(1 for r in successful_predictions if r['prediction'] == 'Fake')
        real_count = sum(1 for r in successful_predictions if r['prediction'] == 'Real')
        error_count = len(results) - len(successful_predictions)
        
        summary = {
            'total_texts': len(results),
            'successful_predictions': len(successful_predictions),
            'fake_count': fake_count,
            'real_count': real_count,
            'error_count': error_count,
            'fake_percentage': (fake_count / len(successful_predictions) * 100) if successful_predictions else 0,
            'real_percentage': (real_count / len(successful_predictions) * 100) if successful_predictions else 0
        }
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': summary
        })
        
    except ValidationError as e:
        return jsonify({
            'success': False,
            'error': 'Validation error',
            'details': e.messages
        }), 400
        
    except Exception as e:
        logger.error(f"Unexpected error in predict_batch: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500

@ml_bp.route('/model/info', methods=['GET'])
def model_info():
    """
    Get information about the loaded model
    
    GET /api/ml/model/info
    """
    try:
        pred = get_predictor()
        info = pred.get_model_info()
        
        return jsonify({
            'success': True,
            'model_info': info
        })
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to get model info',
            'details': str(e)
        }), 500

@ml_bp.route('/model/health', methods=['GET'])
def model_health():
    """
    Check if the model is loaded and working
    
    GET /api/ml/model/health
    """
    try:
        pred = get_predictor()
        
        # Test with a simple prediction
        test_result = pred.predict_text(
            title="Test",
            content="This is a simple test to check if the model is working properly."
        )
        
        return jsonify({
            'success': True,
            'status': 'healthy',
            'model_loaded': True,
            'test_prediction': test_result['prediction'],
            'test_confidence': test_result['confidence']
        })
        
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
        return jsonify({
            'success': False,
            'status': 'unhealthy',
            'model_loaded': False,
            'error': str(e)
        }), 500

@ml_bp.route('/predict/demo', methods=['GET'])
def demo_predictions():
    """
    Get demo predictions for sample texts
    
    GET /api/ml/predict/demo
    """
    try:
        pred = get_predictor()
        
        # Demo samples
        samples = [
            {
                'title': 'Scientists Discover Cure for All Diseases',
                'content': 'A team of researchers has discovered a miracle pill that can cure all known diseases in just 24 hours. The pill is made from a rare plant found only in the Amazon rainforest.',
                'description': 'Obviously fake medical claim'
            },
            {
                'title': 'Philippine GDP Growth Reaches 6.2% in Q3',
                'content': 'The Philippines recorded a 6.2% gross domestic product growth in the third quarter, driven by strong consumer spending and infrastructure investments, according to the Philippine Statistics Authority.',
                'description': 'Realistic economic news'
            },
            {
                'title': 'Local Man Builds Flying Car in Garage',
                'content': 'A 35-year-old mechanic from Manila has successfully built a flying car using spare parts from his garage. The vehicle can fly up to 1000 feet and travel at speeds of 200 mph.',
                'description': 'Unrealistic technology claim'
            }
        ]
        
        results = []
        for sample in samples:
            try:
                prediction = pred.predict_text(
                    title=sample['title'],
                    content=sample['content']
                )
                
                results.append({
                    'title': sample['title'],
                    'content': sample['content'][:100] + '...',
                    'description': sample['description'],
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'interpretation': prediction['interpretation']
                })
                
            except Exception as e:
                results.append({
                    'title': sample['title'],
                    'description': sample['description'],
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'demo_predictions': results
        })
        
    except Exception as e:
        logger.error(f"Error in demo predictions: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to generate demo predictions',
            'details': str(e)
        }), 500

# Error handlers
@ml_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/api/ml/predict/text',
            '/api/ml/predict/url', 
            '/api/ml/predict/batch',
            '/api/ml/model/info',
            '/api/ml/model/health',
            '/api/ml/predict/demo'
        ]
    }), 404

@ml_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405

@ml_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500
