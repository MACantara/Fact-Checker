"""
Web interface for fake news detection
"""

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for
from app.ml.predictor import FakeNewsPredictor
import logging

logger = logging.getLogger(__name__)

# Create blueprint
fake_news_bp = Blueprint('fake_news', __name__, url_prefix='/fake-news')

@fake_news_bp.route('/')
def index():
    """Main fake news detection page"""
    return render_template('fake_news/index.html')

@fake_news_bp.route('/check', methods=['GET', 'POST'])
def check():
    """Check if news is fake or real"""
    if request.method == 'GET':
        return render_template('fake_news/check.html')
    
    try:
        # Get form data
        title = request.form.get('title', '').strip()
        content = request.form.get('content', '').strip()
        url = request.form.get('url', '').strip()
        
        if not content and not url:
            flash('Please provide either content or a URL to analyze.', 'error')
            return render_template('fake_news/check.html')
        
        # Initialize predictor
        predictor = FakeNewsPredictor()
        
        # Make prediction
        if url:
            result = predictor.predict_url(url)
            source = 'URL'
            analyzed_title = result.get('extracted_title', '')
            analyzed_content = result.get('extracted_content_preview', '')
        else:
            result = predictor.predict_text(title=title, content=content)
            source = 'Text'
            analyzed_title = title
            analyzed_content = content[:500] + '...' if len(content) > 500 else content
        
        return render_template('fake_news/result.html', 
                             result=result, 
                             source=source,
                             analyzed_title=analyzed_title,
                             analyzed_content=analyzed_content,
                             original_url=url if url else None)
        
    except Exception as e:
        logger.error(f"Error in fake news check: {e}")
        flash(f'Error analyzing content: {str(e)}', 'error')
        return render_template('fake_news/check.html')

@fake_news_bp.route('/batch')
def batch():
    """Batch analysis page"""
    return render_template('fake_news/batch.html')

@fake_news_bp.route('/api/check', methods=['POST'])
def api_check():
    """API endpoint for fake news checking (for AJAX)"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        title = data.get('title', '').strip()
        content = data.get('content', '').strip()
        url = data.get('url', '').strip()
        
        if not content and not url:
            return jsonify({'success': False, 'error': 'Please provide either content or URL'}), 400
        
        # Initialize predictor
        predictor = FakeNewsPredictor()
        
        # Make prediction
        if url:
            result = predictor.predict_url(url)
        else:
            result = predictor.predict_text(title=title, content=content)
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        logger.error(f"Error in API check: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@fake_news_bp.route('/about')
def about():
    """About page explaining the fake news detection system"""
    try:
        predictor = FakeNewsPredictor()
        model_info = predictor.get_model_info()
    except:
        model_info = {'error': 'Model not available'}
    
    return render_template('fake_news/about.html', model_info=model_info)

@fake_news_bp.route('/examples')
def examples():
    """Examples page with sample fake and real news"""
    return render_template('fake_news/examples.html')
