"""
Web interface for fake news detection
"""

from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, current_app
from app.ml.predictor import FakeNewsPredictor
from app.repositories import SearchRepository
from app.services import SearchService
from app.repositories.database import DatabaseRepository
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Create blueprint
fake_news_bp = Blueprint('fake_news', __name__, url_prefix='/fake-news')

@fake_news_bp.route('/')
def index():
    """Main fake news detection page"""
    try:
        predictor = FakeNewsPredictor()
        model_info = predictor.get_model_info()
    except:
        model_info = {'error': 'Model not available'}
        
    return render_template('fake_news/index.html', model_info=model_info)

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

        # Cross-check: query the site's search index for similar articles
        similar_articles = []
        try:
            # Build search service (try Whoosh first, fallback to DB)
            try:
                search_repo = SearchRepository(current_app.config['WHOOSH_INDEX_PATH'])
                search_service = SearchService(search_repo, DatabaseRepository())
            except Exception:
                search_service = SearchService(None, DatabaseRepository())

            # Create a compact query using title + leading content
            sq = f"{analyzed_title} {analyzed_content[:200]}".strip()
            if sq:
                query_data = {
                    'query': sq,
                    'category': None,
                    'source': None,
                    'page': 1,
                    'per_page': 5,
                    'sort_by': 'relevance'
                }
                search_results = search_service.search_articles(query_data)
                similar_articles = search_results.get('articles', [])

                # Convert any ISO strings for published back to datetimes for template
                for a in similar_articles:
                    pub = a.get('published')
                    if isinstance(pub, str):
                        try:
                            a['published'] = datetime.fromisoformat(pub)
                        except Exception:
                            a['published'] = None
        except Exception as e:
            # Fail silently for cross-checks; do not block the main prediction
            logger.warning(f"Cross-check search failed: {e}")

        return render_template('fake_news/result.html', 
                             result=result, 
                             source=source,
                             analyzed_title=analyzed_title,
                             analyzed_content=analyzed_content,
                             original_url=url if url else None,
                             similar_articles=similar_articles)
        
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
