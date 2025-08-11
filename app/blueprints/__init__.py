from flask import Blueprint, render_template, request, jsonify, current_app
from app.services import SearchService, RSSFeedService, FeedUpdateService
from app.repositories import SearchRepository
from app.repositories.database import DatabaseRepository
from app.schemas import SearchQuerySchema
from marshmallow import ValidationError
import json
from datetime import datetime


def datetime_handler(obj):
    """JSON serializer for datetime objects"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


main_bp = Blueprint('main', __name__)

# Import admin blueprint
from .admin import admin_bp


@main_bp.route('/')
def index():
    """Home page with search interface"""
    # Get search metadata
    try:
        search_repo = SearchRepository(current_app.config['WHOOSH_INDEX_PATH'])
        search_service = SearchService(search_repo, DatabaseRepository())
    except Exception:
        # Fallback to database search only
        search_service = SearchService(None, DatabaseRepository())
    
    metadata = search_service.get_search_metadata()
    
    return render_template('index.html', metadata=metadata)


@main_bp.route('/search')
def search():
    """Search endpoint"""
    try:
        # Get query parameters
        query_data = {
            'query': request.args.get('q', ''),
            'category': request.args.get('category'),
            'source': request.args.get('source'),
            'page': int(request.args.get('page', 1)),
            'per_page': int(request.args.get('per_page', current_app.config['SEARCH_RESULTS_PER_PAGE'])),
            'sort_by': request.args.get('sort_by', 'relevance')
        }
        
        # Validate query
        if not query_data['query'].strip():
            return render_template('search_results.html', 
                                 error="Please enter a search query", 
                                 query=query_data['query'])
        
        # Perform search
        try:
            search_repo = SearchRepository(current_app.config['WHOOSH_INDEX_PATH'])
            search_service = SearchService(search_repo, DatabaseRepository())
        except Exception:
            # Fallback to database search only
            search_service = SearchService(None, DatabaseRepository())
        
        results = search_service.search_articles(query_data)
        metadata = search_service.get_search_metadata()
        
        return render_template('search_results.html', 
                             results=results, 
                             metadata=metadata,
                             query_data=query_data)
    
    except ValidationError as e:
        return render_template('search_results.html', 
                             error=f"Invalid search parameters: {e.messages}",
                             query=request.args.get('q', ''))
    
    except Exception as e:
        current_app.logger.error(f"Search error: {e}")
        return render_template('search_results.html', 
                             error="An error occurred while searching. Please try again.",
                             query=request.args.get('q', ''))


@main_bp.route('/api/search')
def api_search():
    """API endpoint for search"""
    try:
        query_data = {
            'query': request.args.get('q', ''),
            'category': request.args.get('category'),
            'source': request.args.get('source'),
            'page': int(request.args.get('page', 1)),
            'per_page': int(request.args.get('per_page', current_app.config['SEARCH_RESULTS_PER_PAGE'])),
            'sort_by': request.args.get('sort_by', 'relevance')
        }
        
        if not query_data['query'].strip():
            return jsonify({'error': 'Query parameter is required'}), 400
        
        try:
            search_repo = SearchRepository(current_app.config['WHOOSH_INDEX_PATH'])
            search_service = SearchService(search_repo, DatabaseRepository())
        except Exception:
            # Fallback to database search only
            search_service = SearchService(None, DatabaseRepository())
        
        results = search_service.search_articles(query_data)
        
        # Convert datetime objects to strings for JSON serialization
        for article in results.get('articles', []):
            if article.get('published') and isinstance(article['published'], datetime):
                article['published'] = article['published'].isoformat()
        
        return jsonify(results)
    
    except ValidationError as e:
        return jsonify({'error': 'Invalid parameters', 'details': e.messages}), 400
    
    except Exception as e:
        current_app.logger.error(f"API search error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@main_bp.route('/api/metadata')
def api_metadata():
    """API endpoint for search metadata"""
    try:
        try:
            search_repo = SearchRepository(current_app.config['WHOOSH_INDEX_PATH'])
            search_service = SearchService(search_repo, DatabaseRepository())
        except Exception:
            # Fallback to database search only
            search_service = SearchService(None, DatabaseRepository())
        
        metadata = search_service.get_search_metadata()
        
        return jsonify(metadata)
    
    except Exception as e:
        current_app.logger.error(f"API metadata error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@main_bp.route('/feeds')
def feeds():
    """Display RSS feeds status"""
    # Get feeds from database
    db_repo = DatabaseRepository()
    rss_service = RSSFeedService(db_repo)
    feed_configs = rss_service.get_feeds_from_db()
    
    # Get search metadata for stats
    try:
        search_repo = SearchRepository(current_app.config['WHOOSH_INDEX_PATH'])
        search_service = SearchService(search_repo, db_repo)
    except Exception:
        # Fallback to database search only
        search_service = SearchService(None, db_repo)
    
    metadata = search_service.get_search_metadata()
    
    return render_template('feeds.html', 
                         feeds=feed_configs, 
                         metadata=metadata)


@main_bp.route('/api/update-feeds', methods=['POST'])
def api_update_feeds():
    """API endpoint to trigger feed updates"""
    try:
        # Initialize services with database repository
        db_repo = DatabaseRepository()
        rss_service = RSSFeedService(db_repo)
        
        try:
            search_repo = SearchRepository(current_app.config['WHOOSH_INDEX_PATH'])
            search_service = SearchService(search_repo, db_repo)
        except Exception:
            # Fallback to database search only
            search_service = SearchService(None, db_repo)
        
        update_service = FeedUpdateService(rss_service, search_service)
        
        # Update feeds (will get feeds from database)
        results = update_service.update_all_feeds()
        
        # Calculate summary
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')
        total_articles = sum(r.get('articles_indexed', 0) for r in results)
        
        return jsonify({
            'status': 'completed',
            'summary': {
                'successful_feeds': successful,
                'failed_feeds': failed,
                'total_articles_indexed': total_articles
            },
            'results': results
        })
    
    except Exception as e:
        current_app.logger.error(f"Feed update error: {e}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500
