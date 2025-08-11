from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from app.repositories.database import DatabaseRepository
from app.models import RSSFeed
from app.extensions import db
from marshmallow import Schema, fields, ValidationError
from datetime import datetime, timezone


class RSSFeedSchema(Schema):
    """Schema for RSS feed validation"""
    key = fields.Str(required=True, validate=lambda x: len(x) > 0)
    name = fields.Str(required=True, validate=lambda x: len(x) > 0)
    url = fields.Url(required=True)
    category = fields.Str(required=True, validate=lambda x: len(x) > 0)
    active = fields.Bool(missing=True)


admin_bp = Blueprint('admin', __name__, url_prefix='/admin')


@admin_bp.route('/')
def index():
    """Admin dashboard"""
    db_repo = DatabaseRepository()
    feeds = db_repo.get_all_feeds()
    
    # Get statistics
    total_feeds = len(feeds)
    active_feeds = sum(1 for feed in feeds if feed.active)
    total_articles = db_repo.get_article_count()
    categories = db_repo.get_categories()
    
    stats = {
        'total_feeds': total_feeds,
        'active_feeds': active_feeds,
        'total_articles': total_articles,
        'categories': len(categories)
    }
    
    return render_template('admin/dashboard.html', feeds=feeds, stats=stats)


@admin_bp.route('/feeds')
def feeds():
    """List all RSS feeds"""
    db_repo = DatabaseRepository()
    feeds = db_repo.get_all_feeds(active_only=False)  # Get all feeds, not just active ones
    
    # Add actual article count to each feed
    for feed in feeds:
        feed.actual_article_count = db_repo.get_article_count_by_feed(feed.key)
    
    return render_template('admin/feeds.html', feeds=feeds)


@admin_bp.route('/feeds/new')
def new_feed():
    """Form to create new RSS feed"""
    return render_template('admin/feed_form.html', feed=None, action='Create')


@admin_bp.route('/feeds/<feed_key>/edit')
def edit_feed(feed_key):
    """Form to edit RSS feed"""
    db_repo = DatabaseRepository()
    feed = db_repo.get_feed_by_key(feed_key)
    
    if not feed:
        flash(f'Feed with key "{feed_key}" not found', 'error')
        return redirect(url_for('admin.feeds'))
    
    return render_template('admin/feed_form.html', feed=feed, action='Update')


@admin_bp.route('/feeds', methods=['POST'])
def create_feed():
    """Create new RSS feed"""
    try:
        schema = RSSFeedSchema()
        feed_data = schema.load(request.form.to_dict())
        
        db_repo = DatabaseRepository()
        
        # Check if feed key already exists
        existing_feed = db_repo.get_feed_by_key(feed_data['key'])
        if existing_feed:
            flash(f'Feed with key "{feed_data["key"]}" already exists', 'error')
            return render_template('admin/feed_form.html', feed=feed_data, action='Create')
        
        # Create new feed
        new_feed = RSSFeed(
            key=feed_data['key'],
            name=feed_data['name'],
            url=feed_data['url'],
            category=feed_data['category'],
            active=feed_data['active']
        )
        
        db.session.add(new_feed)
        db.session.commit()
        
        flash(f'RSS feed "{feed_data["name"]}" created successfully', 'success')
        return redirect(url_for('admin.feeds'))
        
    except ValidationError as e:
        flash(f'Validation error: {e.messages}', 'error')
        return render_template('admin/feed_form.html', feed=request.form.to_dict(), action='Create')
    
    except Exception as e:
        flash(f'Error creating feed: {str(e)}', 'error')
        return render_template('admin/feed_form.html', feed=request.form.to_dict(), action='Create')


@admin_bp.route('/feeds/<feed_key>', methods=['POST'])
def update_feed(feed_key):
    """Update RSS feed"""
    try:
        schema = RSSFeedSchema()
        feed_data = schema.load(request.form.to_dict())
        
        db_repo = DatabaseRepository()
        feed = db_repo.get_feed_by_key(feed_key)
        
        if not feed:
            flash(f'Feed with key "{feed_key}" not found', 'error')
            return redirect(url_for('admin.feeds'))
        
        # Update feed
        feed.key = feed_data['key']
        feed.name = feed_data['name']
        feed.url = feed_data['url']
        feed.category = feed_data['category']
        feed.active = feed_data['active']
        feed.updated_at = datetime.now(timezone.utc)
        
        db.session.commit()
        
        flash(f'RSS feed "{feed.name}" updated successfully', 'success')
        return redirect(url_for('admin.feeds'))
        
    except ValidationError as e:
        flash(f'Validation error: {e.messages}', 'error')
        feed = db_repo.get_feed_by_key(feed_key)
        return render_template('admin/feed_form.html', feed=feed, action='Update')
    
    except Exception as e:
        flash(f'Error updating feed: {str(e)}', 'error')
        feed = db_repo.get_feed_by_key(feed_key)
        return render_template('admin/feed_form.html', feed=feed, action='Update')


@admin_bp.route('/feeds/<feed_key>/delete', methods=['POST'])
def delete_feed(feed_key):
    """Delete RSS feed"""
    try:
        db_repo = DatabaseRepository()
        feed = db_repo.get_feed_by_key(feed_key)
        
        if not feed:
            flash(f'Feed with key "{feed_key}" not found', 'error')
            return redirect(url_for('admin.feeds'))
        
        feed_name = feed.name
        
        # Delete associated articles first
        articles_deleted = db_repo.delete_articles_by_feed(feed_key)
        
        # Delete the feed
        db.session.delete(feed)
        db.session.commit()
        
        flash(f'RSS feed "{feed_name}" and {articles_deleted} associated articles deleted successfully', 'success')
        
    except Exception as e:
        flash(f'Error deleting feed: {str(e)}', 'error')
    
    return redirect(url_for('admin.feeds'))


@admin_bp.route('/feeds/<feed_key>/toggle', methods=['POST'])
def toggle_feed(feed_key):
    """Toggle RSS feed active status"""
    try:
        db_repo = DatabaseRepository()
        feed = db_repo.get_feed_by_key(feed_key)
        
        if not feed:
            return jsonify({'error': 'Feed not found'}), 404
        
        feed.active = not feed.active
        feed.updated_at = datetime.now(timezone.utc)
        db.session.commit()
        
        status = 'activated' if feed.active else 'deactivated'
        return jsonify({
            'success': True,
            'message': f'Feed "{feed.name}" {status}',
            'active': feed.active
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# API endpoints for CRUD operations
@admin_bp.route('/api/feeds', methods=['GET'])
def api_list_feeds():
    """API endpoint to list all feeds"""
    db_repo = DatabaseRepository()
    feeds = db_repo.get_all_feeds()
    
    feed_list = []
    for feed in feeds:
        article_count = db.session.query(db.func.count(db.text('1'))).select_from(
            db.text('article')).filter(db.text('feed_key = :key')).params(key=feed.key).scalar()
        
        feed_list.append({
            'key': feed.key,
            'name': feed.name,
            'url': feed.url,
            'category': feed.category,
            'active': feed.active,
            'article_count': article_count or 0,
            'created_at': feed.created_at.isoformat(),
            'updated_at': feed.updated_at.isoformat(),
            'last_fetched_at': feed.last_fetched_at.isoformat() if feed.last_fetched_at else None
        })
    
    return jsonify({'feeds': feed_list})


@admin_bp.route('/api/feeds', methods=['POST'])
def api_create_feed():
    """API endpoint to create new feed"""
    try:
        schema = RSSFeedSchema()
        feed_data = schema.load(request.json)
        
        db_repo = DatabaseRepository()
        
        # Check if feed key already exists
        existing_feed = db_repo.get_feed_by_key(feed_data['key'])
        if existing_feed:
            return jsonify({'error': f'Feed with key "{feed_data["key"]}" already exists'}), 400
        
        # Create new feed
        new_feed = RSSFeed(
            key=feed_data['key'],
            name=feed_data['name'],
            url=feed_data['url'],
            category=feed_data['category'],
            active=feed_data['active']
        )
        
        db.session.add(new_feed)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'RSS feed "{feed_data["name"]}" created successfully',
            'feed': {
                'key': new_feed.key,
                'name': new_feed.name,
                'url': new_feed.url,
                'category': new_feed.category,
                'active': new_feed.active
            }
        }), 201
        
    except ValidationError as e:
        return jsonify({'error': 'Validation error', 'details': e.messages}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/api/feeds/<feed_key>', methods=['PUT'])
def api_update_feed(feed_key):
    """API endpoint to update feed"""
    try:
        schema = RSSFeedSchema()
        feed_data = schema.load(request.json)
        
        db_repo = DatabaseRepository()
        feed = db_repo.get_feed_by_key(feed_key)
        
        if not feed:
            return jsonify({'error': f'Feed with key "{feed_key}" not found'}), 404
        
        # Update feed
        feed.key = feed_data['key']
        feed.name = feed_data['name']
        feed.url = feed_data['url']
        feed.category = feed_data['category']
        feed.active = feed_data['active']
        feed.updated_at = datetime.now(timezone.utc)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'RSS feed "{feed.name}" updated successfully'
        })
        
    except ValidationError as e:
        return jsonify({'error': 'Validation error', 'details': e.messages}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@admin_bp.route('/api/feeds/<feed_key>', methods=['DELETE'])
def api_delete_feed(feed_key):
    """API endpoint to delete feed"""
    try:
        db_repo = DatabaseRepository()
        feed = db_repo.get_feed_by_key(feed_key)
        
        if not feed:
            return jsonify({'error': f'Feed with key "{feed_key}" not found'}), 404
        
        feed_name = feed.name
        
        # Delete associated articles first
        articles_deleted = db_repo.delete_articles_by_feed(feed_key)
        
        # Delete the feed
        db.session.delete(feed)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'RSS feed "{feed_name}" and {articles_deleted} associated articles deleted successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
