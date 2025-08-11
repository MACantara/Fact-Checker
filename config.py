import os
from typing import Dict, Any


class Config:
    """Base configuration class"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    WHOOSH_INDEX_PATH = os.path.join(os.path.abspath('.'), 'index')
    RSS_FEED_DATA_PATH = os.path.join(os.path.abspath('.'), 'data')
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # RSS Feed update interval in minutes
    RSS_UPDATE_INTERVAL = int(os.environ.get('RSS_UPDATE_INTERVAL', 30))
    
    # Search configuration
    SEARCH_RESULTS_PER_PAGE = int(os.environ.get('SEARCH_RESULTS_PER_PAGE', 20))
    
    # Backup configuration
    BACKUP_ENABLED = os.environ.get('BACKUP_ENABLED', 'true').lower() == 'true'
    BACKUP_SCHEDULE_HOURS = int(os.environ.get('BACKUP_SCHEDULE_HOURS', 24))  # Backup every 24 hours
    BACKUP_KEEP_DAYS = int(os.environ.get('BACKUP_KEEP_DAYS', 30))  # Keep backups for 30 days
    BACKUP_KEEP_COUNT = int(os.environ.get('BACKUP_KEEP_COUNT', 10))  # Always keep last 10 backups
    BACKUP_COMPRESS = os.environ.get('BACKUP_COMPRESS', 'true').lower() == 'true'
    
    # Default RSS feeds
    DEFAULT_RSS_FEEDS = {
        'pna': {
            'name': 'Philippine News Agency',
            'url': 'https://syndication.pna.gov.ph/rss',
            'category': 'Government',
            'active': True
        },
        'inquirer': {
            'name': 'Philippine Daily Inquirer',
            'url': 'https://newsinfo.inquirer.net/rss',
            'category': 'News',
            'active': True
        },
        'rappler': {
            'name': 'Rappler',
            'url': 'https://www.rappler.com/rss',
            'category': 'News',
            'active': True
        },
        'philstar': {
            'name': 'Philippine Star Headlines',
            'url': 'https://www.philstar.com/rss/headlines',
            'category': 'News',
            'active': True
        },
        'philstar_nation': {
            'name': 'Philippine Star Nation',
            'url': 'https://www.philstar.com/rss/nation',
            'category': 'National',
            'active': True
        },
        'malaya': {
            'name': 'Malaya Business Insight',
            'url': 'https://malaya.com.ph/feed/',
            'category': 'Business',
            'active': True
        },
        'sunstar': {
            'name': 'SunStar Philippines',
            'url': 'https://www.sunstar.com.ph/api/v1/collections/home.rss',
            'category': 'News',
            'active': True
        },
        'interaksyon': {
            'name': 'Interaksyon',
            'url': 'https://interaksyon.philstar.com/feed/',
            'category': 'News',
            'active': True
        },
        'philstar_business': {
            'name': 'Philippine Star Business',
            'url': 'https://www.philstar.com/rss/business',
            'category': 'Business',
            'active': True
        },
        'bworld': {
            'name': 'BusinessWorld Online',
            'url': 'https://www.bworldonline.com/feed/',
            'category': 'Business',
            'active': True
        },
        'gma_news': {
            'name': 'GMA News',
            'url': 'https://data.gmanetwork.com/gno/rss/news/feed.xml',
            'category': 'News',
            'active': True
        },
        'gma_world': {
            'name': 'GMA News World',
            'url': 'https://data.gmanetwork.com/gno/rss/news/world/feed.xml',
            'category': 'World',
            'active': True
        },
        'gma_metro': {
            'name': 'GMA News Metro Manila',
            'url': 'https://data.gmanetwork.com/gno/rss/news/metro/feed.xml',
            'category': 'Metro',
            'active': True
        },
        'gma_nation': {
            'name': 'GMA News Nation',
            'url': 'https://data.gmanetwork.com/gno/rss/news/nation/feed.xml',
            'category': 'National',
            'active': True
        },
        'gma_regions': {
            'name': 'GMA News Regions',
            'url': 'https://data.gmanetwork.com/gno/rss/news/regions/feed.xml',
            'category': 'Regional',
            'active': True
        },
        'gma_special': {
            'name': 'GMA News Special Reports',
            'url': 'https://data.gmanetwork.com/gno/rss/news/specialreports/feed.xml',
            'category': 'Special Reports',
            'active': True
        },
        'manila_bulletin': {
            'name': 'Manila Bulletin',
            'url': 'https://mb.com.ph/rss',
            'category': 'News',
            'active': True
        },
        'rappler_feed': {
            'name': 'Rappler Feed',
            'url': 'https://www.rappler.com/feed/',
            'category': 'News',
            'active': True
        }
    }


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    RSS_UPDATE_INTERVAL = 5  # More frequent updates in development
    SQLALCHEMY_DATABASE_URI = 'sqlite:///dev_app.db'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'prod-secret-key-must-be-set')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///prod_app.db'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    RSS_UPDATE_INTERVAL = 1
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
