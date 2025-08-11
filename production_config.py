import os
from config import ProductionConfig

class OptimizedProductionConfig(ProductionConfig):
    """Optimized production configuration"""
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY')
    WTF_CSRF_ENABLED = True
    
    # Performance
    RSS_UPDATE_INTERVAL = int(os.environ.get('RSS_UPDATE_INTERVAL', 30))
    SEARCH_RESULTS_PER_PAGE = int(os.environ.get('SEARCH_RESULTS_PER_PAGE', 20))
    
    # Caching
    SEND_FILE_MAX_AGE_DEFAULT = 31536000  # 1 year for static files
    
    # Database paths (use environment variables in production)
    WHOOSH_INDEX_PATH = os.environ.get('WHOOSH_INDEX_PATH', '/app/index')
    RSS_FEED_DATA_PATH = os.environ.get('RSS_FEED_DATA_PATH', '/app/data')
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Rate limiting (implement if needed)
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'memory://')

# Export for use in Docker
config = OptimizedProductionConfig
