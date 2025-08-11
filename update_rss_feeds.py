#!/usr/bin/env python3
"""
RSS Feed Update Script for Windows Task Scheduler
This script updates all RSS feeds and can be run as a scheduled task.
"""

import os
import sys
import logging
from datetime import datetime

# Add the project directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.services import RSSFeedService, SearchService, FeedUpdateService
from app.repositories import SearchRepository
from app.repositories.database import DatabaseRepository


def setup_logging():
    """Set up logging for the update script"""
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'rss_updates.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def update_rss_feeds():
    """Update all RSS feeds"""
    logger = setup_logging()
    
    try:
        logger.info("=" * 50)
        logger.info("Starting RSS feed update process")
        logger.info("=" * 50)
        
        # Create Flask app context
        app = create_app('development')
        
        with app.app_context():
            # Initialize services
            db_repo = DatabaseRepository()
            rss_service = RSSFeedService(db_repo)
            
            # Initialize search service (try Whoosh first, fallback to database)
            try:
                search_repo = SearchRepository(app.config['WHOOSH_INDEX_PATH'])
                search_service = SearchService(search_repo, db_repo)
                logger.info("Using Whoosh search service")
            except Exception as e:
                logger.warning(f"Whoosh search not available: {e}, using database search")
                search_service = SearchService(None, db_repo)
            
            update_service = FeedUpdateService(rss_service, search_service)
            
            # Get feed count before update
            all_feeds = rss_service.get_feeds_from_db()
            active_feeds = [f for f in all_feeds.values() if f.get('active', True)]
            logger.info(f"Found {len(active_feeds)} active feeds to update")
            
            # Update feeds
            logger.info("Starting feed updates...")
            results = update_service.update_all_feeds()
            
            # Process results
            successful = 0
            failed = 0
            total_articles = 0
            failed_feeds = []
            
            for result in results:
                if result['status'] == 'success':
                    successful += 1
                    articles_count = result.get('articles_indexed', 0)
                    total_articles += articles_count
                    logger.info(f"[OK] {result['feed_name']}: {articles_count} articles")
                else:
                    failed += 1
                    failed_feeds.append(result['feed_name'])
                    error_msg = result.get('message', result.get('error', 'Unknown error'))
                    logger.error(f"[FAIL] {result['feed_name']}: {error_msg}")
            
            # Summary
            logger.info("=" * 50)
            logger.info("RSS UPDATE SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Total feeds processed: {len(results)}")
            logger.info(f"Successful updates: {successful}")
            logger.info(f"Failed updates: {failed}")
            logger.info(f"Total articles indexed: {total_articles}")
            
            if failed_feeds:
                logger.warning(f"Failed feeds: {', '.join(failed_feeds)}")
            
            logger.info("RSS feed update process completed successfully")
            return True
            
    except Exception as e:
        logger.error(f"RSS feed update process failed: {str(e)}")
        logger.exception("Full error traceback:")
        return False


if __name__ == "__main__":
    success = update_rss_feeds()
    sys.exit(0 if success else 1)
