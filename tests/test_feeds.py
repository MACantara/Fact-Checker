"""Test script to update RSS feeds using the database approach"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

from app import create_app
from app.services import RSSFeedService, SearchService, FeedUpdateService
from app.repositories import SearchRepository
from app.repositories.database import DatabaseRepository


def test_feed_update():
    """Test updating a few RSS feeds"""
    app = create_app()
    
    with app.app_context():
        # Initialize services
        db_repo = DatabaseRepository()
        rss_service = RSSFeedService(db_repo)
        
        try:
            search_repo = SearchRepository(app.config['WHOOSH_INDEX_PATH'])
            search_service = SearchService(search_repo, db_repo)
        except Exception as e:
            print(f"Whoosh not available: {e}, using database search only")
            search_service = SearchService(None, db_repo)
        
        update_service = FeedUpdateService(rss_service, search_service)
        
        print("Starting RSS feed updates...")
        
        # Get first 3 feeds to test
        feeds = db_repo.get_all_feeds()
        test_feeds = feeds[:3]  # Just test first 3 feeds
        
        for feed in test_feeds:
            print(f"\nUpdating feed: {feed.name}")
            
            feed_config = {
                'name': feed.name,
                'url': feed.url,
                'category': feed.category,
                'active': feed.active
            }
            
            try:
                result = rss_service.update_feed(feed.key, feed_config)
                
                print(f"Status: {result['status']}")
                print(f"Articles found: {result['articles_count']}")
                print(f"Articles added: {result.get('articles_added', 0)}")
                
                if result['status'] == 'error':
                    print(f"Error: {result['error_message']}")
                
            except Exception as e:
                print(f"Failed to update {feed.name}: {e}")
        
        # Show final status
        print("\n" + "="*50)
        print("Final database status:")
        
        total_articles = db_repo.get_article_count()
        categories = db_repo.get_categories()
        sources = db_repo.get_sources()
        
        print(f"Total articles in database: {total_articles}")
        print(f"Categories: {categories}")
        print(f"Sources: {sources}")


if __name__ == '__main__':
    test_feed_update()
