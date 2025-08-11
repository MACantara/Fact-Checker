"""Database migration script for converting RSS feed configs to database"""

import os
import sys
from datetime import datetime, timezone

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import create_app
from app.extensions import db
from app.models import RSSFeed, Article, FeedUpdateLog
from config import Config


def init_database():
    """Initialize the database with tables"""
    print("Creating database tables...")
    
    # Create all tables
    db.create_all()
    print("Database tables created successfully!")


def migrate_rss_feeds():
    """Migrate RSS feed configurations from config.py to database"""
    print("Migrating RSS feed configurations to database...")
    
    feeds_added = 0
    feeds_updated = 0
    
    for feed_key, feed_config in Config.DEFAULT_RSS_FEEDS.items():
        # Check if feed already exists
        existing_feed = db.session.query(RSSFeed).filter_by(key=feed_key).first()
        
        if existing_feed:
            # Update existing feed
            existing_feed.name = feed_config['name']
            existing_feed.url = feed_config['url']
            existing_feed.category = feed_config['category']
            existing_feed.active = feed_config.get('active', True)
            existing_feed.updated_at = datetime.now(timezone.utc)
            feeds_updated += 1
            print(f"Updated feed: {feed_key}")
        else:
            # Create new feed
            new_feed = RSSFeed(
                key=feed_key,
                name=feed_config['name'],
                url=feed_config['url'],
                category=feed_config['category'],
                active=feed_config.get('active', True)
            )
            db.session.add(new_feed)
            feeds_added += 1
            print(f"Added feed: {feed_key}")
    
    # Commit changes
    db.session.commit()
    
    print(f"Migration completed: {feeds_added} feeds added, {feeds_updated} feeds updated")


def show_feed_status():
    """Show current RSS feed status in database"""
    print("\nCurrent RSS Feeds in Database:")
    print("-" * 80)
    
    feeds = db.session.query(RSSFeed).all()
    
    if not feeds:
        print("No feeds found in database.")
        return
    
    for feed in feeds:
        status = "Active" if feed.active else "Inactive"
        # Count articles for this feed
        article_count = db.session.query(Article).filter_by(feed_key=feed.key).count()
        print(f"Key: {feed.key}")
        print(f"Name: {feed.name}")
        print(f"URL: {feed.url}")
        print(f"Category: {feed.category}")
        print(f"Status: {status}")
        print(f"Articles: {article_count}")
        print(f"Last Updated: {feed.updated_at}")
        print(f"Last Fetch: {feed.last_fetched_at or 'Never'}")
        print("-" * 80)


def reset_database():
    """Reset database by dropping and recreating all tables"""
    print("WARNING: This will delete all data in the database!")
    confirm = input("Are you sure? Type 'yes' to continue: ")
    
    if confirm.lower() == 'yes':
        print("Dropping all tables...")
        db.drop_all()
        print("Tables dropped successfully!")
        
        print("Recreating tables...")
        db.create_all()
        print("Tables recreated successfully!")
        
        # Migrate RSS feeds
        migrate_rss_feeds()
    else:
        print("Operation cancelled.")


def main():
    """Main migration script"""
    app = create_app()
    
    with app.app_context():
        if len(sys.argv) > 1:
            command = sys.argv[1]
            
            if command == 'init':
                init_database()
                migrate_rss_feeds()
            elif command == 'migrate':
                migrate_rss_feeds()
            elif command == 'status':
                show_feed_status()
            elif command == 'reset':
                reset_database()
            else:
                print(f"Unknown command: {command}")
                print("Available commands: init, migrate, status, reset")
        else:
            print("Database Migration Script")
            print("Usage: python migrate_db.py <command>")
            print("\nCommands:")
            print("  init    - Initialize database and migrate RSS feeds")
            print("  migrate - Migrate RSS feeds from config to database")
            print("  status  - Show current RSS feed status")
            print("  reset   - Reset database (WARNING: deletes all data)")


if __name__ == '__main__':
    main()
