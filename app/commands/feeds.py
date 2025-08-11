"""CLI commands for RSS feed management"""

import click
from flask.cli import with_appcontext
from app.extensions import db
from app.models import RSSFeed, Article
from app.repositories.database import DatabaseRepository
from app.services import RSSFeedService, SearchService, FeedUpdateService
from app.repositories import SearchRepository
from datetime import datetime, timezone
from tabulate import tabulate


@click.group()
def feeds():
    """RSS feed management commands"""
    pass


@feeds.command()
@click.option('--active-only', is_flag=True, help='Show only active feeds')
@with_appcontext
def list(active_only):
    """List all RSS feeds"""
    db_repo = DatabaseRepository()
    feeds = db_repo.get_all_feeds(active_only=active_only)
    
    if not feeds:
        click.echo("No feeds found.")
        return
    
    # Prepare data for table
    headers = ['Key', 'Name', 'Category', 'Status', 'Articles', 'Last Fetch', 'Last Status']
    rows = []
    
    for feed in feeds:
        article_count = db.session.query(Article).filter_by(feed_key=feed.key).count()
        last_fetch = feed.last_fetched_at.strftime('%Y-%m-%d %H:%M') if feed.last_fetched_at else 'Never'
        status = 'Active' if feed.active else 'Inactive'
        fetch_status = feed.last_fetch_status or 'Unknown'
        
        rows.append([
            feed.key,
            feed.name[:30] + '...' if len(feed.name) > 30 else feed.name,
            feed.category,
            status,
            article_count,
            last_fetch,
            fetch_status
        ])
    
    click.echo(tabulate(rows, headers=headers, tablefmt='grid'))
    click.echo(f"\nTotal feeds: {len(feeds)}")


@feeds.command()
@click.argument('key')
@click.argument('name')
@click.argument('url')
@click.argument('category')
@click.option('--active/--inactive', default=True, help='Set feed as active or inactive')
@with_appcontext
def add(key, name, url, category, active):
    """Add a new RSS feed"""
    db_repo = DatabaseRepository()
    
    # Check if feed already exists
    existing_feed = db_repo.get_feed_by_key(key)
    if existing_feed:
        click.echo(f"Error: Feed with key '{key}' already exists.", err=True)
        return
    
    try:
        # Create new feed
        new_feed = RSSFeed(
            key=key,
            name=name,
            url=url,
            category=category,
            active=active
        )
        
        db.session.add(new_feed)
        db.session.commit()
        
        click.echo(f"✓ RSS feed '{name}' added successfully with key '{key}'")
        
    except Exception as e:
        click.echo(f"Error adding feed: {e}", err=True)


@feeds.command()
@click.argument('key')
@click.option('--name', help='New name for the feed')
@click.option('--url', help='New URL for the feed')
@click.option('--category', help='New category for the feed')
@click.option('--active/--inactive', default=None, help='Set feed as active or inactive')
@with_appcontext
def update(key, name, url, category, active):
    """Update an existing RSS feed"""
    db_repo = DatabaseRepository()
    feed = db_repo.get_feed_by_key(key)
    
    if not feed:
        click.echo(f"Error: Feed with key '{key}' not found.", err=True)
        return
    
    try:
        # Update fields if provided
        if name:
            feed.name = name
        if url:
            feed.url = url
        if category:
            feed.category = category
        if active is not None:
            feed.active = active
        
        feed.updated_at = datetime.now(timezone.utc)
        db.session.commit()
        
        click.echo(f"✓ RSS feed '{feed.name}' updated successfully")
        
    except Exception as e:
        click.echo(f"Error updating feed: {e}", err=True)


@feeds.command()
@click.argument('key')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@with_appcontext
def delete(key, confirm):
    """Delete an RSS feed and all its articles"""
    db_repo = DatabaseRepository()
    feed = db_repo.get_feed_by_key(key)
    
    if not feed:
        click.echo(f"Error: Feed with key '{key}' not found.", err=True)
        return
    
    article_count = db.session.query(Article).filter_by(feed_key=key).count()
    
    if not confirm:
        click.echo(f"This will delete the feed '{feed.name}' and {article_count} associated articles.")
        if not click.confirm('Are you sure you want to continue?'):
            click.echo('Operation cancelled.')
            return
    
    try:
        # Delete articles first
        db.session.query(Article).filter_by(feed_key=key).delete()
        
        # Delete feed
        db.session.delete(feed)
        db.session.commit()
        
        click.echo(f"✓ RSS feed '{feed.name}' and {article_count} articles deleted successfully")
        
    except Exception as e:
        click.echo(f"Error deleting feed: {e}", err=True)


@feeds.command()
@click.argument('key')
@with_appcontext
def show(key):
    """Show detailed information about a specific feed"""
    db_repo = DatabaseRepository()
    feed = db_repo.get_feed_by_key(key)
    
    if not feed:
        click.echo(f"Error: Feed with key '{key}' not found.", err=True)
        return
    
    article_count = db.session.query(Article).filter_by(feed_key=key).count()
    
    click.echo(f"\n{'='*50}")
    click.echo(f"RSS Feed Details: {feed.name}")
    click.echo(f"{'='*50}")
    click.echo(f"Key: {feed.key}")
    click.echo(f"Name: {feed.name}")
    click.echo(f"URL: {feed.url}")
    click.echo(f"Category: {feed.category}")
    click.echo(f"Status: {'Active' if feed.active else 'Inactive'}")
    click.echo(f"Articles: {article_count}")
    click.echo(f"Created: {feed.created_at.strftime('%Y-%m-%d %H:%M:%S') if feed.created_at else 'Unknown'}")
    click.echo(f"Updated: {feed.updated_at.strftime('%Y-%m-%d %H:%M:%S') if feed.updated_at else 'Unknown'}")
    click.echo(f"Last Fetched: {feed.last_fetched_at.strftime('%Y-%m-%d %H:%M:%S') if feed.last_fetched_at else 'Never'}")
    click.echo(f"Last Fetch Status: {feed.last_fetch_status or 'Unknown'}")
    
    if feed.last_fetch_error:
        click.echo(f"Last Error: {feed.last_fetch_error}")
    
    if feed.feed_title:
        click.echo(f"Feed Title: {feed.feed_title}")
    
    if feed.feed_description:
        click.echo(f"Feed Description: {feed.feed_description[:100]}..." if len(feed.feed_description) > 100 else feed.feed_description)


@feeds.command()
@click.argument('key', required=False)
@click.option('--all', is_flag=True, help='Update all active feeds')
@with_appcontext
def update_feed(key, all):
    """Update articles from RSS feed(s)"""
    if not key and not all:
        click.echo("Error: Please specify a feed key or use --all flag", err=True)
        return
    
    db_repo = DatabaseRepository()
    rss_service = RSSFeedService(db_repo)
    
    # Initialize search service
    try:
        from flask import current_app
        search_repo = SearchRepository(current_app.config['WHOOSH_INDEX_PATH'])
        search_service = SearchService(search_repo, db_repo)
    except Exception:
        search_service = SearchService(None, db_repo)
    
    update_service = FeedUpdateService(rss_service, search_service)
    
    if all:
        click.echo("Updating all active feeds...")
        results = update_service.update_all_feeds()
        
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')
        total_articles = sum(r.get('articles_added', 0) for r in results)
        
        click.echo(f"\nUpdate Summary:")
        click.echo(f"✓ Successful feeds: {successful}")
        click.echo(f"✗ Failed feeds: {failed}")
        click.echo(f"📰 Total articles added: {total_articles}")
        
        # Show failed feeds
        failed_feeds = [r for r in results if r['status'] == 'error']
        if failed_feeds:
            click.echo(f"\nFailed Feeds:")
            for feed_result in failed_feeds:
                click.echo(f"  • {feed_result['feed_name']}: {feed_result['error_message']}")
    
    else:
        feed = db_repo.get_feed_by_key(key)
        if not feed:
            click.echo(f"Error: Feed with key '{key}' not found.", err=True)
            return
        
        if not feed.active:
            click.echo(f"Warning: Feed '{feed.name}' is inactive. Updating anyway...")
        
        click.echo(f"Updating feed: {feed.name}")
        
        feed_config = {
            'name': feed.name,
            'url': feed.url,
            'category': feed.category,
            'active': feed.active
        }
        
        try:
            result = rss_service.update_feed(key, feed_config)
            
            if result['status'] == 'success':
                click.echo(f"✓ Successfully updated '{feed.name}'")
                click.echo(f"  Articles found: {result['articles_count']}")
                click.echo(f"  Articles added: {result.get('articles_added', 0)}")
            else:
                click.echo(f"✗ Failed to update '{feed.name}': {result['error_message']}", err=True)
                
        except Exception as e:
            click.echo(f"✗ Error updating feed: {e}", err=True)


@feeds.command()
@click.argument('key')
@with_appcontext
def toggle(key):
    """Toggle feed active status"""
    db_repo = DatabaseRepository()
    feed = db_repo.get_feed_by_key(key)
    
    if not feed:
        click.echo(f"Error: Feed with key '{key}' not found.", err=True)
        return
    
    try:
        feed.active = not feed.active
        feed.updated_at = datetime.now(timezone.utc)
        db.session.commit()
        
        status = 'activated' if feed.active else 'deactivated'
        click.echo(f"✓ Feed '{feed.name}' {status} successfully")
        
    except Exception as e:
        click.echo(f"Error toggling feed status: {e}", err=True)


@feeds.command()
@with_appcontext
def stats():
    """Show RSS feed statistics"""
    db_repo = DatabaseRepository()
    feeds = db_repo.get_all_feeds(active_only=False)
    
    active_feeds = sum(1 for feed in feeds if feed.active)
    total_articles = db_repo.get_article_count()
    categories = db_repo.get_categories()
    sources = db_repo.get_sources()
    
    click.echo(f"\n{'='*40}")
    click.echo(f"RSS Feed Statistics")
    click.echo(f"{'='*40}")
    click.echo(f"Total Feeds: {len(feeds)}")
    click.echo(f"Active Feeds: {active_feeds}")
    click.echo(f"Inactive Feeds: {len(feeds) - active_feeds}")
    click.echo(f"Total Articles: {total_articles}")
    click.echo(f"Categories: {len(categories)}")
    click.echo(f"Sources: {len(sources)}")
    
    # Recent activity
    recent_feeds = [f for f in feeds if f.last_fetched_at]
    recent_feeds.sort(key=lambda x: x.last_fetched_at, reverse=True)
    
    if recent_feeds:
        click.echo(f"\nRecent Activity (Last 5 updates):")
        for feed in recent_feeds[:5]:
            status_icon = "✓" if feed.last_fetch_status == 'success' else "✗"
            click.echo(f"  {status_icon} {feed.name}: {feed.last_fetched_at.strftime('%Y-%m-%d %H:%M')}")


def register_feed_commands(app):
    """Register feed management commands with the Flask app"""
    app.cli.add_command(feeds)
