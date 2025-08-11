import click
from flask.cli import with_appcontext
from flask import current_app
from app.services import RSSFeedService, SearchService, FeedUpdateService
from app.repositories import SearchRepository
from app.repositories.database import DatabaseRepository
from .feeds import register_feed_commands
from .backup import register_backup_commands


@click.command()
@with_appcontext
def init_index():
    """Initialize the search index"""
    click.echo('Initializing search index...')
    
    search_repo = SearchRepository(current_app.config['WHOOSH_INDEX_PATH'])
    search_repo.clear_index()
    
    click.echo('Search index initialized successfully!')


@click.command()
@with_appcontext
def update_feeds():
    """Update RSS feeds and rebuild search index"""
    click.echo('Starting RSS feed update...')
    
    # Initialize services with database repository
    db_repo = DatabaseRepository()
    rss_service = RSSFeedService(db_repo)
    
    try:
        search_repo = SearchRepository(current_app.config['WHOOSH_INDEX_PATH'])
        search_service = SearchService(search_repo, db_repo)
    except Exception:
        search_service = SearchService(None, db_repo)
    
    update_service = FeedUpdateService(rss_service, search_service)
    
    # Update feeds (will get feeds from database)
    results = update_service.update_all_feeds()
    
    # Display results
    total_articles = 0
    successful_feeds = 0
    failed_feeds = 0
    
    for result in results:
        if result['status'] == 'success':
            successful_feeds += 1
            total_articles += result.get('articles_indexed', 0)
            click.echo(f"✓ {result['feed_name']}: {result['articles_indexed']} articles indexed")
        else:
            failed_feeds += 1
            click.echo(f"✗ {result['feed_name']}: {result['error_message']}")
    
    click.echo(f"\nUpdate completed:")
    click.echo(f"- Successful feeds: {successful_feeds}")
    click.echo(f"- Failed feeds: {failed_feeds}")
    click.echo(f"- Total articles indexed: {total_articles}")


@click.command()
@with_appcontext
def optimize_index():
    """Optimize the search index"""
    click.echo('Optimizing search index...')
    
    try:
        search_repo = SearchRepository(current_app.config['WHOOSH_INDEX_PATH'])
        search_repo.optimize_index()
        click.echo('Search index optimized successfully!')
    except Exception as e:
        click.echo(f'Warning: Could not optimize search index: {e}')


@click.command()
@with_appcontext
def show_stats():
    """Show search index statistics"""
    try:
        search_repo = SearchRepository(current_app.config['WHOOSH_INDEX_PATH'])
        
        article_count = search_repo.get_article_count()
        categories = search_repo.get_categories()
        sources = search_repo.get_sources()
        
        click.echo(f"Search Index Statistics:")
        click.echo(f"- Total articles: {article_count}")
        click.echo(f"- Categories: {len(categories)}")
        click.echo(f"- Sources: {len(sources)}")
        
        if categories:
            click.echo(f"\nAvailable categories:")
            for category in categories:
                click.echo(f"  - {category}")
        
        if sources:
            click.echo(f"\nAvailable sources:")
            for source in sources:
                click.echo(f"  - {source}")
    except Exception as e:
        click.echo(f'Warning: Could not get search statistics: {e}')
        
        # Fallback to database statistics
        db_repo = DatabaseRepository()
        article_count = db_repo.get_article_count()
        categories = db_repo.get_categories()
        sources = db_repo.get_sources()
        
        click.echo(f"Database Statistics:")
        click.echo(f"- Total articles: {article_count}")
        click.echo(f"- Categories: {len(categories)}")
        click.echo(f"- Sources: {len(sources)}")


def register_commands(app):
    """Register CLI commands with the Flask app"""
    app.cli.add_command(init_index)
    app.cli.add_command(update_feeds)
    app.cli.add_command(optimize_index)
    app.cli.add_command(show_stats)
    
    # Register feed management commands
    register_feed_commands(app)
    
    # Register backup management commands
    register_backup_commands(app)
