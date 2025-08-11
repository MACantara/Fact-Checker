from flask import Flask
from config import config
from app.extensions import init_extensions, scheduler
from app.blueprints import main_bp
from app.commands import register_commands
from app.services import RSSFeedService, SearchService, FeedUpdateService
from app.repositories import SearchRepository
from app.repositories.database import DatabaseRepository
import os


def create_app(config_name=None):
    """Application factory pattern"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Load environment variables
    if os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv()
    
    # Initialize extensions (database, scheduler, etc.)
    init_extensions(app)
    
    # Register blueprints
    app.register_blueprint(main_bp)
    
    # Import and register admin blueprint
    from app.blueprints.admin import admin_bp
    app.register_blueprint(admin_bp)
    
    # Register CLI commands
    register_commands(app)
    
    # Set up scheduled tasks
    setup_scheduled_tasks(app)
    
    return app


def setup_scheduled_tasks(app):
    """Set up scheduled RSS feed updates"""
    
    def update_feeds_job():
        """Background job to update RSS feeds"""
        with app.app_context():
            try:
                # Initialize services with database repository
                db_repo = DatabaseRepository()
                rss_service = RSSFeedService(db_repo)
                
                # Initialize search service (try Whoosh first, fallback to database)
                try:
                    search_repo = SearchRepository(app.config['WHOOSH_INDEX_PATH'])
                    search_service = SearchService(search_repo, db_repo)
                except Exception as e:
                    app.logger.warning(f"Whoosh search not available: {e}, using database search")
                    search_service = SearchService(None, db_repo)
                
                update_service = FeedUpdateService(rss_service, search_service)
                
                # Update feeds (will get feeds from database)
                results = update_service.update_all_feeds()
                
                # Log results
                successful = sum(1 for r in results if r['status'] == 'success')
                failed = sum(1 for r in results if r['status'] == 'error')
                total_articles = sum(r.get('articles_indexed', 0) for r in results)
                
                app.logger.info(f"RSS update completed: {successful} successful, {failed} failed, {total_articles} articles indexed")
                
            except Exception as e:
                app.logger.error(f"Scheduled RSS update failed: {e}")
    
    # Schedule the job to run every configured interval
    if scheduler:
        scheduler.add_job(
            func=update_feeds_job,
            trigger='interval',
            minutes=app.config['RSS_UPDATE_INTERVAL'],
            id='rss_update_job',
            name='Update RSS Feeds',
            replace_existing=True
        )
