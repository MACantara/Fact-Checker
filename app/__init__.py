from flask import Flask
from config import config
from app.extensions import init_extensions
from app.blueprints import main_bp
from app.commands import register_commands
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
    
    # Import and register ML blueprints
    from app.blueprints.ml import ml_bp
    from app.blueprints.fake_news import fake_news_bp
    app.register_blueprint(ml_bp)
    app.register_blueprint(fake_news_bp)
    
    # Register CLI commands
    register_commands(app)
    
    # Note: RSS feed updates are now handled by Windows Task Scheduler
    # instead of APScheduler for more reliable background processing
    
    return app
