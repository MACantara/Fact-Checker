from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from app.models import Base

# Global extensions
db = SQLAlchemy(model_class=Base)
migrate = Migrate()
scheduler = None


def init_extensions(app):
    """Initialize all Flask extensions"""
    # Initialize database
    db.init_app(app)
    migrate.init_app(app, db)
    
    # Initialize scheduler
    init_scheduler(app)


def init_scheduler(app):
    """Initialize the APScheduler"""
    global scheduler
    
    if scheduler is None:
        executors = {
            'default': ThreadPoolExecutor(10)
        }
        
        job_defaults = {
            'coalesce': False,
            'max_instances': 3
        }
        
        scheduler = BackgroundScheduler(
            executors=executors,
            job_defaults=job_defaults,
            timezone='UTC'
        )
        
        scheduler.start()
        
        # Register shutdown handler
        import atexit
        atexit.register(lambda: scheduler.shutdown())
    
    app.scheduler = scheduler
    return scheduler


def get_scheduler():
    """Get the scheduler instance"""
    return scheduler
