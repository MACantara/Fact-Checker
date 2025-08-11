"""
Database backup CLI commands for Flask application
"""

import click
import os
import sys
import shutil
import gzip
import logging
from pathlib import Path
from datetime import datetime, timedelta
from flask.cli import with_appcontext
from flask import current_app
from app.extensions import db
from app.models import RSSFeed, Article, FeedUpdateLog


@click.group()
def backup():
    """Database backup management commands"""
    pass


@backup.command()
@click.option('--compress/--no-compress', default=True, help='Compress backup files')
@with_appcontext
def create(compress):
    """Create a database backup"""
    click.echo("Creating database backup...")
    backup_path = backup_sqlite(compress)
    
    if backup_path:
        click.echo(f"✓ Backup created successfully: {backup_path}")
        
        # Show backup info
        backups = list_backups()
        if backups:
            latest = backups[0]
            click.echo(f"  Size: {latest['size_mb']} MB")
            click.echo(f"  Type: {latest['database_type']}")
            click.echo(f"  Compressed: {latest['compressed']}")
    else:
        click.echo("✗ Backup creation failed", err=True)


@backup.command()
@with_appcontext
def list():
    """List all available backups"""
    backups = list_backups()
    
    if backups:
        click.echo(f"{'Filename':<40} {'Size (MB)':<10} {'Created':<20} {'Type':<12}")
        click.echo("-" * 82)
        for backup in backups:
            created_str = backup['created'].strftime('%Y-%m-%d %H:%M')
            click.echo(f"{backup['filename']:<40} {backup['size_mb']:<10} "
                      f"{created_str:<20} {backup['database_type']:<12}")
        
        click.echo(f"\nTotal backups: {len(backups)}")
    else:
        click.echo("No backups found")


@backup.command()
@click.option('--keep-days', default=30, help='Days to keep backups')
@click.option('--keep-count', default=10, help='Number of recent backups to always keep')
@with_appcontext
def cleanup(keep_days, keep_count):
    """Clean up old backup files"""
    click.echo(f"Cleaning up backups older than {keep_days} days (keeping last {keep_count})...")
    deleted_count = cleanup_old_backups(keep_days, keep_count)
    
    if deleted_count > 0:
        click.echo(f"✓ Removed {deleted_count} old backup files")
    else:
        click.echo("No old backup files to remove")


@backup.command()
@click.argument('backup_path')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@with_appcontext
def restore(backup_path, confirm):
    """Restore database from backup file (SQLite only)"""
    if not confirm:
        click.confirm(
            f"This will replace the current database with the backup from {backup_path}. "
            "Are you sure you want to continue?", 
            abort=True
        )
    
    # Get database file path
    db_uri = db.engine.url
    if not db_uri.drivername.startswith('sqlite'):
        click.echo("✗ Restore currently only supports SQLite databases", err=True)
        return
    
    backup_file = Path(backup_path)
    if not backup_file.exists():
        click.echo(f"✗ Backup file not found: {backup_path}", err=True)
        return
    
    db_path = db_uri.database
    
    try:
        # Create backup of current database
        current_backup = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(db_path, current_backup)
        click.echo(f"Current database backed up to: {current_backup}")
        
        # Restore from backup
        if backup_path.endswith('.gz'):
            # Decompress and restore
            with gzip.open(backup_file, 'rb') as f_in:
                with open(db_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            # Direct copy
            shutil.copy2(backup_file, db_path)
        
        click.echo(f"✓ Database restored from: {backup_path}")
        
    except Exception as e:
        click.echo(f"✗ Restore failed: {str(e)}", err=True)
        # Try to restore original database if it exists
        if os.path.exists(current_backup):
            shutil.copy2(current_backup, db_path)
            click.echo("Original database restored after failed restore")


def get_backup_dir():
    """Get backup directory path"""
    backup_dir = Path('backups')
    backup_dir.mkdir(exist_ok=True)
    return backup_dir


def setup_backup_logging():
    """Set up logging for backup operations"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / 'database_backups.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


def backup_sqlite(compress=True):
    """Create backup of SQLite database"""
    logger = setup_backup_logging()
    backup_dir = get_backup_dir()
    
    # Get database file path
    db_uri = db.engine.url
    if not db_uri.drivername.startswith('sqlite'):
        raise ValueError("This method only supports SQLite databases")
    
    db_path = db_uri.database
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return None
    
    # Create backup filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f"backup_sqlite_{timestamp}.db"
    
    if compress:
        backup_filename += ".gz"
    
    backup_path = backup_dir / backup_filename
    
    try:
        logger.info(f"Starting SQLite backup: {db_path} -> {backup_path}")
        
        if compress:
            # Backup with compression
            with open(db_path, 'rb') as f_in:
                with gzip.open(backup_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            # Simple file copy
            shutil.copy2(db_path, backup_path)
        
        # Verify backup
        backup_size = backup_path.stat().st_size
        logger.info(f"Backup completed successfully: {backup_size} bytes")
        
        return str(backup_path)
        
    except Exception as e:
        logger.error(f"Backup failed: {str(e)}")
        if backup_path.exists():
            backup_path.unlink()  # Clean up failed backup
        return None


def list_backups():
    """List all available backups"""
    backup_dir = get_backup_dir()
    backups = []
    
    for backup_file in backup_dir.glob('backup_*'):
        if backup_file.is_file():
            stat = backup_file.stat()
            
            # Parse backup info from filename
            filename = backup_file.name
            parts = filename.replace('.gz', '').replace('.sql', '').replace('.db', '').split('_')
            
            if len(parts) >= 3:
                db_type = parts[1]
                timestamp_str = '_'.join(parts[2:])
                
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                except ValueError:
                    timestamp = datetime.fromtimestamp(stat.st_mtime)
            else:
                db_type = 'unknown'
                timestamp = datetime.fromtimestamp(stat.st_mtime)
            
            backups.append({
                'filename': filename,
                'path': str(backup_file),
                'size': stat.st_size,
                'size_mb': round(stat.st_size / (1024 * 1024), 2),
                'created': timestamp,
                'database_type': db_type,
                'compressed': filename.endswith('.gz')
            })
    
    # Sort by creation time (newest first)
    backups.sort(key=lambda x: x['created'], reverse=True)
    return backups


def cleanup_old_backups(keep_days=30, keep_count=10):
    """Clean up old backup files"""
    logger = setup_backup_logging()
    backups = list_backups()
    cutoff_date = datetime.now() - timedelta(days=keep_days)
    
    deleted_count = 0
    
    # Keep the most recent backups regardless of age
    backups_to_keep = backups[:keep_count]
    backups_to_check = backups[keep_count:]
    
    for backup in backups_to_check:
        if backup['created'] < cutoff_date:
            try:
                Path(backup['path']).unlink()
                logger.info(f"Deleted old backup: {backup['filename']}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete backup {backup['filename']}: {str(e)}")
    
    logger.info(f"Cleanup completed: {deleted_count} old backups removed")
    return deleted_count


def get_database_stats():
    """Get database statistics"""
    try:
        stats = {
            'feeds_count': db.session.query(RSSFeed).count(),
            'articles_count': db.session.query(Article).count(),
            'update_logs_count': db.session.query(FeedUpdateLog).count(),
            'active_feeds': db.session.query(RSSFeed).filter_by(active=True).count(),
            'backup_timestamp': datetime.now().isoformat()
        }
        
        return stats
        
    except Exception as e:
        return {'error': str(e)}


@backup.command()
@with_appcontext
def stats():
    """Show database statistics"""
    stats = get_database_stats()
    
    click.echo("Database Statistics:")
    click.echo("-" * 30)
    
    for key, value in stats.items():
        if key != 'backup_timestamp':
            click.echo(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Show backup summary
    backups = list_backups()
    if backups:
        total_size = sum(b['size'] for b in backups)
        total_size_mb = round(total_size / (1024 * 1024), 2)
        
        click.echo(f"\nBackup Summary:")
        click.echo(f"  Total Backups: {len(backups)}")
        click.echo(f"  Total Size: {total_size_mb} MB")
        click.echo(f"  Latest Backup: {backups[0]['created'].strftime('%Y-%m-%d %H:%M')}")


def register_backup_commands(app):
    """Register backup commands with Flask app"""
    app.cli.add_command(backup)
