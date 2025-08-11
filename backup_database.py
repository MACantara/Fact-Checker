#!/usr/bin/env python3
"""
Database Backup System for Philippine News Search Engine
Supports SQLite and can be extended for other database types.
"""

import os
import sys
import shutil
import sqlite3
import gzip
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import tempfile

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from app.extensions import db
from config import config


class DatabaseBackupManager:
    """Manages database backups with support for multiple database types"""
    
    def __init__(self, app_config=None):
        self.app = create_app(app_config or 'development')
        self.backup_dir = Path('backups')
        self.backup_dir.mkdir(exist_ok=True)
        self.setup_logging()
        
    def setup_logging(self):
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
        
        self.logger = logging.getLogger(__name__)
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get database connection information"""
        with self.app.app_context():
            db_uri = db.engine.url
            
            return {
                'drivername': db_uri.drivername,
                'database': db_uri.database,
                'host': db_uri.host,
                'port': db_uri.port,
                'username': db_uri.username,
                'query': dict(db_uri.query) if db_uri.query else {}
            }
    
    def backup_sqlite(self, compress: bool = True) -> Optional[str]:
        """Create backup of SQLite database"""
        db_info = self.get_database_info()
        
        if not db_info['drivername'].startswith('sqlite'):
            raise ValueError("This method only supports SQLite databases")
        
        # Get database file path
        db_path = db_info['database']
        if not os.path.exists(db_path):
            self.logger.error(f"Database file not found: {db_path}")
            return None
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"backup_sqlite_{timestamp}.db"
        
        if compress:
            backup_filename += ".gz"
        
        backup_path = self.backup_dir / backup_filename
        
        try:
            self.logger.info(f"Starting SQLite backup: {db_path} -> {backup_path}")
            
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
            self.logger.info(f"Backup completed successfully: {backup_size} bytes")
            
            # Get database statistics
            stats = self.get_database_stats()
            self.logger.info(f"Database stats: {stats}")
            
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Backup failed: {str(e)}")
            if backup_path.exists():
                backup_path.unlink()  # Clean up failed backup
            return None
    
    def backup_postgresql(self, compress: bool = True) -> Optional[str]:
        """Create backup of PostgreSQL database using pg_dump"""
        db_info = self.get_database_info()
        
        if db_info['drivername'] != 'postgresql':
            raise ValueError("This method only supports PostgreSQL databases")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"backup_postgresql_{timestamp}.sql"
        
        if compress:
            backup_filename += ".gz"
        
        backup_path = self.backup_dir / backup_filename
        
        try:
            # Build pg_dump command
            cmd_parts = [
                'pg_dump',
                f"--host={db_info['host'] or 'localhost'}",
                f"--port={db_info['port'] or 5432}",
                f"--username={db_info['username']}",
                '--no-password',
                '--verbose',
                '--clean',
                '--no-acl',
                '--no-owner',
                db_info['database']
            ]
            
            import subprocess
            
            self.logger.info(f"Starting PostgreSQL backup: {backup_path}")
            
            if compress:
                # Pipe pg_dump output through gzip
                with gzip.open(backup_path, 'wt') as f:
                    process = subprocess.run(
                        cmd_parts,
                        stdout=f,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True
                    )
            else:
                # Direct output to file
                with open(backup_path, 'w') as f:
                    process = subprocess.run(
                        cmd_parts,
                        stdout=f,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True
                    )
            
            backup_size = backup_path.stat().st_size
            self.logger.info(f"PostgreSQL backup completed: {backup_size} bytes")
            
            return str(backup_path)
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"pg_dump failed: {e.stderr}")
            if backup_path.exists():
                backup_path.unlink()
            return None
        except Exception as e:
            self.logger.error(f"PostgreSQL backup failed: {str(e)}")
            if backup_path.exists():
                backup_path.unlink()
            return None
    
    def backup_mysql(self, compress: bool = True) -> Optional[str]:
        """Create backup of MySQL database using mysqldump"""
        db_info = self.get_database_info()
        
        if not db_info['drivername'].startswith('mysql'):
            raise ValueError("This method only supports MySQL databases")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"backup_mysql_{timestamp}.sql"
        
        if compress:
            backup_filename += ".gz"
        
        backup_path = self.backup_dir / backup_filename
        
        try:
            # Build mysqldump command
            cmd_parts = [
                'mysqldump',
                f"--host={db_info['host'] or 'localhost'}",
                f"--port={db_info['port'] or 3306}",
                f"--user={db_info['username']}",
                '--single-transaction',
                '--routines',
                '--triggers',
                db_info['database']
            ]
            
            import subprocess
            
            self.logger.info(f"Starting MySQL backup: {backup_path}")
            
            if compress:
                with gzip.open(backup_path, 'wt') as f:
                    process = subprocess.run(
                        cmd_parts,
                        stdout=f,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True
                    )
            else:
                with open(backup_path, 'w') as f:
                    process = subprocess.run(
                        cmd_parts,
                        stdout=f,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True
                    )
            
            backup_size = backup_path.stat().st_size
            self.logger.info(f"MySQL backup completed: {backup_size} bytes")
            
            return str(backup_path)
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"mysqldump failed: {e.stderr}")
            if backup_path.exists():
                backup_path.unlink()
            return None
        except Exception as e:
            self.logger.error(f"MySQL backup failed: {str(e)}")
            if backup_path.exists():
                backup_path.unlink()
            return None
    
    def create_backup(self, compress: bool = True) -> Optional[str]:
        """Create backup based on database type"""
        db_info = self.get_database_info()
        driver = db_info['drivername']
        
        try:
            if driver.startswith('sqlite'):
                return self.backup_sqlite(compress)
            elif driver == 'postgresql':
                return self.backup_postgresql(compress)
            elif driver.startswith('mysql'):
                return self.backup_mysql(compress)
            else:
                self.logger.error(f"Unsupported database type: {driver}")
                return None
                
        except Exception as e:
            self.logger.error(f"Backup creation failed: {str(e)}")
            return None
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.app.app_context():
            try:
                # Get table counts
                from app.models import RSSFeed, Article, FeedUpdateLog
                
                stats = {
                    'feeds_count': db.session.query(RSSFeed).count(),
                    'articles_count': db.session.query(Article).count(),
                    'update_logs_count': db.session.query(FeedUpdateLog).count(),
                    'active_feeds': db.session.query(RSSFeed).filter_by(active=True).count(),
                    'backup_timestamp': datetime.now().isoformat()
                }
                
                return stats
                
            except Exception as e:
                self.logger.error(f"Failed to get database stats: {str(e)}")
                return {'error': str(e)}
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups"""
        backups = []
        
        for backup_file in self.backup_dir.glob('backup_*'):
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
    
    def cleanup_old_backups(self, keep_days: int = 30, keep_count: int = 10) -> int:
        """Clean up old backup files"""
        backups = self.list_backups()
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        deleted_count = 0
        
        # Keep the most recent backups regardless of age
        backups_to_keep = backups[:keep_count]
        backups_to_check = backups[keep_count:]
        
        for backup in backups_to_check:
            if backup['created'] < cutoff_date:
                try:
                    Path(backup['path']).unlink()
                    self.logger.info(f"Deleted old backup: {backup['filename']}")
                    deleted_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to delete backup {backup['filename']}: {str(e)}")
        
        self.logger.info(f"Cleanup completed: {deleted_count} old backups removed")
        return deleted_count
    
    def restore_sqlite_backup(self, backup_path: str) -> bool:
        """Restore SQLite database from backup"""
        backup_file = Path(backup_path)
        if not backup_file.exists():
            self.logger.error(f"Backup file not found: {backup_path}")
            return False
        
        db_info = self.get_database_info()
        if not db_info['drivername'].startswith('sqlite'):
            self.logger.error("This method only supports SQLite databases")
            return False
        
        db_path = db_info['database']
        
        try:
            # Create backup of current database
            current_backup = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(db_path, current_backup)
            self.logger.info(f"Current database backed up to: {current_backup}")
            
            # Restore from backup
            if backup_path.endswith('.gz'):
                # Decompress and restore
                with gzip.open(backup_file, 'rb') as f_in:
                    with open(db_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                # Direct copy
                shutil.copy2(backup_file, db_path)
            
            self.logger.info(f"Database restored from: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {str(e)}")
            # Try to restore original database if it exists
            if os.path.exists(current_backup):
                shutil.copy2(current_backup, db_path)
                self.logger.info("Original database restored after failed restore")
            return False


def main():
    """Command-line interface for backup operations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Database Backup Manager')
    parser.add_argument('action', choices=['backup', 'list', 'cleanup', 'restore', 'stats'],
                       help='Action to perform')
    parser.add_argument('--compress', action='store_true', default=True,
                       help='Compress backup files (default: True)')
    parser.add_argument('--no-compress', action='store_false', dest='compress',
                       help='Do not compress backup files')
    parser.add_argument('--keep-days', type=int, default=30,
                       help='Days to keep backups during cleanup (default: 30)')
    parser.add_argument('--keep-count', type=int, default=10,
                       help='Number of recent backups to always keep (default: 10)')
    parser.add_argument('--config', default='development',
                       help='Configuration to use (default: development)')
    parser.add_argument('--backup-path', help='Path to backup file for restore')
    
    args = parser.parse_args()
    
    backup_manager = DatabaseBackupManager(args.config)
    
    if args.action == 'backup':
        backup_path = backup_manager.create_backup(args.compress)
        if backup_path:
            print(f"Backup created successfully: {backup_path}")
            sys.exit(0)
        else:
            print("Backup failed")
            sys.exit(1)
    
    elif args.action == 'list':
        backups = backup_manager.list_backups()
        if backups:
            print(f"{'Filename':<40} {'Size (MB)':<10} {'Created':<20} {'Type':<12}")
            print("-" * 82)
            for backup in backups:
                print(f"{backup['filename']:<40} {backup['size_mb']:<10} "
                     f"{backup['created'].strftime('%Y-%m-%d %H:%M'):<20} "
                     f"{backup['database_type']:<12}")
        else:
            print("No backups found")
    
    elif args.action == 'cleanup':
        deleted = backup_manager.cleanup_old_backups(args.keep_days, args.keep_count)
        print(f"Cleaned up {deleted} old backup files")
    
    elif args.action == 'restore':
        if not args.backup_path:
            print("Error: --backup-path is required for restore action")
            sys.exit(1)
        
        success = backup_manager.restore_sqlite_backup(args.backup_path)
        if success:
            print(f"Database restored from: {args.backup_path}")
            sys.exit(0)
        else:
            print("Restore failed")
            sys.exit(1)
    
    elif args.action == 'stats':
        stats = backup_manager.get_database_stats()
        print("Database Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
