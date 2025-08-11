# Database Backup System

This Philippine News Search Engine includes a comprehensive database backup system that supports multiple database types and provides both manual and automatic backup capabilities.

## Features

- **Multi-Database Support**: SQLite, PostgreSQL, MySQL
- **Automatic Compression**: Reduces backup file sizes
- **Scheduled Backups**: Windows Task Scheduler integration
- **Retention Policies**: Automatic cleanup of old backups
- **CLI Management**: Full command-line interface
- **Restore Capabilities**: Easy database restoration
- **Backup Verification**: File integrity checking

## Supported Database Types

### SQLite (Default)
- **Method**: File copy with optional compression
- **Requirements**: No external tools needed
- **Backup Size**: ~50-90% reduction with compression
- **Restore**: Direct file replacement

### PostgreSQL
- **Method**: `pg_dump` utility
- **Requirements**: PostgreSQL client tools
- **Backup Format**: SQL dump files
- **Features**: Schema + data, clean dumps

### MySQL/MariaDB
- **Method**: `mysqldump` utility  
- **Requirements**: MySQL client tools
- **Backup Format**: SQL dump files
- **Features**: Single-transaction, routines, triggers

## Quick Start

### 1. Manual Backup
```bash
# Create a backup now
python backup_database.py backup

# Create uncompressed backup
python backup_database.py backup --no-compress

# List all backups
python backup_database.py list

# Show database statistics
python backup_database.py stats
```

### 2. Flask CLI Commands
```bash
# Create backup using Flask CLI
flask backup create

# List backups
flask backup list

# Clean up old backups
flask backup cleanup

# Show statistics
flask backup stats

# Restore from backup
flask backup restore path/to/backup.db.gz --confirm
```

### 3. Automated Backups
```bash
# Run the batch file (Windows)
backup_database.bat

# Set up Windows Task Scheduler
# - Program: C:\Programming-Projects\Fact-Checker\backup_database.bat
# - Schedule: Daily at 2:00 AM
# - Settings: Run whether user is logged in or not
```

## Configuration

### Environment Variables
```bash
# Enable/disable backups
BACKUP_ENABLED=true

# Backup schedule (hours between backups)
BACKUP_SCHEDULE_HOURS=24

# Retention policy
BACKUP_KEEP_DAYS=30
BACKUP_KEEP_COUNT=10

# Compression
BACKUP_COMPRESS=true
```

### Config File Settings
```python
class Config:
    BACKUP_ENABLED = True
    BACKUP_SCHEDULE_HOURS = 24
    BACKUP_KEEP_DAYS = 30
    BACKUP_KEEP_COUNT = 10
    BACKUP_COMPRESS = True
```

## Windows Task Scheduler Setup

### Method 1: GUI Setup
1. Open Task Scheduler (`taskschd.msc`)
2. Click "Create Basic Task"
3. **Name**: `Database Backup`
4. **Trigger**: Daily at 2:00 AM
5. **Action**: Start a program
6. **Program**: `C:\Programming-Projects\Fact-Checker\backup_database.bat`
7. **Start in**: `C:\Programming-Projects\Fact-Checker`

### Method 2: PowerShell Command
```powershell
$action = New-ScheduledTaskAction -Execute "C:\Programming-Projects\Fact-Checker\backup_database.bat" -WorkingDirectory "C:\Programming-Projects\Fact-Checker"

$trigger = New-ScheduledTaskTrigger -Daily -At "02:00"

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName "Database Backup" -Action $action -Trigger $trigger -Settings $settings -Description "Automatic database backup for Philippine News Search Engine"
```

## Backup File Structure

### Filename Format
```
backup_<database_type>_<timestamp>.<extension>[.gz]

Examples:
- backup_sqlite_20250812_143022.db.gz
- backup_postgresql_20250812_143022.sql.gz
- backup_mysql_20250812_143022.sql
```

### Directory Structure
```
backups/
├── backup_sqlite_20250812_143022.db.gz
├── backup_sqlite_20250811_143022.db.gz
├── backup_sqlite_20250810_143022.db.gz
└── ...

logs/
├── database_backups.log
└── rss_updates.log
```

## Backup Operations

### Creating Backups

#### SQLite Backup Process
1. **Validation**: Check database file exists
2. **Copy**: Create compressed/uncompressed copy
3. **Verification**: Verify backup file size
4. **Logging**: Record backup statistics

#### PostgreSQL Backup Process
1. **Connection**: Connect using pg_dump
2. **Export**: Create SQL dump with schema + data
3. **Compression**: Optional gzip compression
4. **Validation**: Check dump completed successfully

#### MySQL Backup Process
1. **Connection**: Connect using mysqldump
2. **Export**: Single-transaction dump
3. **Features**: Include routines, triggers
4. **Compression**: Optional gzip compression

### Restoring Backups

#### SQLite Restore
```bash
# Using backup script
python backup_database.py restore backups/backup_sqlite_20250812_143022.db.gz

# Using Flask CLI
flask backup restore backups/backup_sqlite_20250812_143022.db.gz --confirm
```

#### Safety Features
- **Current DB Backup**: Creates backup of current database before restore
- **Validation**: Checks backup file exists and is readable
- **Rollback**: Can restore original if restore fails
- **Confirmation**: Requires explicit confirmation

### Cleanup Operations

#### Retention Policy
- **Keep Count**: Always preserve the N most recent backups
- **Keep Days**: Remove backups older than specified days
- **Combined**: Both policies work together for safety

#### Example Cleanup
```bash
# Keep last 10 backups OR 30 days (whichever is more)
python backup_database.py cleanup --keep-days 30 --keep-count 10

# More aggressive cleanup
python backup_database.py cleanup --keep-days 7 --keep-count 5
```

## Monitoring and Logs

### Log Files
- **Location**: `logs/database_backups.log`
- **Format**: Timestamped entries with backup details
- **Content**: Success/failure, file sizes, statistics

### Log Entry Example
```
2025-08-12 14:30:22,123 - INFO - Starting SQLite backup: dev_app.db -> backups/backup_sqlite_20250812_143022.db.gz
2025-08-12 14:30:23,456 - INFO - Backup completed successfully: 2048576 bytes
2025-08-12 14:30:23,457 - INFO - Database stats: {'feeds_count': 85, 'articles_count': 1909, 'update_logs_count': 45, 'active_feeds': 85}
```

### Monitoring Commands
```bash
# Show backup statistics
flask backup stats

# List all backups with details
flask backup list

# Check database health
python backup_database.py stats
```

## Integration with Existing Systems

### RSS Update Integration
The backup system works alongside the RSS feed update system:

1. **RSS Updates**: Every 5 minutes via Windows Task Scheduler
2. **Database Backups**: Daily via Windows Task Scheduler
3. **Log Management**: Separate log files for each system
4. **Error Handling**: Independent failure handling

### Search Index Consistency
- **Backup Scope**: Database only (search index rebuilt from database)
- **Restore Process**: Rebuild search index after database restore
- **Commands**: `flask init-index` after restore

## Best Practices

### Backup Schedule
- **Development**: Daily backups, keep 10 recent
- **Production**: Multiple daily backups, longer retention
- **Critical Systems**: Hourly backups with off-site storage

### Storage Considerations
- **Compression**: ~50-90% size reduction
- **Retention**: Balance storage vs. recovery needs
- **Location**: Consider off-site backup storage

### Recovery Testing
- **Regular Testing**: Periodically test restore procedures
- **Documentation**: Keep restore procedures documented
- **Verification**: Verify restored database integrity

## Troubleshooting

### Common Issues

#### Backup Creation Fails
```bash
# Check database permissions
ls -la dev_app.db

# Check disk space
df -h

# Check logs
tail logs/database_backups.log
```

#### Restore Fails
```bash
# Verify backup file integrity
python backup_database.py list

# Check if backup file is readable
gzip -t backup_file.gz  # For compressed files
```

#### Windows Task Scheduler Issues
```bash
# Check task history in Task Scheduler
# Verify batch file paths are absolute
# Ensure Python virtual environment is accessible
```

### Error Recovery
- **Failed Backup**: Check logs, retry manually
- **Failed Restore**: Original database backup created automatically
- **Corrupted Backup**: Use previous backup, investigate corruption cause

## Security Considerations

### Access Control
- **File Permissions**: Restrict backup file access
- **Database Credentials**: Secure connection strings
- **Log Security**: Protect log files from unauthorized access

### Data Protection
- **Compression**: Provides some obfuscation
- **Encryption**: Consider encrypting sensitive backups
- **Transport**: Secure methods for off-site backup transfer

## Future Enhancements

### Planned Features
- **Encryption**: Built-in backup encryption
- **Cloud Storage**: Integration with cloud backup services
- **Email Notifications**: Backup status notifications
- **Incremental Backups**: For large databases
- **Cross-Platform**: Linux/macOS support

### Extension Points
- **Custom Storage**: Plugin system for backup destinations
- **Backup Validation**: Content verification beyond file checks
- **Monitoring Integration**: Prometheus/Grafana metrics
- **API Integration**: RESTful backup management API

The backup system provides comprehensive protection for your Philippine News Search Engine database with minimal maintenance requirements and maximum reliability.
