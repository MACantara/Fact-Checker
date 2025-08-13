@echo off
REM Complete Setup script for Windows Task Scheduler
REM This script sets up both RSS feed updates and database backups

echo ================================================================
echo Philippine News Search Engine - Automatic Task Setup
echo ================================================================
echo.
echo This script will set up two scheduled tasks:
echo 1. RSS Feed Updates - Every 5 minutes
echo 2. Database Backups - Daily at 2:00 AM
echo.
echo Please ensure you are running this script as Administrator.
echo.
pause

echo Setting up RSS Feed Updates...
echo ----------------------------------------------------------------

REM Create RSS update task
schtasks /create ^
    /tn "Philippine News RSS Feed Updates" ^
    /tr "C:\Programming-Projects\Fact-Checker\scripts\update_rss_feeds.bat" ^
    /sc minute ^
    /mo 5 ^
    /ru "SYSTEM" ^
    /rl highest ^
    /f

if %errorlevel% equ 0 (
    echo ✓ RSS Feed Update task created successfully!
) else (
    echo ✗ Failed to create RSS update task
    set "rss_error=1"
)

echo.
echo Setting up Database Backups...
echo ----------------------------------------------------------------

REM Create backup task
schtasks /create ^
    /tn "Philippine News Database Backup" ^
    /tr "C:\Programming-Projects\Fact-Checker\scripts\backup_database.bat" ^
    /sc daily ^
    /st 02:00 ^
    /ru "SYSTEM" ^
    /rl highest ^
    /f

if %errorlevel% equ 0 (
    echo ✓ Database Backup task created successfully!
) else (
    echo ✗ Failed to create backup task
    set "backup_error=1"
)

echo.
echo ================================================================
echo Setup Summary
echo ================================================================

if not defined rss_error (
    echo ✓ RSS Feed Updates: CONFIGURED
    echo   - Schedule: Every 5 minutes
    echo   - Updates 85 RSS feeds automatically
    echo   - Indexes new articles for search
    echo   - Logs to: logs/rss_updates.log
) else (
    echo ✗ RSS Feed Updates: FAILED
)

echo.

if not defined backup_error (
    echo ✓ Database Backups: CONFIGURED  
    echo   - Schedule: Daily at 2:00 AM
    echo   - Creates compressed SQLite backups
    echo   - Automatic cleanup of old backups
    echo   - Logs to: logs/database_backups.log
) else (
    echo ✗ Database Backups: FAILED
)

echo.
echo ================================================================
echo Management Commands
echo ================================================================
echo.
echo View tasks in Task Scheduler:
echo   taskschd.msc
echo.
echo Monitor RSS updates:
echo   flask feeds stats
echo   type logs\rss_updates.log
echo.
echo Monitor backups:
echo   flask backup list
echo   flask backup stats
echo   type logs\database_backups.log
echo.
echo Manual operations:
echo   .\update_rss_feeds.bat     (Update feeds now)
echo   .\backup_database.bat      (Backup database now)
echo.
echo Disable/Enable tasks:
echo   schtasks /change /tn "Philippine News RSS Feed Updates" /disable
echo   schtasks /change /tn "Philippine News RSS Feed Updates" /enable
echo   schtasks /change /tn "Philippine News Database Backup" /disable
echo   schtasks /change /tn "Philippine News Database Backup" /enable
echo.

if not defined rss_error if not defined backup_error (
    echo ✓ Setup completed successfully! 
    echo Your Philippine News Search Engine is now fully automated.
    echo.
    echo Current Status:
    echo   - 85 RSS feeds will update every 5 minutes
    echo   - Database will backup daily at 2:00 AM  
    echo   - All logs are saved for monitoring
    echo   - System will run automatically in background
) else (
    echo ⚠ Setup completed with errors. Please run as Administrator and try again.
)

echo.
pause
