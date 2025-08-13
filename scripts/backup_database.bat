@echo off
REM Database Backup Script for Windows Task Scheduler
REM Runs automatic database backups for the Philippine News Search Engine

cd /d "C:\Programming-Projects\Fact-Checker"

REM Activate virtual environment
call ".venv\Scripts\activate.bat"

REM Run database backup with cleanup
echo [%date% %time%] Starting database backup...
python backup_database.py backup --compress

REM Clean up old backups (keep last 10 backups, or 30 days)
echo [%date% %time%] Cleaning up old backups...
python backup_database.py cleanup --keep-days 30 --keep-count 10

REM Deactivate virtual environment
deactivate

echo [%date% %time%] Database backup completed
exit /b %ERRORLEVEL%
