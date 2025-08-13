@echo off
REM Setup script for Windows Task Scheduler - Database Backups
REM This script creates a scheduled task for automatic database backups

echo Setting up Windows Task Scheduler for Database Backups...
echo.

REM Create the scheduled task
schtasks /create ^
    /tn "Philippine News Database Backup" ^
    /tr "C:\Programming-Projects\Fact-Checker\scripts\backup_database.bat" ^
    /sc daily ^
    /st 02:00 ^
    /ru "SYSTEM" ^
    /rl highest ^
    /f

if %errorlevel% equ 0 (
    echo ✓ Scheduled task created successfully!
    echo.
    echo Task Details:
    echo   Name: Philippine News Database Backup
    echo   Schedule: Daily at 2:00 AM
    echo   Script: backup_database.bat
    echo   User: SYSTEM
    echo.
    echo You can view/modify this task in Task Scheduler ^(taskschd.msc^)
) else (
    echo ✗ Failed to create scheduled task
    echo Please run this script as Administrator
)

echo.
pause
