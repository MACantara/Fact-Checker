@echo off
REM Setup script for Windows Task Scheduler - RSS Feed Updates
REM This script creates a scheduled task for automatic RSS feed updates every 5 minutes

echo Setting up Windows Task Scheduler for RSS Feed Updates...
echo.

REM Create the scheduled task for RSS updates every 5 minutes
schtasks /create ^
    /tn "Philippine News RSS Feed Updates" ^
    /tr "C:\Programming-Projects\Fact-Checker\update_rss_feeds.bat" ^
    /sc minute ^
    /mo 5 ^
    /ru "SYSTEM" ^
    /rl highest ^
    /f

if %errorlevel% equ 0 (
    echo ✓ RSS Feed Update scheduled task created successfully!
    echo.
    echo Task Details:
    echo   Name: Philippine News RSS Feed Updates
    echo   Schedule: Every 5 minutes
    echo   Script: update_rss_feeds.bat
    echo   User: SYSTEM
    echo.
    echo This will update all 85 RSS feeds every 5 minutes automatically.
    echo You can view/modify this task in Task Scheduler ^(taskschd.msc^)
    echo.
    echo The task will:
    echo   - Update all active RSS feeds
    echo   - Index new articles for search
    echo   - Log results to logs/rss_updates.log
    echo   - Handle feed errors gracefully
) else (
    echo ✗ Failed to create RSS update scheduled task
    echo Please run this script as Administrator
)

echo.
echo Additional setup options:
echo.
echo 1. To change update frequency, modify the task in Task Scheduler
echo 2. To disable updates temporarily, disable the task
echo 3. To monitor updates, check logs/rss_updates.log
echo 4. To view feed statistics, run: flask feeds stats
echo.
pause
