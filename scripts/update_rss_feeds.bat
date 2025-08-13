@echo off
REM RSS Feed Update Batch Script for Windows Task Scheduler
REM This batch file activates the virtual environment and runs the RSS update script

cd /d "C:\Programming-Projects\Fact-Checker"

REM Activate virtual environment
call ".venv\Scripts\activate.bat"

REM Run the RSS update script
python update_rss_feeds.py

REM Deactivate virtual environment
deactivate

REM Exit with the same code as the Python script
exit /b %ERRORLEVEL%
