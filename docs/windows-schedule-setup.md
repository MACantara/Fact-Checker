# RSS Feed Automatic Updates - Windows Scheduled Task Setup

This document explains how to set up automatic RSS feed updates using Windows Task Scheduler, replacing the Flask APScheduler approach.

## Why Use Windows Task Scheduler?

The Flask APScheduler only works when the Flask application is running as a server. For more reliable, standalone RSS updates that don't require the Flask app to be constantly running, Windows Task Scheduler is a better solution.

## Files Created

1. **`update_rss_feeds.py`** - Python script that updates all RSS feeds
2. **`update_rss_feeds.bat`** - Batch file wrapper for Windows Task Scheduler
3. **`logs/rss_updates.log`** - Log file (created automatically)

## Setting Up Windows Scheduled Task

### Method 1: Using Task Scheduler GUI

1. **Open Task Scheduler**
   - Press `Win + R`, type `taskschd.msc`, press Enter
   - Or search "Task Scheduler" in Start menu

2. **Create Basic Task**
   - Click "Create Basic Task..." in the right panel
   - Name: `RSS Feed Updates`
   - Description: `Automatic RSS feed updates for Philippine News Search Engine`

3. **Set Trigger (When to run)**
   - Choose "Daily" for now
   - Start time: Choose a convenient time (e.g., 9:00 AM)
   - Recur every: 1 days
   - Click Next

4. **Set Action (What to run)**
   - Choose "Start a program"
   - Program/script: `C:\Programming-Projects\Fact-Checker\update_rss_feeds.bat`
   - Start in: `C:\Programming-Projects\Fact-Checker`
   - Click Next, then Finish

5. **Configure Advanced Settings**
   - Right-click your new task → Properties
   - Go to "Triggers" tab → Edit the trigger
   - Check "Repeat task every:" and set to `5 minutes`
   - Set "for a duration of:" to `Indefinitely`
   - Click OK

### Method 2: Using PowerShell Command

Run this in PowerShell as Administrator:

```powershell
$action = New-ScheduledTaskAction -Execute "C:\Programming-Projects\Fact-Checker\update_rss_feeds.bat" -WorkingDirectory "C:\Programming-Projects\Fact-Checker"

$trigger = New-ScheduledTaskTrigger -Daily -At "09:00"
$trigger.Repetition = New-ScheduledTaskTrigger -Once -At "09:00" -RepetitionInterval (New-TimeSpan -Minutes 5)

$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask -TaskName "RSS Feed Updates" -Action $action -Trigger $trigger -Settings $settings -Description "Automatic RSS feed updates for Philippine News Search Engine"
```

## Configuration Options

### Changing Update Frequency

Edit the trigger in Task Scheduler:
- **Every 5 minutes**: Set "Repeat task every: 5 minutes"
- **Every 15 minutes**: Set "Repeat task every: 15 minutes"  
- **Every 30 minutes**: Set "Repeat task every: 30 minutes"
- **Hourly**: Set "Repeat task every: 1 hour"

### Testing the Setup

1. **Test the script manually:**
   ```cmd
   cd C:\Programming-Projects\Fact-Checker
   update_rss_feeds.bat
   ```

2. **Test the scheduled task:**
   - Right-click the task in Task Scheduler
   - Click "Run"
   - Check the "Last Run Result" column

3. **Check logs:**
   - View `logs/rss_updates.log` for detailed output
   - Check Task Scheduler History tab

## Monitoring and Troubleshooting

### Log Files
- **Location**: `C:\Programming-Projects\Fact-Checker\logs\rss_updates.log`
- **Contents**: Detailed update results, errors, and statistics
- **Rotation**: Logs append to the same file (consider manual cleanup)

### Task Scheduler History
- Enable history: Task Scheduler → Action menu → Enable All Tasks History
- View task history: Right-click task → View History

### Common Issues

1. **Task doesn't run**
   - Check the user account has proper permissions
   - Ensure the path to the batch file is correct
   - Verify Python virtual environment is accessible

2. **Script fails**
   - Check `logs/rss_updates.log` for error details
   - Test the script manually first
   - Ensure Flask app database is accessible

3. **Unicode errors**
   - Already fixed in the script with UTF-8 encoding
   - Check Windows console encoding if issues persist

## Removing the Flask APScheduler

Since we're now using Windows Task Scheduler, you can optionally remove the APScheduler code from the Flask app if you want to reduce dependencies.

## Benefits of This Approach

✅ **Reliable**: Runs independently of Flask app
✅ **Configurable**: Easy to change schedules via Task Scheduler
✅ **Monitored**: Built-in logging and Windows event tracking  
✅ **System-level**: Integrates with Windows system management
✅ **Resource-efficient**: Only runs when needed

## Current Status

- **52 RSS feeds** configured for updates
- **5-minute intervals** recommended for news feeds
- **Comprehensive logging** for monitoring
- **2 feeds may need attention** (Manila Bulletin, Manila Times Maritime)

The scheduled task will now handle RSS updates automatically, keeping your Philippine news search engine up-to-date without requiring the Flask server to run continuously!
