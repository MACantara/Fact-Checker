# RSS Feed CRUD Management

This Philippine News Search application now includes comprehensive CRUD (Create, Read, Update, Delete) functionality for managing RSS feeds through both web interface and command-line interface.

## Web Admin Interface

### Access the Admin Panel
- Navigate to: http://127.0.0.1:5000/admin
- Or click "Admin" in the navigation menu

### Features
- **Dashboard**: Overview of feeds, statistics, and quick actions
- **Feed Management**: List, create, edit, and delete RSS feeds
- **Status Toggle**: Quickly activate/deactivate feeds
- **Real-time Updates**: Trigger feed updates directly from the interface

### Admin Dashboard
The admin dashboard provides:
- Statistics cards showing total feeds, active feeds, total articles, and categories
- Quick action buttons for adding feeds and updating all feeds
- Overview table of recent feeds with their status

### Managing Feeds
1. **View All Feeds**: `/admin/feeds` - See all RSS feeds in a table format
2. **Add New Feed**: `/admin/feeds/new` - Form to create a new RSS feed
3. **Edit Feed**: Click edit icon next to any feed to modify its details
4. **Delete Feed**: Click delete icon to remove a feed and all its articles
5. **Toggle Status**: Click the status badge to activate/deactivate a feed

## Command Line Interface (CLI)

### Available Commands

#### List Feeds
```bash
flask feeds list                # List all active feeds
flask feeds list --active-only  # List only active feeds
```

#### Add New Feed
```bash
flask feeds add <key> <name> <url> <category> [--active/--inactive]
```
Example:
```bash
flask feeds add cnn_news "CNN News" "https://rss.cnn.com/rss/edition.rss" "News" --active
```

#### Update Feed
```bash
flask feeds update <key> [--name NAME] [--url URL] [--category CATEGORY] [--active/--inactive]
```
Example:
```bash
flask feeds update cnn_news --name "CNN International" --category "International News"
```

#### Delete Feed
```bash
flask feeds delete <key> [--confirm]
```
Example:
```bash
flask feeds delete cnn_news --confirm  # Skip confirmation prompt
```

#### Show Feed Details
```bash
flask feeds show <key>
```

#### Toggle Feed Status
```bash
flask feeds toggle <key>
```

#### Update Feed Content
```bash
flask feeds update-feed <key>        # Update specific feed
flask feeds update-feed --all         # Update all active feeds
```

#### Feed Statistics
```bash
flask feeds stats
```

### General Commands
```bash
flask update-feeds    # Update all feeds using database
flask show-stats      # Show search statistics
flask init-index      # Initialize search index
flask optimize-index  # Optimize search index
```

## API Endpoints

### REST API for Feed Management

#### List All Feeds
```
GET /admin/api/feeds
```

#### Create New Feed
```
POST /admin/api/feeds
Content-Type: application/json

{
    "key": "feed_key",
    "name": "Feed Name",
    "url": "https://example.com/rss.xml",
    "category": "Category",
    "active": true
}
```

#### Update Feed
```
PUT /admin/api/feeds/<feed_key>
Content-Type: application/json

{
    "key": "new_key",
    "name": "Updated Name",
    "url": "https://example.com/new-rss.xml",
    "category": "New Category",
    "active": false
}
```

#### Delete Feed
```
DELETE /admin/api/feeds/<feed_key>
```

#### Toggle Feed Status
```
POST /admin/feeds/<feed_key>/toggle
```

## Database Schema

The application uses SQLite database with the following tables:

### RSS_Feed Table
- `id`: Primary key
- `key`: Unique feed identifier
- `name`: Display name
- `url`: RSS feed URL
- `category`: Feed category
- `active`: Active status
- `created_at`: Creation timestamp
- `updated_at`: Last update timestamp
- `last_fetched_at`: Last fetch timestamp
- `last_fetch_status`: Success/error status
- `last_fetch_error`: Error message if any
- `last_articles_count`: Number of articles in last fetch
- `feed_title`: Feed metadata title
- `feed_description`: Feed metadata description

### Article Table
- `id`: Primary key
- `article_id`: Unique article identifier (hash)
- `title`: Article title
- `summary`: Article summary
- `content`: Article content
- `link`: Original article URL
- `published`: Publication date
- `source`: Source name
- `category`: Article category
- `author`: Article author
- `tags`: Article tags (JSON)
- `feed_key`: Reference to RSS feed
- `created_at`: Creation timestamp

### Feed_Update_Log Table
- `id`: Primary key
- `feed_key`: Reference to RSS feed
- `started_at`: Update start time
- `completed_at`: Update completion time
- `status`: Update status
- `articles_found`: Number of articles found
- `articles_new`: Number of new articles
- `error_message`: Error message if any
- `feed_entries_count`: Total entries in feed
- `feed_metadata`: Feed metadata (JSON)

## Migration from File-based to Database

The application has been migrated from JSON file storage to SQLite database. To initialize:

1. **Initialize Database**:
   ```bash
   python migrate_db.py init
   ```

2. **Migrate Existing Config**:
   ```bash
   python migrate_db.py migrate
   ```

3. **Check Status**:
   ```bash
   python migrate_db.py status
   ```

4. **Reset Database** (if needed):
   ```bash
   python migrate_db.py reset
   ```

## Security Notes

- The admin interface currently has no authentication
- For production use, implement proper authentication and authorization
- Consider adding rate limiting for API endpoints
- Validate and sanitize all user inputs

## Example Usage

1. **Add a new Philippine news source**:
   ```bash
   flask feeds add abs_cbn "ABS-CBN News" "https://news.abs-cbn.com/rss" "News"
   ```

2. **View feed details**:
   ```bash
   flask feeds show abs_cbn
   ```

3. **Update the feed to get latest articles**:
   ```bash
   flask feeds update-feed abs_cbn
   ```

4. **Check overall statistics**:
   ```bash
   flask feeds stats
   ```

5. **Deactivate a problematic feed**:
   ```bash
   flask feeds toggle problematic_feed
   ```

The CRUD functionality provides comprehensive management of RSS feeds for the Philippine news search engine, supporting both web-based and command-line workflows.
