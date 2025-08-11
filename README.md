# Philippine News Search Engine

A web application for searching through Philippine news articles using Whoosh search engine and RSS feeds from leading Philippine news sources.

## Features

- **Fast Search**: Powered by Whoosh search engine for lightning-fast full-text search
- **Real-time Updates**: Automatically updates RSS feeds every 30 minutes
- **Smart Filtering**: Filter by category, source, or date
- **Trusted Sources**: Content from reputable Philippine news organizations
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **RESTful API**: JSON API endpoints for integration

## News Sources

The application aggregates news from 18 leading Philippine news sources:

- Philippine News Agency (Government)
- Philippine Daily Inquirer
- Rappler
- Philippine Star (Headlines, Nation, Business)
- GMA News (National, World, Metro, Regions, Special Reports)
- Manila Bulletin
- Malaya Business Insight
- BusinessWorld Online
- SunStar Philippines
- Interaksyon

## Technology Stack

- **Backend**: Python 3.12, Flask
- **Search Engine**: Whoosh
- **RSS Processing**: feedparser
- **Task Scheduling**: APScheduler
- **Data Validation**: Marshmallow
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Icons**: Bootstrap Icons
- **Deployment**: Docker, Docker Compose

## Architecture

The application follows Flask best practices with:

- **Factory Pattern**: App creation with configuration
- **Blueprint Pattern**: Modular route organization
- **Repository Pattern**: Data access abstraction
- **Service Layer**: Business logic separation
- **DTO/Schema Pattern**: Data validation with Marshmallow
- **Command Pattern**: Flask CLI commands
- **Extension Initialization**: Dependency injection

## Installation

### Prerequisites

- Python 3.12+
- pip
- Virtual environment (recommended)

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Fact-Checker
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   copy .env.example .env  # Windows
   # cp .env.example .env  # Linux/Mac
   ```

5. **Initialize search index**
   ```bash
   set FLASK_APP=run.py  # Windows
   # export FLASK_APP=run.py  # Linux/Mac
   flask init-index
   ```

6. **Update RSS feeds**
   ```bash
   flask update-feeds
   ```

7. **Run the application**
   ```bash
   python run.py
   ```

Visit `http://127.0.0.1:5000` in your browser.

### Docker Deployment

1. **Using Docker Compose** (recommended)
   ```bash
   docker-compose up -d
   ```

2. **Using Docker directly**
   ```bash
   docker build -t fact-checker .
   docker run -p 5000:5000 fact-checker
   ```

## Configuration

Environment variables can be set in `.env` file:

```env
FLASK_APP=run.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
RSS_UPDATE_INTERVAL=30
SEARCH_RESULTS_PER_PAGE=20
```

## CLI Commands

The application provides several CLI commands:

```bash
# Initialize search index
flask init-index

# Update all RSS feeds
flask update-feeds

# Show search statistics
flask show-stats

# Optimize search index
flask optimize-index
```

## API Endpoints

### Search Articles
```
GET /api/search?q=query&category=News&source=GMA&page=1&per_page=20&sort_by=relevance
```

### Get Metadata
```
GET /api/metadata
```

### Update Feeds
```
POST /api/update-feeds
```

## Search Features

- **Full-text search** across title, summary, and content
- **Filter by category**: Government, News, Business, etc.
- **Filter by source**: Specific news organization
- **Sort by relevance** or date
- **Pagination** support
- **Advanced query syntax** supported by Whoosh

## Project Structure

```
Fact-Checker/
├── app/
│   ├── blueprints/          # Route handlers
│   ├── commands/            # CLI commands
│   ├── extensions/          # Extension initialization
│   ├── repositories/        # Data access layer
│   ├── schemas/             # Data validation schemas
│   ├── services/            # Business logic
│   ├── static/              # Static files
│   ├── templates/           # Jinja2 templates
│   └── __init__.py          # App factory
├── data/                    # RSS feed data storage
├── index/                   # Whoosh search index
├── config.py                # Configuration classes
├── requirements.txt         # Python dependencies
├── run.py                   # Application entry point
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
└── README.md               # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and support, please create an issue in the repository.

## Monitoring

The application includes:
- Health check endpoints
- Logging for debugging
- Performance monitoring
- Error tracking

## Security

- Input validation and sanitization
- CSRF protection
- Rate limiting (recommended for production)
- Security headers via nginx
- Non-root Docker user

## Performance

- Efficient Whoosh indexing
- Pagination for large result sets
- Background RSS updates
- Index optimization
- Nginx caching for static files
