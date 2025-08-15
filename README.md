# Philippine News Search Engine

A web application for searching through Philippine news articles using Whoosh se6. **Train the fake news detection model**
   ```bash
   python train_model.py
   ```
   Or using Flask CLI:
   ```bash
   flask ml train
   ```

7. **Run the application**rch engine and RSS feeds from leading Philippine news sources. Now includes **AI-powered fake news detection** using machine learning.

## Features

- **Fast Search**: Powered by Whoosh search engine for lightning-fast full-text search
- **🤖 AI Fake News Detection**: Machine learning model trained on 72K+ articles to identify potentially fake or misleading news
- **Real-time Updates**: Automatically updates RSS feeds every 5 minutes via Windows Task Scheduler
- **Smart Filtering**: Filter by category, source, or date
- **Trusted Sources**: Content from reputable Philippine news organizations
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **RESTful API**: JSON API endpoints for integration
- **🛡️ URL Analysis**: Extract and analyze content directly from news URLs
- **📊 Batch Processing**: Analyze multiple articles simultaneously

## News Sources

The application aggregates news from 85 leading Philippine news sources including:

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
- And 75+ more regional and national sources

## 🤖 Machine Learning Models

Our hybrid fake news detection system combines two powerful models for optimal performance:

### 1. DistilBERT (Primary Model)
- **Architecture**: DistilBERT-base-uncased (66M parameters)
- **Features**: Contextual understanding of text
- **Accuracy**: 90-95% on test data
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA
- **Memory Efficient**: Runs on GPUs with as little as 2GB VRAM

### 2. Logistic Regression (Fallback Model)
- **Algorithm**: Fast and lightweight
- **Features**: TF-IDF vectorization with 10,000 features
- **Accuracy**: 94.6% on validation set
- **Deployment**: CPU-only, minimal resource requirements

### Dataset
- **Source**: WELFake Dataset with 72,134 labeled articles
- **Training/Test Split**: 80/20 stratified split
- **Class Balance**: Approximately 50/50 real vs fake news
- **Preprocessing**: Advanced text cleaning and tokenization

### Training Infrastructure
- **GPU Support**: Automatic detection and optimization for NVIDIA GPUs
- **Memory Management**: Automatic batch size adjustment for available VRAM
- **Mixed Precision**: FP16 training for faster convergence (on supported GPUs)
- **Checkpointing**: Automatic saving of best model during training
- **Early Stopping**: Prevents overfitting and saves training time

### Deployment
- **Hybrid System**: Automatically selects best available model
- **API Endpoints**: RESTful API for model inference
- **Web Interface**: User-friendly interface for predictions
- **Batch Processing**: Support for analyzing multiple articles at once

## Technology Stack

- **Backend**: Python 3.12, Flask
- **Search Engine**: Whoosh
- **RSS Processing**: feedparser
- **Task Scheduling**: Windows Task Scheduler (production) / APScheduler (development)
- **Machine Learning**: scikit-learn, NLTK, pandas, numpy
- **Fake News Detection**: TF-IDF + Logistic Regression
- **Data Validation**: Marshmallow
- **Frontend**: HTML5, Tailwind CSS, JavaScript
- **Icons**: Bootstrap Icons
- **Deployment**: Docker, Docker Compose
- **Database**: SQLite (with PostgreSQL/MySQL support)

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
   # Install base requirements
   pip install -r requirements.txt
   
   # For GPU training (recommended)
   python setup_gpu_training.py
   
   # Or for CPU-only installation
   # pip install torch torchvision torchaudio
   # pip install transformers datasets accelerate
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

### Machine Learning Commands

#### Training
```bash
# Quick training (5-10 minutes)
python train_hybrid_models.py --quick

# Full training (30-60 minutes)
python train_hybrid_models.py

# Train specific models
python train_hybrid_models.py --distilbert-only  # Train only DistilBERT
python train_hybrid_models.py --lr-only          # Train only Logistic Regression

# Custom training parameters
python train_hybrid_models.py --batch-size 4 --learning-rate 2e-5 --epochs 5
```

#### Prediction & Evaluation
```bash
# Make predictions
flask ml predict "news text"     # Predict if text is fake/real
flask ml predict-url "https://example.com/article"  # Predict from URL

# Model information
flask ml model-info              # Show model information
flask ml test-samples            # Test with sample cases

# Test model performance
python test_models.py            # Run comprehensive tests
```

#### GPU Commands (Linux/Windows)
```bash
# Check GPU status (NVIDIA)
nvidia-smi

# Monitor GPU usage (Linux)
nvidia-smi -l 1  # Updates every second

# Monitor GPU usage (Windows)
nvidia-smi -l 1
```

# Feed Management
flask feeds add "Feed Name" "URL" "Category"
flask feeds list
flask feeds stats
flask feeds remove <feed_id>

# Database Backup
flask backup create              # Create database backup
flask backup list               # List all backups
flask backup stats              # Show backup statistics
flask backup restore <backup_file>  # Restore from backup
```

## API Endpoints

### Search Articles
```
GET /api/search?q=query&category=News&source=GMA&page=1&per_page=20&sort_by=relevance
```

### Fake News Detection
```
POST /api/ml/predict/text
{
    "title": "Article title",
    "content": "Article content to analyze"
}

POST /api/ml/predict/url
{
    "url": "https://example.com/news-article"
}

POST /api/ml/predict/batch
{
    "texts": [
        {"title": "Title 1", "content": "Content 1"},
        {"title": "Title 2", "content": "Content 2"}
    ]
}

GET /api/ml/model/info          # Get model information
GET /api/ml/model/health        # Check model health
GET /api/ml/predict/demo        # Get demo predictions
```

### Metadata and Feeds
```
GET /api/metadata              # Get search metadata
POST /api/update-feeds         # Update RSS feeds
```

## Search Features

- **Full-text search** across title, summary, and content
- **Filter by category**: Government, News, Business, etc.
- **Filter by source**: Specific news organization
- **Sort by relevance** or date
- **Pagination** support
- **Advanced query syntax** supported by Whoosh

## Fake News Detection Features

- **Text Analysis**: Paste or type news content for immediate analysis
- **URL Analysis**: Extract and analyze content directly from news URLs
- **Batch Processing**: Analyze multiple articles simultaneously
- **Confidence Scoring**: Get probability scores for fake vs real classification
- **Model Transparency**: View detailed model information and performance metrics
- **API Integration**: RESTful API endpoints for external integration
- **Sample Testing**: Pre-loaded examples to test the system

## Project Structure

```
Fact-Checker/
├── app/
│   ├── blueprints/          # Route handlers
│   │   ├── main.py         # Main search routes
│   │   ├── admin.py        # Admin interface
│   │   ├── ml.py           # ML API endpoints
│   │   └── fake_news.py    # Fake news web interface
│   ├── commands/            # CLI commands
│   │   ├── feeds.py        # Feed management
│   │   ├── backup.py       # Database backup
│   │   └── ml.py           # Machine learning commands
│   ├── extensions/          # Extension initialization
│   ├── ml/                  # Machine learning components
│   │   ├── model_trainer.py # ML model training
│   │   ├── predictor.py    # Prediction service
│   │   └── models/         # Trained models storage
│   ├── repositories/        # Data access layer
│   ├── schemas/             # Data validation schemas
│   ├── services/            # Business logic
│   ├── static/              # Static files
│   ├── templates/           # Jinja2 templates
│   │   ├── fake_news/      # Fake news detection templates
│   │   └── ...             # Other templates
│   └── __init__.py          # App factory
├── datasets/                # ML training datasets
│   └── WELFake_Dataset.csv # Fake news training data
├── data/                    # RSS feed data storage
├── index/                   # Whoosh search index
├── logs/                    # Application logs
├── backups/                 # Database backups
├── scripts/                 # Utility scripts
├── train_model.py           # ML model training script
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
