import os
import hashlib
import feedparser
import requests
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin
from app.repositories import SearchRepository
from app.repositories.database import DatabaseRepository
from app.schemas import ArticleSchema, SearchQuerySchema, SearchResultSchema, FeedUpdateStatusSchema


class RSSFeedService:
    """Service for handling RSS feed operations"""
    
    def __init__(self, db_repository: DatabaseRepository = None):
        self.db_repo = db_repository or DatabaseRepository()
    
    def fetch_feed(self, feed_url: str, timeout: int = 30) -> Dict[str, Any]:
        """Fetch RSS feed from URL"""
        try:
            # Set user agent to avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(feed_url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            feed = feedparser.parse(response.content)
            
            if feed.bozo and feed.bozo_exception:
                raise Exception(f"Feed parsing error: {feed.bozo_exception}")
            
            return feed
        
        except Exception as e:
            raise Exception(f"Failed to fetch feed from {feed_url}: {str(e)}")
    
    def parse_feed_entries(self, feed: Dict[str, Any], source_name: str, category: str, feed_key: str) -> List[Dict[str, Any]]:
        """Parse RSS feed entries into article format"""
        articles = []
        
        for entry in feed.entries:
            try:
                # Extract basic information
                title = getattr(entry, 'title', '')
                link = getattr(entry, 'link', '')
                summary = getattr(entry, 'summary', '') or getattr(entry, 'description', '')
                
                # Extract content
                content = ''
                if hasattr(entry, 'content') and entry.content:
                    content = entry.content[0].value if isinstance(entry.content, list) else entry.content
                elif summary:
                    content = summary
                
                # Parse published date
                published = None
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    try:
                        published = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                    except (TypeError, ValueError):
                        published = datetime.now(timezone.utc)
                elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                    try:
                        published = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                    except (TypeError, ValueError):
                        published = datetime.now(timezone.utc)
                else:
                    published = datetime.now(timezone.utc)
                
                # Extract author
                author = getattr(entry, 'author', '') or ''
                
                # Extract tags
                tags = []
                if hasattr(entry, 'tags'):
                    tags = [tag.term for tag in entry.tags if hasattr(tag, 'term')]
                
                article = {
                    'title': title,
                    'summary': summary,
                    'content': content,
                    'link': link,
                    'published': published,
                    'source': source_name,
                    'category': category,
                    'author': author,
                    'tags': tags,
                    'feed_key': feed_key
                }
                
                # Skip validation for now and just use the data
                articles.append(article)
            
            except Exception as e:
                print(f"Error parsing entry: {e}")
                continue
        
        return articles
    
    def update_feed(self, feed_key: str, feed_config: Dict[str, Any]) -> Dict[str, Any]:
        """Update a single RSS feed"""
        start_time = datetime.now(timezone.utc)
        
        # Create update log
        update_log = self.db_repo.create_update_log(feed_key)
        
        try:
            # Fetch and parse feed
            feed_data = self.fetch_feed(feed_config['url'])
            articles = self.parse_feed_entries(
                feed_data, 
                feed_config['name'], 
                feed_config['category'],
                feed_key
            )
            
            # Store articles in database
            articles_added = self.db_repo.bulk_create_articles(articles)
            
            # Update feed metadata
            feed_metadata = {
                'title': getattr(feed_data.feed, 'title', ''),
                'description': getattr(feed_data.feed, 'description', ''),
                'updated': getattr(feed_data.feed, 'updated', ''),
            }
            
            # Update feed status
            self.db_repo.update_feed_fetch_status(
                feed_key, 
                'success', 
                articles_count=articles_added,
                feed_metadata=feed_metadata
            )
            
            # Complete update log
            self.db_repo.complete_update_log(
                update_log.id,
                'success',
                articles_found=len(articles),
                articles_new=articles_added,
                feed_entries_count=len(feed_data.entries),
                feed_metadata=feed_metadata
            )
            
            status = {
                'feed_key': feed_key,
                'feed_name': feed_config['name'],
                'status': 'success',
                'articles_count': len(articles),
                'articles_added': articles_added,
                'error_message': None,
                'updated_at': start_time,
                'articles': articles
            }
            
            return status
        
        except Exception as e:
            error_msg = str(e)
            
            # Update feed status
            self.db_repo.update_feed_fetch_status(feed_key, 'error', error=error_msg)
            
            # Complete update log
            self.db_repo.complete_update_log(
                update_log.id,
                'error',
                error_message=error_msg
            )
            
            status = {
                'feed_key': feed_key,
                'feed_name': feed_config['name'],
                'status': 'error',
                'articles_count': 0,
                'articles_added': 0,
                'error_message': error_msg,
                'updated_at': start_time,
                'articles': []
            }
            
            return status
    
    def get_feeds_from_db(self) -> Dict[str, Dict[str, Any]]:
        """Get RSS feeds from database"""
        feeds = self.db_repo.get_all_feeds()
        feed_dict = {}
        
        for feed in feeds:
            feed_dict[feed.key] = {
                'name': feed.name,
                'url': feed.url,
                'category': feed.category,
                'active': feed.active
            }
        
        return feed_dict


class SearchService:
    """Service for search operations"""
    
    def __init__(self, search_repository: SearchRepository = None, db_repository: DatabaseRepository = None):
        self.search_repository = search_repository
        self.db_repo = db_repository or DatabaseRepository()
    
    def search_articles(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Search articles using the provided query"""
        # Validate query data
        schema = SearchQuerySchema()
        validated_query = schema.load(query_data)
        
        # Use database search if no Whoosh repository provided
        if self.search_repository:
            # Perform Whoosh search
            articles, total = self.search_repository.search(
                query=validated_query['query'],
                category=validated_query.get('category'),
                source=validated_query.get('source'),
                page=validated_query['page'],
                per_page=validated_query['per_page'],
                sort_by=validated_query['sort_by']
            )
            
            # Convert to dict format
            articles = [dict(article) for article in articles]
        else:
            # Perform database search
            db_articles, total = self.db_repo.search_articles(
                query=validated_query['query'],
                category=validated_query.get('category'),
                source=validated_query.get('source'),
                page=validated_query['page'],
                per_page=validated_query['per_page'],
                sort_by=validated_query['sort_by']
            )
            
            articles = [article.to_dict() for article in db_articles]
        
        # Calculate pagination
        pages = (total + validated_query['per_page'] - 1) // validated_query['per_page']
        has_prev = validated_query['page'] > 1
        has_next = validated_query['page'] < pages
        
        # Prepare result
        result = {
            'articles': articles,
            'total': total,
            'page': validated_query['page'],
            'per_page': validated_query['per_page'],
            'pages': pages,
            'has_prev': has_prev,
            'has_next': has_next,
            'query': validated_query['query']
        }
        
        # Return result without validation since articles come from our sources
        return result
    
    def get_search_metadata(self) -> Dict[str, Any]:
        """Get metadata for search interface"""
        if self.search_repository:
            return {
                'categories': self.search_repository.get_categories(),
                'sources': self.search_repository.get_sources(),
                'total_articles': self.search_repository.get_article_count()
            }
        else:
            return {
                'categories': self.db_repo.get_categories(),
                'sources': self.db_repo.get_sources(),
                'total_articles': self.db_repo.get_article_count()
            }
    
    def add_articles_to_index(self, articles: List[Dict[str, Any]]) -> int:
        """Add articles to search index"""
        if self.search_repository:
            return self.search_repository.add_articles(articles)
        else:
            # Articles are already in database, just return count
            return len(articles)


class FeedUpdateService:
    """Service for coordinating feed updates"""
    
    def __init__(self, rss_service: RSSFeedService, search_service: SearchService):
        self.rss_service = rss_service
        self.search_service = search_service
    
    def update_all_feeds(self, feed_configs: Dict[str, Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Update all active RSS feeds"""
        # If no feed configs provided, get from database
        if feed_configs is None:
            feed_configs = self.rss_service.get_feeds_from_db()
        
        results = []
        total_articles_added = 0
        
        for feed_key, feed_config in feed_configs.items():
            if not feed_config.get('active', True):
                continue
            
            try:
                # Update individual feed
                feed_result = self.rss_service.update_feed(feed_key, feed_config)
                
                # Add articles to search index if successful and search repository available
                if (feed_result['status'] == 'success' and 
                    feed_result['articles'] and 
                    self.search_service.search_repository):
                    
                    articles_indexed = self.search_service.add_articles_to_index(feed_result['articles'])
                    feed_result['articles_indexed'] = articles_indexed
                    total_articles_added += articles_indexed
                else:
                    feed_result['articles_indexed'] = feed_result.get('articles_added', 0)
                    total_articles_added += feed_result.get('articles_added', 0)
                
                # Remove articles from result to save memory
                if 'articles' in feed_result:
                    del feed_result['articles']
                
                results.append(feed_result)
            
            except Exception as e:
                error_result = {
                    'feed_key': feed_key,
                    'feed_name': feed_config.get('name', feed_key),
                    'status': 'error',
                    'articles_count': 0,
                    'articles_added': 0,
                    'articles_indexed': 0,
                    'error_message': str(e),
                    'updated_at': datetime.now(timezone.utc)
                }
                results.append(error_result)
        
        # Optimize search index after bulk updates
        if total_articles_added > 0 and self.search_service.search_repository:
            try:
                self.search_service.search_repository.optimize_index()
            except Exception as e:
                print(f"Index optimization failed: {e}")
        
        return results
