from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.orm import Session
from app.extensions import db
from app.models import RSSFeed, Article, FeedUpdateLog
from datetime import datetime, timezone
import hashlib


class DatabaseRepository:
    """Repository for database operations"""
    
    def __init__(self):
        self.db = db
    
    # RSS Feed operations
    def get_all_feeds(self, active_only: bool = True) -> List[RSSFeed]:
        """Get all RSS feeds"""
        query = self.db.session.query(RSSFeed)
        if active_only:
            query = query.filter(RSSFeed.active == True)
        return query.order_by(RSSFeed.name).all()
    
    def get_feed_by_key(self, key: str) -> Optional[RSSFeed]:
        """Get RSS feed by key"""
        return self.db.session.query(RSSFeed).filter(RSSFeed.key == key).first()
    
    def create_feed(self, key: str, name: str, url: str, category: str, active: bool = True) -> RSSFeed:
        """Create new RSS feed"""
        feed = RSSFeed(
            key=key,
            name=name,
            url=url,
            category=category,
            active=active
        )
        self.db.session.add(feed)
        self.db.session.commit()
        return feed
    
    def update_feed_fetch_status(self, feed_key: str, status: str, error: str = None, 
                                articles_count: int = 0, feed_metadata: Dict = None) -> None:
        """Update feed fetch status"""
        feed = self.get_feed_by_key(feed_key)
        if feed:
            feed.last_fetched_at = datetime.now(timezone.utc)
            feed.last_fetch_status = status
            feed.last_fetch_error = error
            feed.last_articles_count = articles_count
            
            if feed_metadata:
                feed.feed_title = feed_metadata.get('title', '')
                feed.feed_description = feed_metadata.get('description', '')
            
            self.db.session.commit()
    
    def delete_articles_by_feed(self, feed_key: str) -> int:
        """Delete all articles for a specific feed"""
        deleted_count = self.db.session.query(Article).filter_by(feed_key=feed_key).count()
        self.db.session.query(Article).filter_by(feed_key=feed_key).delete()
        self.db.session.commit()
        return deleted_count
    
    # Article operations
    def get_article_by_id(self, article_id: str) -> Optional[Article]:
        """Get article by ID"""
        return self.db.session.query(Article).filter(Article.article_id == article_id).first()
    
    def create_article(self, article_data: Dict[str, Any]) -> Article:
        """Create new article"""
        # Generate article ID
        article_id = self._generate_article_id(article_data.get('title', ''), article_data.get('link', ''))
        
        article = Article(
            article_id=article_id,
            title=article_data.get('title', ''),
            summary=article_data.get('summary'),
            content=article_data.get('content'),
            link=article_data.get('link', ''),
            published=article_data.get('published'),
            source=article_data.get('source', ''),
            category=article_data.get('category', ''),
            author=article_data.get('author'),
            tags=article_data.get('tags', []),
            feed_key=article_data.get('feed_key', '')
        )
        
        self.db.session.add(article)
        self.db.session.commit()
        return article
    
    def bulk_create_articles(self, articles_data: List[Dict[str, Any]]) -> int:
        """Bulk create articles, return count of new articles"""
        new_articles = []
        
        for article_data in articles_data:
            article_id = self._generate_article_id(article_data.get('title', ''), article_data.get('link', ''))
            
            # Check if article already exists
            existing = self.get_article_by_id(article_id)
            if existing:
                continue
            
            article = Article(
                article_id=article_id,
                title=article_data.get('title', ''),
                summary=article_data.get('summary'),
                content=article_data.get('content'),
                link=article_data.get('link', ''),
                published=article_data.get('published'),
                source=article_data.get('source', ''),
                category=article_data.get('category', ''),
                author=article_data.get('author'),
                tags=article_data.get('tags', []),
                feed_key=article_data.get('feed_key', '')
            )
            
            new_articles.append(article)
        
        if new_articles:
            self.db.session.add_all(new_articles)
            self.db.session.commit()
        
        return len(new_articles)
    
    def search_articles(self, query: str = None, category: str = None, source: str = None,
                       page: int = 1, per_page: int = 20, sort_by: str = 'created_at') -> Tuple[List[Article], int]:
        """Search articles in database"""
        db_query = self.db.session.query(Article)
        
        # Apply filters
        filters = []
        
        if query:
            # Simple text search in title, summary, and content
            search_filter = or_(
                Article.title.ilike(f'%{query}%'),
                Article.summary.ilike(f'%{query}%'),
                Article.content.ilike(f'%{query}%')
            )
            filters.append(search_filter)
        
        if category:
            filters.append(Article.category.ilike(f'%{category}%'))
        
        if source:
            filters.append(Article.source.ilike(f'%{source}%'))
        
        if filters:
            db_query = db_query.filter(and_(*filters))
        
        # Apply sorting
        if sort_by == 'date' or sort_by == 'published':
            db_query = db_query.order_by(desc(Article.published))
        elif sort_by == 'title':
            db_query = db_query.order_by(asc(Article.title))
        else:  # Default to creation date
            db_query = db_query.order_by(desc(Article.created_at))
        
        # Get total count
        total = db_query.count()
        
        # Apply pagination
        articles = db_query.offset((page - 1) * per_page).limit(per_page).all()
        
        return articles, total
    
    def get_categories(self) -> List[str]:
        """Get all unique categories"""
        result = self.db.session.query(Article.category).distinct().all()
        return sorted([cat[0] for cat in result if cat[0]])
    
    def get_sources(self) -> List[str]:
        """Get all unique sources"""
        result = self.db.session.query(Article.source).distinct().all()
        return sorted([source[0] for source in result if source[0]])
    
    def get_article_count(self) -> int:
        """Get total number of articles"""
        return self.db.session.query(Article).count()
    
    def get_article_count_by_feed(self, feed_key: str) -> int:
        """Get article count for a specific feed"""
        return self.db.session.query(Article).filter_by(feed_key=feed_key).count()
    
    def get_recent_articles(self, limit: int = 10) -> List[Article]:
        """Get most recent articles"""
        return (self.db.session.query(Article)
                .order_by(desc(Article.created_at))
                .limit(limit)
                .all())
    
    # Feed update log operations
    def create_update_log(self, feed_key: str) -> FeedUpdateLog:
        """Create new feed update log"""
        log = FeedUpdateLog(
            feed_key=feed_key,
            started_at=datetime.now(timezone.utc),
            status='running'
        )
        self.db.session.add(log)
        self.db.session.commit()
        return log
    
    def complete_update_log(self, log_id: int, status: str, articles_found: int = 0,
                           articles_new: int = 0, articles_updated: int = 0,
                           error_message: str = None, feed_entries_count: int = 0,
                           feed_metadata: Dict = None) -> None:
        """Complete feed update log"""
        log = self.db.session.query(FeedUpdateLog).get(log_id)
        if log:
            log.completed_at = datetime.now(timezone.utc)
            log.status = status
            log.articles_found = articles_found
            log.articles_new = articles_new
            log.articles_updated = articles_updated
            log.error_message = error_message
            log.feed_entries_count = feed_entries_count
            log.feed_metadata = feed_metadata
            self.db.session.commit()
    
    def get_recent_update_logs(self, feed_key: str = None, limit: int = 10) -> List[FeedUpdateLog]:
        """Get recent update logs"""
        query = self.db.session.query(FeedUpdateLog)
        if feed_key:
            query = query.filter(FeedUpdateLog.feed_key == feed_key)
        return query.order_by(desc(FeedUpdateLog.started_at)).limit(limit).all()
    
    # Statistics
    def get_feed_statistics(self) -> Dict[str, Any]:
        """Get feed statistics"""
        total_feeds = self.db.session.query(RSSFeed).count()
        active_feeds = self.db.session.query(RSSFeed).filter(RSSFeed.active == True).count()
        total_articles = self.get_article_count()
        
        # Articles by category
        category_counts = (self.db.session.query(Article.category, func.count(Article.id))
                          .group_by(Article.category)
                          .all())
        
        # Articles by source
        source_counts = (self.db.session.query(Article.source, func.count(Article.id))
                        .group_by(Article.source)
                        .all())
        
        return {
            'total_feeds': total_feeds,
            'active_feeds': active_feeds,
            'total_articles': total_articles,
            'categories': dict(category_counts),
            'sources': dict(source_counts)
        }
    
    def _generate_article_id(self, title: str, link: str) -> str:
        """Generate unique ID for article based on title and link"""
        content = f"{title}{link}"
        return hashlib.md5(content.encode()).hexdigest()
