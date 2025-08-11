from datetime import datetime, timezone
from sqlalchemy import Integer, String, Text, DateTime, Boolean, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from typing import Optional, List, Dict, Any


class Base(DeclarativeBase):
    pass


class RSSFeed(Base):
    """Model for RSS feed configurations"""
    __tablename__ = 'rss_feeds'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    key: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    url: Mapped[str] = mapped_column(String(500), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Last fetch information
    last_fetched_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_fetch_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # 'success', 'error'
    last_fetch_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_articles_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Feed metadata
    feed_title: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    feed_description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    feed_updated: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    
    def __repr__(self):
        return f'<RSSFeed {self.key}: {self.name}>'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'key': self.key,
            'name': self.name,
            'url': self.url,
            'category': self.category,
            'active': self.active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_fetched_at': self.last_fetched_at.isoformat() if self.last_fetched_at else None,
            'last_fetch_status': self.last_fetch_status,
            'last_fetch_error': self.last_fetch_error,
            'last_articles_count': self.last_articles_count,
            'feed_title': self.feed_title,
            'feed_description': self.feed_description,
            'feed_updated': self.feed_updated
        }


class Article(Base):
    """Model for storing article data"""
    __tablename__ = 'articles'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    article_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)  # MD5 hash of title+link
    
    # Article content
    title: Mapped[str] = mapped_column(String(1000), nullable=False, index=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    link: Mapped[str] = mapped_column(String(1000), nullable=False)
    
    # Metadata
    published: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    author: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Tags stored as JSON array
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    
    # System metadata
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    
    # Feed relationship
    feed_key: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    def __repr__(self):
        return f'<Article {self.article_id}: {self.title[:50]}...>'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.article_id,
            'title': self.title,
            'summary': self.summary,
            'content': self.content,
            'link': self.link,
            'published': self.published.isoformat() if self.published else None,
            'source': self.source,
            'category': self.category,
            'author': self.author,
            'tags': self.tags or [],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'feed_key': self.feed_key
        }


class FeedUpdateLog(Base):
    """Model for logging feed update operations"""
    __tablename__ = 'feed_update_logs'
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    feed_key: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # Update details
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # 'running', 'success', 'error'
    
    # Results
    articles_found: Mapped[int] = mapped_column(Integer, default=0)
    articles_new: Mapped[int] = mapped_column(Integer, default=0)
    articles_updated: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Raw feed data
    feed_entries_count: Mapped[int] = mapped_column(Integer, default=0)
    feed_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    
    def __repr__(self):
        return f'<FeedUpdateLog {self.feed_key} at {self.started_at}>'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'feed_key': self.feed_key,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'status': self.status,
            'articles_found': self.articles_found,
            'articles_new': self.articles_new,
            'articles_updated': self.articles_updated,
            'error_message': self.error_message,
            'feed_entries_count': self.feed_entries_count,
            'feed_metadata': self.feed_metadata
        }
