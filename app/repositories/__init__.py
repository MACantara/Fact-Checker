import os
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from whoosh import index
from whoosh.fields import Schema, TEXT, KEYWORD, DATETIME, ID
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import And, Term
from whoosh.analysis import StemmingAnalyzer
from whoosh.sorting import FieldFacet
from app.schemas import ArticleSchema


class SearchRepository:
    """Repository for Whoosh search index operations"""
    
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.schema = Schema(
            id=ID(stored=True, unique=True),
            title=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            summary=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            content=TEXT(stored=True, analyzer=StemmingAnalyzer()),
            link=TEXT(stored=True),
            published=DATETIME(stored=True, sortable=True),
            source=KEYWORD(stored=True, lowercase=True),
            category=KEYWORD(stored=True, lowercase=True),
            author=TEXT(stored=True),
            tags=KEYWORD(stored=True, lowercase=True, commas=True)
        )
        self._ensure_index_exists()
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist"""
        if not os.path.exists(self.index_path):
            os.makedirs(self.index_path)
        
        if not index.exists_in(self.index_path):
            index.create_in(self.index_path, self.schema)
    
    def _generate_article_id(self, article_data: Dict[str, Any]) -> str:
        """Generate unique ID for article based on title and link"""
        content = f"{article_data.get('title', '')}{article_data.get('link', '')}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def add_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Add articles to the search index"""
        ix = index.open_dir(self.index_path)
        writer = ix.writer()
        
        added_count = 0
        
        try:
            for article_data in articles:
                # Generate unique ID
                article_id = self._generate_article_id(article_data)
                
                # Check if article already exists
                with ix.searcher() as searcher:
                    existing = searcher.document(id=article_id)
                    if existing:
                        continue  # Skip existing articles
                
                # Prepare document for indexing
                doc = {
                    'id': article_id,
                    'title': article_data.get('title', ''),
                    'summary': article_data.get('summary', ''),
                    'content': article_data.get('content', ''),
                    'link': article_data.get('link', ''),
                    'published': article_data.get('published'),
                    'source': article_data.get('source', ''),
                    'category': article_data.get('category', ''),
                    'author': article_data.get('author', ''),
                    'tags': ','.join(article_data.get('tags', []))
                }
                
                writer.add_document(**doc)
                added_count += 1
            
            writer.commit()
            
        except Exception as e:
            writer.cancel()
            raise e
        
        return added_count
    
    def search(self, query: str, category: Optional[str] = None, 
               source: Optional[str] = None, page: int = 1, 
               per_page: int = 20, sort_by: str = 'relevance') -> Tuple[List[Dict], int]:
        """Search articles in the index"""
        ix = index.open_dir(self.index_path)
        
        with ix.searcher() as searcher:
            # Build query parser for multiple fields
            parser = MultifieldParser(['title', 'summary', 'content'], ix.schema)
            parsed_query = parser.parse(query)
            
            # Add filters
            filters = []
            if category:
                filters.append(Term('category', category.lower()))
            if source:
                filters.append(Term('source', source.lower()))
            
            if filters:
                final_query = And([parsed_query] + filters)
            else:
                final_query = parsed_query
            
            # Set up sorting
            if sort_by == 'date':
                sortedby = FieldFacet('published', reverse=True)
            else:
                sortedby = None
            
            # Execute search
            results = searcher.search_page(
                final_query, 
                page, 
                pagelen=per_page,
                sortedby=sortedby
            )
            
            # Convert results to dictionaries
            articles = []
            for hit in results:
                article = dict(hit)
                # Convert tags back to list
                if article.get('tags'):
                    article['tags'] = [tag.strip() for tag in article['tags'].split(',') if tag.strip()]
                else:
                    article['tags'] = []
                
                # Ensure published is a proper datetime object or None
                if article.get('published'):
                    from datetime import datetime
                    if not isinstance(article['published'], datetime):
                        article['published'] = None
                
                articles.append(article)
            
            return articles, len(results)
    
    def get_categories(self) -> List[str]:
        """Get all available categories"""
        ix = index.open_dir(self.index_path)
        
        with ix.searcher() as searcher:
            categories = set()
            for doc in searcher.all_stored_fields():
                if doc.get('category'):
                    categories.add(doc['category'])
            
            return sorted(list(categories))
    
    def get_sources(self) -> List[str]:
        """Get all available sources"""
        ix = index.open_dir(self.index_path)
        
        with ix.searcher() as searcher:
            sources = set()
            for doc in searcher.all_stored_fields():
                if doc.get('source'):
                    sources.add(doc['source'])
            
            return sorted(list(sources))
    
    def get_article_count(self) -> int:
        """Get total number of articles in the index"""
        ix = index.open_dir(self.index_path)
        
        with ix.searcher() as searcher:
            return searcher.doc_count()
    
    def clear_index(self):
        """Clear all documents from the index"""
        ix = index.open_dir(self.index_path)
        with ix.writer() as writer:
            # Create empty commit to clear index
            pass
    
    def optimize_index(self):
        """Optimize the search index"""
        ix = index.open_dir(self.index_path)
        writer = ix.writer()
        writer.commit(optimize=True)
