from marshmallow import Schema, fields, validate
from datetime import datetime


class ArticleSchema(Schema):
    """Schema for article data transfer objects"""
    id = fields.Str(allow_none=True)  # Allow None since we generate this in repository
    title = fields.Str(required=True, validate=validate.Length(min=1, max=500))
    summary = fields.Str(allow_none=True)
    content = fields.Str(allow_none=True)
    link = fields.Url(required=True)
    published = fields.DateTime(allow_none=True)  # Allow None and handle in service
    source = fields.Str(required=True)
    category = fields.Str(required=True)
    author = fields.Str(allow_none=True)
    tags = fields.List(fields.Str(), missing=[])


class SearchQuerySchema(Schema):
    """Schema for search query validation"""
    query = fields.Str(required=True, validate=validate.Length(min=1, max=200))
    category = fields.Str(allow_none=True)
    source = fields.Str(allow_none=True)
    page = fields.Int(missing=1, validate=validate.Range(min=1))
    per_page = fields.Int(missing=20, validate=validate.Range(min=1, max=100))
    sort_by = fields.Str(missing='relevance', validate=validate.OneOf(['relevance', 'date']))


class SearchResultSchema(Schema):
    """Schema for search results"""
    articles = fields.List(fields.Nested(ArticleSchema))
    total = fields.Int(required=True)
    page = fields.Int(required=True)
    per_page = fields.Int(required=True)
    pages = fields.Int(required=True)
    has_prev = fields.Bool(required=True)
    has_next = fields.Bool(required=True)
    query = fields.Str(required=True)


class RSSFeedSchema(Schema):
    """Schema for RSS feed configuration"""
    name = fields.Str(required=True)
    url = fields.Url(required=True)
    category = fields.Str(required=True)
    active = fields.Bool(missing=True)


class FeedUpdateStatusSchema(Schema):
    """Schema for feed update status"""
    feed_key = fields.Str(required=True)
    feed_name = fields.Str(required=True)
    status = fields.Str(required=True, validate=validate.OneOf(['success', 'error']))
    articles_count = fields.Int(missing=0)
    error_message = fields.Str(allow_none=True)
    updated_at = fields.DateTime(required=True)
