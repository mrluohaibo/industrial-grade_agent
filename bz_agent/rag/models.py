"""
Data models for RAG document processing.

This module defines the data structures used throughout the document
processing pipeline, including chunk information, refinement results,
and document processing results.

Author: RAG Team
Created: 2026-03-02
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


# ============================================================================
# Data Models (for internal processing)
# ============================================================================


@dataclass
class ChunkInfo:
    """Information about a document chunk."""

    document_id: str
    chunk_id: str
    chunk_index: int
    original_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RefinementResult:
    """Result of semantic refinement for a chunk."""

    chunk_id: str
    refined_summary: str
    keywords: List[str]
    entities: List[str] = field(default_factory=list)


@dataclass
class ChunkDocument:
    """
    Complete chunk document with all processed data.

    This is the final data structure that gets stored in both
    Milvus (with embedding) and Elasticsearch.
    """

    document_id: str
    chunk_id: str
    chunk_index: int
    original_content: str
    refined_summary: str
    keywords: List[str]
    entities: List[str]
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_milvus_dict(self) -> Dict[str, Any]:
        """Convert to Milvus storage format."""
        return {
            "id": int(self.chunk_id.split("_")[-1]),
            "document_id": self.document_id,
            "origin_content": self.original_content,
            "vector": self.embedding,
        }

    def to_es_dict(self) -> Dict[str, Any]:
        """Convert to Elasticsearch storage format."""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "content": self.original_content,
            "refined_summary": self.refined_summary,
            "keywords": self.keywords,
            "entities": self.entities,
            "metadata": self.metadata,
        }


@dataclass
class DocumentProcessResult:
    """Result of processing a document."""

    document_id: str
    filename: str
    chunk_count: int
    status: str  # "success" | "partial" | "failed"
    version: int = 1  # Document version number
    chunks: List[ChunkDocument] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class DocumentInfo:
    """Basic information about a document."""

    document_id: str
    filename: str
    upload_time: datetime
    chunk_count: int
    version: int = 1  # Document version number


@dataclass
class DocumentVersionInfo:
    """Information about a document version."""

    document_id: str
    version: int
    filename: str
    upload_time: datetime
    chunk_count: int
    current: bool = False  # Whether this is the current version


@dataclass
class SearchResultItem:
    """Single search result item."""

    document_id: str
    chunk_id: str
    chunk_index: int
    content: str
    refined_summary: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Search query result."""

    query: str
    results: List[SearchResultItem]
    total_hits: int = 0
    search_time_ms: float = 0.0


# ============================================================================
# Pydantic Models (for API requests/responses)
# ============================================================================


class ChunkResult(BaseModel):
    """Chunk result for API response."""

    chunk_id: str
    chunk_index: int
    content: str
    refined_summary: str
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "1234567890123456789_0",
                "chunk_index": 0,
                "content": "原文内容...",
                "refined_summary": "摘要内容...",
                "keywords": ["关键词1", "关键词2"],
                "entities": ["实体1"],
                "metadata": {"split_strategy": "recursive"},
            }
        }


class DocumentProcessResultResponse(BaseModel):
    """Document processing result for API response."""

    document_id: str
    filename: str
    chunk_count: int
    status: str
    chunks: List[ChunkResult] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "1234567890123456789",
                "filename": "document.pdf",
                "chunk_count": 5,
                "status": "success",
                "chunks": [],
                "errors": [],
            }
        }


class DocumentInfoResponse(BaseModel):
    """Document information for API response."""

    document_id: str
    filename: str
    upload_time: str
    chunk_count: int

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "1234567890123456789",
                "filename": "document.pdf",
                "upload_time": "2026-03-02T10:00:00",
                "chunk_count": 5,
            }
        }


class SearchItem(BaseModel):
    """Search result item for API response."""

    document_id: str
    chunk_id: str
    chunk_index: int
    content: str
    refined_summary: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "1234567890123456789",
                "chunk_id": "1234567890123456789_0",
                "chunk_index": 0,
                "content": "相关内容...",
                "refined_summary": "摘要...",
                "score": 0.95,
                "metadata": {},
            }
        }


class SearchResponse(BaseModel):
    """Search response for API."""

    query: str
    results: List[SearchItem]
    total_hits: int
    search_time_ms: float

    class Config:
        json_schema_extra = {
            "example": {
                "query": "搜索关键词",
                "results": [],
                "total_hits": 10,
                "search_time_ms": 150.5,
            }
        }


# ============================================================================
# Export all models
# ============================================================================

__all__ = [
    # Data models
    "ChunkInfo",
    "RefinementResult",
    "ChunkDocument",
    "DocumentProcessResult",
    "DocumentInfo",
    "SearchResultItem",
    "SearchResult",
    # Pydantic models
    "ChunkResult",
    "DocumentProcessResultResponse",
    "DocumentInfoResponse",
    "SearchItem",
    "SearchResponse",
]
