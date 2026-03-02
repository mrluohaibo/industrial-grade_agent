"""
API request and response schemas for RAG document processing API.

This module defines the Pydantic models used for API requests and responses,
including validation and documentation.

Author: RAG Team
Created: 2026-03-02
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Base Response Models
# ============================================================================


class ApiResponse(BaseModel):
    """Base API response with code, message, and data."""

    code: int = Field(default=0, description="Response code, 0 for success")
    message: str = Field(default="success", description="Response message")
    data: Optional[Any] = Field(default=None, description="Response data")

    class Config:
        json_schema_extra = {
            "example": {
                "code": 0,
                "message": "success",
                "data": None
            }
        }


# ============================================================================
# Document Upload Schemas
# ============================================================================


class UploadDocumentRequest(BaseModel):
    """Request model for document upload parameters."""

    split_strategy: str = Field(
        default="recursive",
        description="Document splitting strategy",
        pattern="^(recursive|markdown_header|semantic|hybrid)$"
    )
    chunk_size: int = Field(
        default=500,
        ge=100,
        le=5000,
        description="Maximum chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=500,
        description="Chunk overlap in characters"
    )
    enable_refinement: bool = Field(
        default=True,
        description="Whether to enable semantic refinement"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "split_strategy": "recursive",
                "chunk_size": 500,
                "chunk_overlap": 50,
                "enable_refinement": True
            }
        }


class ChunkResult(BaseModel):
    """Result of a processed chunk."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    chunk_index: int = Field(..., description="Chunk index in document")
    content: str = Field(..., description="Original chunk content")
    refined_summary: str = Field(..., description="Refined summary from LLM")
    keywords: List[str] = Field(
        default_factory=list,
        description="Extracted keywords"
    )
    entities: List[str] = Field(
        default_factory=list,
        description="Extracted entities"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "1234567890123456789_0",
                "chunk_index": 0,
                "content": "Document content...",
                "refined_summary": "Summary of content...",
                "keywords": ["keyword1", "keyword2"],
                "entities": ["entity1"],
                "metadata": {
                    "split_strategy": "recursive",
                    "filename": "document.pdf"
                }
            }
        }


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""

    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    chunk_count: int = Field(..., description="Number of chunks created")
    status: str = Field(..., description="Processing status: success/partial/failed")
    chunks: List[ChunkResult] = Field(
        default_factory=list,
        description="List of processed chunks"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of errors if any"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "1234567890123456789",
                "filename": "document.pdf",
                "chunk_count": 5,
                "status": "success",
                "chunks": [],
                "errors": []
            }
        }


# ============================================================================
# Document Info Schemas
# ============================================================================


class DocumentInfoResponse(BaseModel):
    """Response model for document information."""

    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original filename")
    upload_time: str = Field(..., description="Upload timestamp")
    chunk_count: int = Field(..., description="Number of chunks")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "1234567890123456789",
                "filename": "document.pdf",
                "upload_time": "2026-03-02T10:00:00",
                "chunk_count": 5
            }
        }


class DocumentChunksResponse(BaseModel):
    """Response model for document chunks."""

    document_id: str = Field(..., description="Document ID")
    chunks: List[ChunkResult] = Field(..., description="List of chunks")
    total: int = Field(..., description="Total number of chunks")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "1234567890123456789",
                "chunks": [],
                "total": 5
            }
        }


# ============================================================================
# Document Delete Schemas
# ============================================================================


class DocumentDeleteResponse(BaseModel):
    """Response model for document deletion."""

    document_id: str = Field(..., description="Deleted document ID")
    deleted_chunks: int = Field(..., description="Number of deleted chunks")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "1234567890123456789",
                "deleted_chunks": 5
            }
        }


# ============================================================================
# Search Schemas
# ============================================================================


class SearchRequest(BaseModel):
    """Request model for RAG search."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return"
    )
    use_rerank: bool = Field(
        default=False,
        description="Whether to use reranking"
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Filter by document ID (optional)"
    )
    filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Elasticsearch filter query (optional)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "搜索内容",
                "top_k": 10,
                "use_rerank": False,
                "document_id": None,
                "filter": None
            }
        }


class SearchItem(BaseModel):
    """Single search result item."""

    document_id: str = Field(..., description="Document ID")
    chunk_id: str = Field(..., description="Chunk ID")
    chunk_index: int = Field(..., description="Chunk index")
    content: str = Field(..., description="Chunk content")
    refined_summary: str = Field(..., description="Refined summary")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "1234567890123456789",
                "chunk_id": "1234567890123456789_0",
                "chunk_index": 0,
                "content": "Matched content...",
                "refined_summary": "Summary...",
                "score": 0.95,
                "metadata": {
                    "filename": "document.pdf"
                }
            }
        }


class SearchResponse(BaseModel):
    """Response model for RAG search."""

    query: str = Field(..., description="Original search query")
    results: List[SearchItem] = Field(..., description="Search results")
    total_hits: int = Field(..., description="Total number of hits")
    search_time_ms: float = Field(..., description="Search time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "搜索内容",
                "results": [],
                "total_hits": 10,
                "search_time_ms": 150.5
            }
        }


# ============================================================================
# Health Check Schemas
# ============================================================================


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(default="1.0.0", description="API version")
    services: Dict[str, bool] = Field(
        default_factory=dict,
        description="Status of dependent services"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "services": {
                    "milvus": True,
                    "elasticsearch": True,
                    "embedding": True
                }
            }
        }


# ============================================================================
# Error Schemas
# ============================================================================


class ErrorResponse(BaseModel):
    """Response model for errors."""

    code: int = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "code": 400,
                "message": "Invalid file type",
                "details": {
                    "allowed_types": [".pdf", ".docx", ".md", ".txt"]
                }
            }
        }


# ============================================================================
# Batch Upload Schemas
# ============================================================================


class BatchUploadResult(BaseModel):
    """Result of a single file in batch upload."""

    filename: str = Field(..., description="Filename")
    document_id: Optional[str] = Field(
        default=None,
        description="Document ID if successful"
    )
    status: str = Field(..., description="Status: success/failed")
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "document1.pdf",
                "document_id": "1234567890123456789",
                "status": "success",
                "error": None
            }
        }


class BatchUploadResponse(BaseModel):
    """Response model for batch upload."""

    total_files: int = Field(..., description="Total number of files")
    successful: int = Field(..., description="Number of successfully uploaded files")
    failed: int = Field(..., description="Number of failed uploads")
    results: List[BatchUploadResult] = Field(..., description="Individual file results")

    class Config:
        json_schema_extra = {
            "example": {
                "total_files": 3,
                "successful": 2,
                "failed": 1,
                "results": []
            }
        }


# ============================================================================
# Export all schemas
# ============================================================================

__all__ = [
    # Base
    "ApiResponse",
    # Document upload
    "UploadDocumentRequest",
    "ChunkResult",
    "DocumentUploadResponse",
    # Document info
    "DocumentInfoResponse",
    "DocumentChunksResponse",
    # Document delete
    "DocumentDeleteResponse",
    # Search
    "SearchRequest",
    "SearchItem",
    "SearchResponse",
    # Health check
    "HealthResponse",
    # Error
    "ErrorResponse",
    # Batch upload
    "BatchUploadResult",
    "BatchUploadResponse",
]
