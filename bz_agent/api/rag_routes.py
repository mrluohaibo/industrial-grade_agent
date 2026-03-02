"""
RAG search API routes.

Handles semantic search and retrieval for RAG applications.

Author: RAG Team
Created: 2026-03-02
"""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status

from bz_agent.api.schemas import ApiResponse, SearchItem, SearchResponse
from bz_agent.rag.document_processor import DocumentProcessor
from utils.logger_config import get_logger

logger = get_logger(__name__)

# Create router
rag_router = APIRouter(prefix="/api/v1/rag", tags=["RAG Search"])

# Global document processor instance (initialized on startup)
_processor: Optional[DocumentProcessor] = None


def get_processor() -> DocumentProcessor:
    """Get or create the document processor instance."""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor


@rag_router.get(
    "/search",
    response_model=ApiResponse,
    summary="RAG search",
    description="Search for relevant document chunks using semantic search"
)
async def search(
    query: str = Query(..., min_length=1, max_length=1000, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=100, description="Number of results"),
    use_rerank: bool = Query(default=False, description="Use reranking"),
    document_id: Optional[str] = Query(default=None, description="Filter by document ID"),
):
    """
    Search for relevant document chunks.

    Performs semantic search using vector embeddings. Optionally can
    use reranking to improve result quality.
    """
    start_time = time.time()

    try:
        processor = get_processor()

        logger.info(f"Searching with query: '{query}', top_k={top_k}, use_rerank={use_rerank}")

        # Perform search
        search_result = processor.search_documents(
            query=query,
            top_k=top_k,
            use_rerank=use_rerank,
        )

        # Filter by document_id if specified
        if document_id:
            search_result.results = [
                r for r in search_result.results if r.document_id == document_id
            ]
            search_result.total_hits = len(search_result.results)

        search_time = time.time() - start_time
        search_result.search_time_ms = search_time * 1000

        # Convert to API response format
        results = [
            SearchItem(
                document_id=item.document_id,
                chunk_id=item.chunk_id,
                chunk_index=item.chunk_index,
                content=item.content,
                refined_summary=item.refined_summary,
                score=item.score,
                metadata=item.metadata,
            )
            for item in search_result.results
        ]

        logger.info(f"Search completed: {len(results)} results found in {search_time*1000:.2f}ms")

        return ApiResponse(
            code=0,
            message="Search completed successfully",
            data=SearchResponse(
                query=search_result.query,
                results=results,
                total_hits=search_result.total_hits,
                search_time_ms=search_result.search_time_ms,
            ).model_dump(),
        )

    except ValueError as e:
        logger.error(f"Search validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@rag_router.post(
    "/search",
    response_model=ApiResponse,
    summary="RAG search (POST)",
    description="Search for relevant document chunks using semantic search (POST method)"
)
async def search_post(
    query: str = Query(..., min_length=1, max_length=1000, description="Search query"),
    top_k: int = Query(default=10, ge=1, le=100, description="Number of results"),
    use_rerank: bool = Query(default=False, description="Use reranking"),
    document_id: Optional[str] = Query(default=None, description="Filter by document ID"),
):
    """
    Search for relevant document chunks (POST method).

    Same functionality as GET /search but allows for longer queries
    or future expansion with request body.
    """
    return await search(
        query=query,
        top_k=top_k,
        use_rerank=use_rerank,
        document_id=document_id,
    )


@rag_router.get(
    "/health",
    response_model=ApiResponse,
    summary="RAG service health check",
    description="Check health of RAG search service"
)
async def health_check():
    """
    Health check for RAG search service.

    Returns the status of Milvus, Elasticsearch, and embedding model.
    """
    try:
        processor = get_processor()

        # Check Milvus
        milvus_healthy = False
        try:
            from pymilvus import utility
            collections = utility.list_collections()
            milvus_healthy = collections is not None
        except Exception as e:
            logger.warning(f"Milvus health check failed: {e}")

        # Check Elasticsearch
        es_healthy = False
        try:
            if processor._es_service:
                es_healthy = processor._es_service._client.ping()
        except Exception as e:
            logger.warning(f"Elasticsearch health check failed: {e}")

        # Check embedding model
        embedding_healthy = processor._embedding_model is not None

        services_status = {
            "milvus": milvus_healthy,
            "elasticsearch": es_healthy,
            "embedding": embedding_healthy,
        }

        overall_status = "healthy" if all(services_status.values()) else "degraded"

        return ApiResponse(
            code=0,
            message="Service health check completed",
            data={
                "status": overall_status,
                "version": "1.0.0",
                "services": services_status,
            }
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


# ============================================================================
# Export router
# ============================================================================

__all__ = ["rag_router"]
