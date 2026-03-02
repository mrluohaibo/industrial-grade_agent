"""
Rerank service for improving search results.

Uses BGE-Reranker to rerank search results for better relevance.

Author: RAG Team
Created: 2026-03-02
"""

from typing import Any, Dict, List, Optional

from bz_agent.rag.bge_reranker import BGEReranker

from utils.config_init import application_conf
from utils.logger_config import get_logger

logger = get_logger(__name__)


class RerankService:
    """
    Service for reranking search results using BGE-Reranker.

    Improves the relevance of search results by reranking them
    based on the query text.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the rerank service.

        Args:
            model_path: Path to the BGE-Reranker model.
        """
        self._model_path = model_path
        self._reranker = None
        self._enabled = False

        if model_path is None:
            model_path = application_conf.get_properties("milvus.bge_reranker_path")

        if model_path:
            self._init_reranker(model_path)

    def _init_reranker(self, model_path: str):
        """Initialize the BGE-Reranker model."""
        try:
            self._reranker = BGEReranker(
                model_name_or_path=model_path,
                use_fp16=True,
            )
            self._enabled = True
            logger.info(f"Reranker initialized with model: {model_path}")
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {e}")
            logger.warning("Reranking will be disabled")
            self._enabled = False

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on query relevance.

        Args:
            query: Search query string
            documents: List of document texts to rerank
            top_k: Maximum number of results to return (None for all)

        Returns:
            List of dictionaries with 'index', 'score', and 'text' keys,
            sorted by relevance score in descending order.
        """
        if not self._enabled or not self._reranker:
            # Return original order if reranking is disabled
            return [
                {
                    "index": i,
                    "score": 0.0,
                    "text": doc,
                }
                for i, doc in enumerate(documents)
            ]

        if not documents:
            return []

        try:
            # Perform reranking
            logger.debug(f"Reranking {len(documents)} documents for query: {query[:50]}...")

            results = self._reranker.rank(
                query=query,
                documents=documents,
                top_k=top_k or len(documents),
            )

            # Convert results to our format
            reranked = []
            for result in results:
                reranked.append({
                    "index": result["corpus_id"],
                    "score": result["score"],
                    "text": documents[result["corpus_id"]],
                })

            logger.debug(f"Reranking completed. Top score: {reranked[0]['score'] if reranked else 0}")
            return reranked

        except Exception as e:
            logger.error(f"Failed to rerank: {e}")
            # Return original order on failure
            return [
                {
                    "index": i,
                    "score": 0.0,
                    "text": doc,
                }
                for i, doc in enumerate(documents)
            ]

    def rerank_with_metadata(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        content_field: str = "content",
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results with metadata preserved.

        Args:
            query: Search query string
            results: List of result dictionaries with metadata
            top_k: Maximum number of results to return
            content_field: Field name containing the text content

        Returns:
            Reranked results with original metadata preserved
        """
        if not results:
            return []

        # Extract documents for reranking
        documents = [result.get(content_field, "") for result in results]

        # Perform reranking
        rerank_results = self.rerank(query, documents, top_k)

        # Map back to original results with metadata
        reranked_with_metadata = []
        seen_indices = set()

        for rerank_item in rerank_results:
            original_index = rerank_item["index"]
            if original_index < len(results) and original_index not in seen_indices:
                original_result = results[original_index].copy()
                original_result["rerank_score"] = rerank_item["score"]
                reranked_with_metadata.append(original_result)
                seen_indices.add(original_index)

        return reranked_with_metadata

    def is_enabled(self) -> bool:
        """Check if reranking is enabled."""
        return self._enabled


# ============================================================================
# Convenience functions
# ============================================================================


def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    model_path: Optional[str] = None,
    top_k: Optional[int] = None,
    content_field: str = "content",
) -> List[Dict[str, Any]]:
    """
    Convenience function to rerank search results.

    Args:
        query: Search query string
        results: List of result dictionaries
        model_path: Path to BGE-Reranker model
        top_k: Maximum number of results to return
        content_field: Field name containing text content

    Returns:
        Reranked results
    """
    service = RerankService(model_path)
    return service.rerank_with_metadata(query, results, top_k, content_field)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "RerankService",
    "rerank_results",
]
