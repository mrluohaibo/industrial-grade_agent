"""
Elasticsearch document store for RAG chunks.

Stores and retrieves document chunks in Elasticsearch for full-text search.

Author: RAG Team
Created: 2026-03-02
"""

from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from bz_agent.rag.models import DocumentInfo
from utils.config_init import application_conf
from utils.logger_config import get_logger

logger = get_logger(__name__)


class DocumentESService:
    """
    Elasticsearch service for storing and retrieving document chunks.

    Provides:
    - Save chunks to Elasticsearch
    - Delete document chunks
    - Search chunks (full-text search)
    - Get document info and chunks
    """

    # Index name for document chunks
    INDEX_NAME = "rag_chunks"

    # Index mapping configuration
    INDEX_MAPPING = {
        "mappings": {
            "properties": {
                "document_id": {
                    "type": "keyword"
                },
                "chunk_id": {
                    "type": "keyword"
                },
                "chunk_index": {
                    "type": "integer"
                },
                "content": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_smart"
                },
                "refined_summary": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_smart"
                },
                "keywords": {
                    "type": "keyword"
                },
                "entities": {
                    "type": "keyword"
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "keyword"},
                        "upload_time": {"type": "date"},
                        "split_strategy": {"type": "keyword"},
                        "version": {"type": "integer"},
                        "current": {"type": "boolean"}
                    }
                },
                "version": {
                    "type": "integer"
                },
                "current": {
                    "type": "boolean"
                }
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 1,
            "analysis": {
                "analyzer": {
                    "ik_max_word": {
                        "type": "custom",
                        "tokenizer": "ik_max_word"
                    },
                    "ik_smart": {
                        "type": "custom",
                        "tokenizer": "ik_smart"
                    }
                }
            }
        }
    }

    def __init__(self, es_host: Optional[str] = None):
        """
        Initialize Elasticsearch service.

        Args:
            es_host: Elasticsearch host URL
        """
        if es_host is None:
            es_host = application_conf.get_properties("es.host")

        es_username = application_conf.get_properties("es.u_name")
        es_password = application_conf.get_properties("es.u_pwd")

        self._client = Elasticsearch(
            hosts=[es_host],
            basic_auth=(es_username, es_password),
            verify_certs=False,
            ssl_show_warn=False,
        )

        # Ensure index exists
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Create index if it doesn't exist."""
        if not self._client.indices.exists(index=self.INDEX_NAME):
            self._client.indices.create(
                index=self.INDEX_NAME,
                body=self.INDEX_MAPPING
            )
            logger.info(f"Created Elasticsearch index: {self.INDEX_NAME}")

    def save_chunk(self, chunk: Dict[str, Any]) -> bool:
        """
        Save a single chunk to Elasticsearch.

        Args:
            chunk: Chunk document dictionary

        Returns:
            True if successful, False otherwise
        """
        try:
            self._client.index(
                index=self.INDEX_NAME,
                id=chunk.get("chunk_id"),
                body=chunk
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save chunk: {e}")
            return False

    def save_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Save multiple chunks to Elasticsearch using bulk API.

        Args:
            chunks: List of chunk documents

        Returns:
            True if successful, False otherwise
        """
        if not chunks:
            return True

        try:
            actions = [
                {
                    "_index": self.INDEX_NAME,
                    "_id": chunk.get("chunk_id"),
                    "_source": chunk
                }
                for chunk in chunks
            ]

            success, failed = bulk(
                self._client,
                actions,
                raise_on_error=False
            )

            if failed:
                logger.warning(f"Some chunks failed to save: {len(failed)}")

            logger.info(f"Saved {success} chunks to Elasticsearch")
            return True

        except Exception as e:
            logger.error(f"Failed to save chunks in bulk: {e}")
            return False

    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document ID to delete

        Returns:
            Number of deleted chunks
        """
        try:
            # Query all chunks for this document
            query = {
                "query": {
                    "term": {
                        "document_id": document_id
                    }
                }
            }

            # Delete by query
            response = self._client.delete_by_query(
                index=self.INDEX_NAME,
                body=query
            )

            deleted_count = response.get("deleted", 0)
            logger.info(f"Deleted {deleted_count} chunks for document {document_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return 0

    def search_chunks(
        self,
        query: str,
        document_id: Optional[str] = None,
        top_k: int = 10,
        filter_query: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for chunks using full-text search.

        Args:
            query: Search query string
            document_id: Optional document ID to filter by
            top_k: Maximum number of results
            filter_query: Additional Elasticsearch filter query

        Returns:
            List of matching chunks with scores
        """
        try:
            # Build query
            es_query = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": [
                                        "content^2",
                                        "refined_summary^1.5",
                                        "keywords^3",
                                        "entities^2"
                                    ],
                                    "type": "best_fields",
                                    "operator": "or"
                                }
                            }
                        ]
                    }
                },
                "size": top_k
            }

            # Add document ID filter
            if document_id:
                es_query["query"]["bool"]["filter"] = [
                    {"term": {"document_id": document_id}}
                ]

            # Add custom filter
            if filter_query:
                if "filter" not in es_query["query"]["bool"]:
                    es_query["query"]["bool"]["filter"] = []
                es_query["query"]["bool"]["filter"].append(filter_query)

            # Execute search
            response = self._client.search(
                index=self.INDEX_NAME,
                body=es_query
            )

            # Process results
            results = []
            for hit in response["hits"]["hits"]:
                result = hit["_source"]
                result["_score"] = hit["_score"]
                results.append(result)

            logger.info(f"Found {len(results)} chunks for query: {query}")
            return results

        except Exception as e:
            logger.error(f"Failed to search chunks: {e}")
            return []

    def get_document_info(self, document_id: str) -> Optional[DocumentInfo]:
        """
        Get document information.

        Args:
            document_id: Document ID

        Returns:
            DocumentInfo if found, None otherwise
        """
        try:
            # Query for first chunk to get metadata
            query = {
                "query": {
                    "term": {
                        "document_id": document_id
                    }
                },
                "size": 1,
                "sort": [
                    {"chunk_index": {"order": "asc"}}
                ]
            }

            response = self._client.search(
                index=self.INDEX_NAME,
                body=query
            )

            hits = response["hits"]["hits"]
            if not hits:
                return None

            first_chunk = hits[0]["_source"]
            total_chunks = response["hits"]["total"]["value"]

            metadata = first_chunk.get("metadata", {})

            return DocumentInfo(
                document_id=document_id,
                filename=metadata.get("filename", "unknown"),
                upload_time=metadata.get("upload_time", ""),
                chunk_count=total_chunks,
            )

        except Exception as e:
            logger.error(f"Failed to get document info: {e}")
            return None

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.

        Args:
            document_id: Document ID

        Returns:
            List of chunk dictionaries
        """
        try:
            query = {
                "query": {
                    "term": {
                        "document_id": document_id
                    }
                },
                "size": 10000,  # Max chunks per document
                "sort": [
                    {"chunk_index": {"order": "asc"}}
                ]
            }

            response = self._client.search(
                index=self.INDEX_NAME,
                body=query
            )

            return [hit["_source"] for hit in response["hits"]["hits"]]

        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            return []

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific chunk by ID.

        Args:
            chunk_id: Chunk ID

        Returns:
            Chunk dictionary if found, None otherwise
        """
        try:
            response = self._client.get(
                index=self.INDEX_NAME,
                id=chunk_id
            )
            return response["_source"]
        except Exception as e:
            logger.error(f"Failed to get chunk {chunk_id}: {e}")
            return None

    def update_chunk(self, chunk_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update a chunk's data.

        Args:
            chunk_id: Chunk ID
            update_data: Data to update

        Returns:
            True if successful, False otherwise
        """
        try:
            self._client.update(
                index=self.INDEX_NAME,
                id=chunk_id,
                body={"doc": update_data}
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update chunk {chunk_id}: {e}")
            return False

    def get_document_count(self) -> int:
        """
        Get the total number of documents.

        Returns:
            Number of unique document IDs
        """
        try:
            response = self._client.search(
                index=self.INDEX_NAME,
                body={
                    "size": 0,
                    "aggs": {
                        "document_count": {
                            "cardinality": {
                                "field": "document_id"
                            }
                        }
                    }
                }
            )

            return response["aggregations"]["document_count"]["value"]

        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0

    def get_chunk_count(self) -> int:
        """
        Get the total number of chunks.

        Returns:
            Number of chunks in the index
        """
        try:
            return self._client.count(index=self.INDEX_NAME)["count"]
        except Exception as e:
            logger.error(f"Failed to get chunk count: {e}")
            return 0

    # ============================================================================
    # Version Management Methods
    # ============================================================================

    def get_document_versions(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a document.

        Args:
            document_id: Document ID

        Returns:
            List of document versions with metadata
        """
        try:
            query = {
                "query": {
                    "term": {
                        "document_id": document_id
                    }
                },
                "size": 1,  # Get one chunk per version
                "sort": [
                    {"metadata.version": {"order": "desc"}}
                ],
                "collapse": {
                    "field": "metadata.version"
                }
            }

            response = self._client.search(
                index=self.INDEX_NAME,
                body=query
            )

            # Extract version information
            versions = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                metadata = source.get("metadata", {})
                version = metadata.get("version", 1)

                # Check if we already have this version
                existing = next((v for v in versions if v["version"] == version), None)
                if existing is None:
                    versions.append({
                        "document_id": document_id,
                        "version": version,
                        "filename": metadata.get("filename", "unknown"),
                        "upload_time": metadata.get("upload_time", ""),
                        "chunk_count": 0,  # Will be updated below
                        "current": metadata.get("current", True)
                    })

                # Count chunks for this version
                version_data = next(v for v in versions if v["version"] == version)
                version_data["chunk_count"] += 1

            # Sort by version descending
            versions.sort(key=lambda x: x["version"], reverse=True)

            logger.info(f"Found {len(versions)} versions for document {document_id}")
            return versions

        except Exception as e:
            logger.error(f"Failed to get document versions: {e}")
            return []

    def mark_version_as_current(
        self, document_id: str, target_version: int
    ) -> bool:
        """
        Mark a specific version as current by deactivating others.

        Args:
            document_id: Document ID
            target_version: Version to mark as current

        Returns:
            True if successful, False otherwise
        """
        try:
            # Deactivate all versions first
            update_by_query_body = {
                "query": {
                    "term": {
                        "document_id": document_id
                    }
                },
                "script": {
                    "source": "ctx._source.current = false",
                    "lang": "painless"
                }
            }

            self._client.update_by_query(
                index=self.INDEX_NAME,
                body=update_by_query_body
            )

            # Activate target version
            target_query_body = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"document_id": document_id}},
                            {"term": {"metadata.version": target_version}}
                        ]
                    }
                },
                "script": {
                    "source": "ctx._source.current = true",
                    "lang": "painless"
                }
            }

            self._client.update_by_query(
                index=self.INDEX_NAME,
                body=target_query_body
            )

            logger.info(f"Marked version {target_version} as current for document {document_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to mark version as current: {e}")
            return False

    def delete_old_versions(
        self, document_id: str, keep_versions: int = 5
    ) -> int:
        """
        Delete old versions of a document, keeping only the most recent N.

        Args:
            document_id: Document ID
            keep_versions: Number of recent versions to keep

        Returns:
            Number of deleted versions
        """
        try:
            versions = self.get_document_versions(document_id)

            if len(versions) <= keep_versions:
                logger.info(f"No old versions to delete for document {document_id}")
                return 0

            # Versions to delete (excluding the most recent N)
            versions_to_delete = versions[keep_versions:]
            deleted_count = 0

            for version_info in versions_to_delete:
                version = version_info["version"]
                query = {
                    "query": {
                        "bool": {
                            "must": [
                                {"term": {"document_id": document_id}},
                                {"term": {"metadata.version": version}}
                            ]
                        }
                    }
                }

                response = self._client.delete_by_query(
                    index=self.INDEX_NAME,
                    body=query
                )

                deleted_count += response.get("deleted", 0)

            logger.info(f"Deleted {deleted_count} chunks for old versions of document {document_id}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete old versions: {e}")
            return 0

    def get_current_version(self, document_id: str) -> Optional[int]:
        """
        Get the current (active) version of a document.

        Args:
            document_id: Document ID

        Returns:
            Current version number, or None if not found
        """
        try:
            query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"document_id": document_id}},
                            {"term": {"metadata.current": True}}
                        ]
                    }
                },
                "size": 1,
                "sort": [
                    {"metadata.version": {"order": "desc"}}
                ]
            }

            response = self._client.search(
                index=self.INDEX_NAME,
                body=query
            )

            hits = response["hits"]["hits"]
            if hits:
                return hits[0]["_source"].get("metadata", {}).get("version", 1)

            return None

        except Exception as e:
            logger.error(f"Failed to get current version: {e}")
            return None


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "DocumentESService",
]
