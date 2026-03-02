"""
Document processing service - Core orchestration for RAG pipeline.

This service orchestrates the entire document processing workflow:
- File parsing
- Document splitting
- Semantic refinement
- Vectorization
- Dual storage (Milvus + Elasticsearch)

Author: RAG Team
Created: 2026-03-02
"""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from pymilvus import utility

from bz_agent.rag.document_splitter import DocumentSplitter, SplitResult
from bz_agent.rag.file_parser import FileParser
from bz_agent.rag.models import (
    ChunkDocument,
    DocumentInfo,
    DocumentProcessResult,
    SearchResult,
    SearchResultItem,
)
from bz_agent.rag.semantic_refiner import (
    RefinementConfig,
    SemanticRefiner,
)
from bz_agent.rag.save_embedding_to_milvus import (
    MilvusAndEmbeddingClient,
)
from bz_agent.rag.embedding_data_handler import DataEmbeddingOrm
from FlagEmbedding import BGEM3FlagModel
from utils.config_init import application_conf
from utils.logger_config import get_logger
from utils.snow_flake import snowflake

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Document processing service for RAG pipeline.

    Handles complete document lifecycle: upload, process, store, delete.
    """

    # Milvus collection name for chunks
    COLLECTION_NAME = "rag_chunks"

    def __init__(
        self,
        milvus_url: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        refinement_config: Optional[RefinementConfig] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """
        Initialize the document processor.

        Args:
            milvus_url: Milvus connection URL
            embedding_model_path: Path to BGE-M3 model
            refinement_config: Configuration for semantic refinement
            progress_callback: Callback for progress updates (current, total, message)
        """
        self.progress_callback = progress_callback

        # Load configuration
        self.storage_path = application_conf.get_properties(
            "document.storage_path", "./data/documents"
        )
        self.max_file_size = application_conf.get_properties(
            "document.max_file_size", 10485760  # 10MB
        )
        self.allowed_extensions = application_conf.get_properties(
            "document.allowed_extensions", [".pdf", ".docx", ".md", ".txt"]
        )

        # Initialize components
        self._file_parser = FileParser(allowed_extensions=self.allowed_extensions)
        self._refiner = SemanticRefiner(refinement_config)

        # Initialize Milvus and embedding model
        self._init_milvus_and_embedding(milvus_url, embedding_model_path)

        # Initialize Elasticsearch service
        self._init_elasticsearch()

    def _init_milvus_and_embedding(
        self, milvus_url: Optional[str], embedding_model_path: Optional[str]
    ):
        """Initialize Milvus client and embedding model."""
        try:
            if milvus_url is None:
                milvus_url = application_conf.get_properties("milvus.ip")
                milvus_port = application_conf.get_properties("milvus.port", 19530)
                milvus_url = f"http://{milvus_url}:{milvus_port}"

            if embedding_model_path is None:
                embedding_model_path = application_conf.get_properties(
                    "milvus.bge_m3_model_path"
                )

            self._milvus_client = MilvusAndEmbeddingClient(milvus_url=milvus_url)

            # Initialize embedding model
            self._embedding_model = BGEM3FlagModel(
                model_name_or_path=embedding_model_path,
                use_fp16=True,
            )

            # Initialize embedding handler
            self._embedding_handler = DataEmbeddingOrm(
                self._encode_embeddings, self._milvus_client
            )

            # Ensure collection exists
            self._ensure_collection_exists()

            # Initialize reranker
            self._init_reranker()

            logger.info("Milvus and embedding model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Milvus and embedding: {e}")
            raise

    def _init_elasticsearch(self):
        """Initialize Elasticsearch service."""
        try:
            from bz_agent.rag.es_document_store import DocumentESService

            self._es_service = DocumentESService()
            logger.info("Elasticsearch service initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Elasticsearch: {e}")
            self._es_service = None

    def _init_reranker(self):
        """Initialize BGE-Reranker for result reranking."""
        try:
            from bz_agent.rag.rerank_service import RerankService

            self._reranker = RerankService()
            if self._reranker.is_enabled():
                logger.info("Reranker service initialized successfully")
            else:
                logger.info("Reranker service is disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {e}")
            self._reranker = None

    def _encode_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings using BGE-M3 model."""
        result = self._embedding_model.encode(
            sentences=texts, return_dense=True, return_sparse=False
        )
        return result["dense_vecs"]

    def _ensure_collection_exists(self):
        """Ensure Milvus collection exists, create if not."""
        if not utility.has_collection(self.COLLECTION_NAME):
            # Create collection schema
            from pymilvus import CollectionSchema, DataType, FieldSchema

            id_field = FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,
                auto_id=False,
            )

            document_id_field = FieldSchema(
                name="document_id",
                dtype=DataType.VARCHAR,
                max_length=128,
            )

            content_field = FieldSchema(
                name="origin_content",
                dtype=DataType.VARCHAR,
                max_length=65535,
            )

            embedding_dim = 1024  # BGE-M3 dimension
            vector_field = FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=embedding_dim,
            )

            schema = CollectionSchema(
                fields=[id_field, document_id_field, content_field, vector_field],
                description="RAG chunks collection",
            )

            self._milvus_client.client.create_collection(
                collection_name=self.COLLECTION_NAME, schema=schema
            )
            logger.info(f"Created Milvus collection: {self.COLLECTION_NAME}")

    def process_document(
        self,
        file_bytes: bytes,
        filename: str,
        split_strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        enable_refinement: bool = True,
    ) -> DocumentProcessResult:
        """
        Process a document through the complete RAG pipeline.

        Args:
            file_bytes: File content as bytes
            filename: Original filename
            split_strategy: Splitting strategy
            chunk_size: Maximum chunk size
            chunk_overlap: Chunk overlap size
            enable_refinement: Whether to enable semantic refinement

        Returns:
            DocumentProcessResult with processed chunks and status
        """
        self._report_progress(0, 5, "Starting document processing")

        # Step 1: Generate document ID
        document_id = str(snowflake.generate_id())
        self._report_progress(1, 5, f"Generated document ID: {document_id}")

        errors = []

        try:
            # Step 2: Parse file content
            self._report_progress(2, 5, "Parsing file content")
            text_content = self._file_parser.parse_file_bytes(file_bytes, filename)

            if not text_content.strip():
                raise ValueError("File content is empty")

            # Step 3: Split document into chunks
            self._report_progress(3, 5, "Splitting document into chunks")
            splitter = DocumentSplitter(
                strategy=split_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            split_results = splitter.split_text(text_content, document_id=document_id)

            if not split_results:
                raise ValueError("No chunks generated from document")

            # Step 4: Refine chunks (if enabled)
            self._report_progress(4, 5, f"Refining {len(split_results)} chunks")
            refined_chunks = self._refine_chunks(
                split_results, enable_refinement
            )

            # Step 5: Vectorize and store
            self._report_progress(5, 5, "Vectorizing and storing chunks")
            stored_chunks = self._store_chunks(
                refined_chunks, document_id, filename
            )

            self._report_progress(5, 5, "Document processing completed")

            return DocumentProcessResult(
                document_id=document_id,
                filename=filename,
                chunk_count=len(stored_chunks),
                status="success",
                chunks=stored_chunks,
                errors=errors,
            )

        except Exception as e:
            logger.error(f"Failed to process document {filename}: {e}")
            errors.append(str(e))

            # Attempt to clean up on failure
            try:
                self.delete_document(document_id)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup after error: {cleanup_error}")

            return DocumentProcessResult(
                document_id=document_id,
                filename=filename,
                chunk_count=0,
                status="failed",
                chunks=[],
                errors=errors,
            )

    def _refine_chunks(
        self,
        split_results: List[SplitResult],
        enable_refinement: bool,
    ) -> List[ChunkDocument]:
        """Refine chunks using semantic refiner."""
        if not enable_refinement:
            # Create basic ChunkDocument without refinement
            return [
                ChunkDocument(
                    document_id=result.metadata.get("document_id", ""),
                    chunk_id=result.chunk_id,
                    chunk_index=result.index,
                    original_content=result.text,
                    refined_summary=result.text[:200],
                    keywords=[],
                    entities=[],
                    embedding=[],  # Will be filled during storage
                    metadata=result.metadata,
                )
                for result in split_results
            ]

        # Prepare chunks for refinement
        chunks_to_refine = [(result.chunk_id, result.text) for result in split_results]

        # Refine all chunks
        refinement_results = self._refiner.refine_chunks(chunks_to_refine)

        # Create ChunkDocument with refinement
        chunk_documents = []
        for split_result, refinement_result in zip(split_results, refinement_results):
            if refinement_result.success:
                chunk_documents.append(
                    ChunkDocument(
                        document_id=split_result.metadata.get("document_id", ""),
                        chunk_id=split_result.chunk_id,
                        chunk_index=split_result.index,
                        original_content=split_result.text,
                        refined_summary=refinement_result.refined_summary,
                        keywords=refinement_result.keywords,
                        entities=refinement_result.entities,
                        embedding=[],  # Will be filled during storage
                        metadata=split_result.metadata,
                    )
                )
            else:
                # Fallback for failed refinement
                chunk_documents.append(
                    ChunkDocument(
                        document_id=split_result.metadata.get("document_id", ""),
                        chunk_id=split_result.chunk_id,
                        chunk_index=split_result.index,
                        original_content=split_result.text,
                        refined_summary=split_result.text[:200],
                        keywords=[],
                        entities=[],
                        embedding=[],
                        metadata=split_result.metadata,
                    )
                )

        return chunk_documents

    def _store_chunks(
        self,
        chunks: List[ChunkDocument],
        document_id: str,
        filename: str,
    ) -> List[ChunkDocument]:
        """
        Store chunks in both Milvus and Elasticsearch.

        Args:
            chunks: List of ChunkDocument to store
            document_id: Document ID
            filename: Original filename

        Returns:
            List of stored ChunkDocument with embeddings
        """
        if not chunks:
            return []

        # Step 1: Generate embeddings
        texts = [chunk.original_content for chunk in chunks]
        embeddings = self._encode_embeddings(texts)

        # Step 2: Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            chunk.metadata["filename"] = filename
            chunk.metadata["upload_time"] = datetime.now().isoformat()

        # Step 3: Store in Milvus
        try:
            for chunk in chunks:
                milvus_data = chunk.to_milvus_dict()
                self._milvus_client.client.insert(
                    collection_name=self.COLLECTION_NAME,
                    data=[milvus_data]
                )

            # Flush to ensure data is persisted
            self._milvus_client.client.flush(self.COLLECTION_NAME)
            logger.info(f"Stored {len(chunks)} chunks in Milvus")
        except Exception as e:
            logger.error(f"Failed to store chunks in Milvus: {e}")
            raise

        # Step 4: Store in Elasticsearch
        if self._es_service:
            try:
                es_documents = [chunk.to_es_dict() for chunk in chunks]
                self._es_service.save_chunks(es_documents)
                logger.info(f"Stored {len(chunks)} chunks in Elasticsearch")
            except Exception as e:
                logger.warning(f"Failed to store chunks in Elasticsearch: {e}")

        return chunks

    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks.

        Args:
            document_id: Document ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete from Milvus
            self._delete_from_milvus(document_id)

            # Delete from Elasticsearch
            if self._es_service:
                self._es_service.delete_document(document_id)

            logger.info(f"Deleted document: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    def _delete_from_milvus(self, document_id: str):
        """Delete document chunks from Milvus."""
        # Query all chunks for this document
        expr = f'document_id == "{document_id}"'
        self._milvus_client.client.delete(
            collection_name=self.COLLECTION_NAME,
            expr=expr
        )
        self._milvus_client.client.flush(self.COLLECTION_NAME)

    def get_document_info(self, document_id: str) -> Optional[DocumentInfo]:
        """
        Get document information.

        Args:
            document_id: Document ID

        Returns:
            DocumentInfo if found, None otherwise
        """
        try:
            # Query from Elasticsearch
            if self._es_service:
                return self._es_service.get_document_info(document_id)

            # Fallback: query from Milvus

            results = self._milvus_client.client.query(
                collection_name=self.COLLECTION_NAME,
                expr=f'document_id == "{document_id}"',
                limit=1,
                output_fields=["document_id"]
            )

            if results:
                return DocumentInfo(
                    document_id=document_id,
                    filename="unknown",
                    upload_time=datetime.now(),
                    chunk_count=0,
                )

            return None
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
            if self._es_service:
                return self._es_service.get_document_chunks(document_id)

            return []
        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            return []

    def search_documents(
        self,
        query: str,
        top_k: int = 10,
        use_rerank: bool = False,
    ) -> SearchResult:
        """
        Search for relevant document chunks.

        Args:
            query: Search query
            top_k: Number of results to return
            use_rerank: Whether to use reranking

        Returns:
            SearchResult with matched chunks
        """
        try:
            # Generate query embedding
            query_embedding = self._encode_embeddings([query])[0]

            # Search in Milvus
            search_results = self._milvus_client.client.search(
                collection_name=self.COLLECTION_NAME,
                data=[query_embedding],
                limit=top_k * 2 if use_rerank else top_k,  # Get more for reranking
                output_fields=["document_id", "origin_content"]
            )

            # Convert to SearchResultItem format
            results = []
            for hit in search_results[0]:
                results.append({
                    "document_id": hit.entity.get("document_id", ""),
                    "chunk_id": f"{hit.entity.get('document_id', '')}_{hit.id}",
                    "chunk_index": hit.id,
                    "content": hit.entity.get("origin_content", ""),
                    "refined_summary": "",
                    "score": hit.distance,
                    "metadata": {}
                })

            # Apply reranking if enabled
            if use_rerank and self._reranker and self._reranker.is_enabled():
                logger.debug(f"Applying reranking to {len(results)} results")
                results = self._reranker.rerank_with_metadata(
                    query=query,
                    results=results,
                    top_k=top_k,
                    content_field="content",
                )

            # Convert final results to SearchResultItem
            final_results = [
                SearchResultItem(
                    document_id=r.get("document_id", ""),
                    chunk_id=r.get("chunk_id", ""),
                    chunk_index=r.get("chunk_index", 0),
                    content=r.get("content", ""),
                    refined_summary=r.get("refined_summary", ""),
                    score=r.get("rerank_score", r.get("score", 0)),
                    metadata=r.get("metadata", {})
                )
                for r in results
            ]

            return SearchResult(
                query=query,
                results=final_results,
                total_hits=len(final_results),
                search_time_ms=0,  # TODO: measure actual time
            )

        except Exception as e:
            logger.error(f"Failed to search documents: {e}")
            return SearchResult(
                query=query,
                results=[],
                total_hits=0,
                search_time_ms=0,
            )

    def update_document(
        self,
        document_id: str,
        file_bytes: bytes,
        filename: str,
        split_strategy: str = "recursive",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        enable_refinement: bool = True,
    ) -> DocumentProcessResult:
        """
        Update a document by creating a new version.

        Args:
            document_id: Document ID to update
            file_bytes: New file content
            filename: New filename
            split_strategy: Splitting strategy
            chunk_size: Maximum chunk size
            chunk_overlap: Chunk overlap size
            enable_refinement: Whether to enable semantic refinement

        Returns:
            DocumentProcessResult with updated version info
        """
        self._report_progress(0, 6, "Starting document update")

        errors = []

        try:
            # Step 1: Get current version
            self._report_progress(1, 6, "Getting current document version")
            current_version = 1
            if self._es_service:
                current_version = self._es_service.get_current_version(document_id) or 1

            new_version = current_version + 1

            # Step 2: Mark current version as inactive
            self._report_progress(2, 6, f"Deactivating version {current_version}")
            if self._es_service:
                self._es_service.mark_version_as_current(
                    document_id, -1
                )  # Deactivate all

            # Step 3: Delete old chunks from Milvus (keeping in ES for history)
            self._report_progress(3, 6, "Deleting old chunks from Milvus")
            self._delete_from_milvus(document_id)

            # Step 4: Parse file content
            self._report_progress(4, 6, "Parsing file content")
            text_content = self._file_parser.parse_file_bytes(file_bytes, filename)

            if not text_content.strip():
                raise ValueError("File content is empty")

            # Step 5: Split document into chunks
            self._report_progress(5, 6, "Splitting document into chunks")
            splitter = DocumentSplitter(
                strategy=split_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            split_results = splitter.split_text(text_content, document_id=document_id)

            if not split_results:
                raise ValueError("No chunks generated from document")

            # Step 6: Refine chunks (if enabled)
            self._report_progress(6, 6, f"Refining {len(split_results)} chunks")
            refined_chunks = self._refine_chunks(
                split_results, enable_refinement
            )

            # Add version information to chunks
            for chunk in refined_chunks:
                chunk.metadata["version"] = new_version
                chunk.metadata["current"] = True
                chunk.metadata["filename"] = filename

            # Step 7: Vectorize and store
            self._report_progress(6, 6, "Vectorizing and storing chunks")
            stored_chunks = self._store_chunks(
                refined_chunks, document_id, filename
            )

            # Step 8: Mark new version as current
            if self._es_service:
                self._es_service.mark_version_as_current(
                    document_id, new_version
                )

            self._report_progress(6, 6, "Document update completed")

            return DocumentProcessResult(
                document_id=document_id,
                filename=filename,
                chunk_count=len(stored_chunks),
                status="success",
                version=new_version,
                chunks=stored_chunks,
                errors=errors,
            )

        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            errors.append(str(e))

            return DocumentProcessResult(
                document_id=document_id,
                filename=filename,
                chunk_count=0,
                status="failed",
                version=current_version,
                chunks=[],
                errors=errors,
            )

    def get_document_versions(
        self, document_id: str
    ) -> List[Any]:
        """
        Get all versions of a document.

        Args:
            document_id: Document ID

        Returns:
            List of document versions
        """
        try:
            if self._es_service:
                return self._es_service.get_document_versions(document_id)
            return []
        except Exception as e:
            logger.error(f"Failed to get document versions: {e}")
            return []

    def rollback_document(
        self, document_id: str, target_version: int
    ) -> bool:
        """
        Rollback a document to a specific version.

        Args:
            document_id: Document ID
            target_version: Version to rollback to

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Rolling back document {document_id} to version {target_version}")

            # Step 1: Get chunks for the target version
            if not self._es_service:
                logger.error("Elasticsearch service not available for rollback")
                return False

            versions = self._es_service.get_document_versions(document_id)
            target_version_info = next(
                (v for v in versions if v["version"] == target_version), None
            )

            if target_version_info is None:
                logger.error(f"Version {target_version} not found for document {document_id}")
                return False

            # Step 2: Get all chunks for the target version
            all_chunks = self._es_service.get_document_chunks(document_id)
            target_chunks = [
                c for c in all_chunks
                if c.get("metadata", {}).get("version") == target_version
            ]

            if not target_chunks:
                logger.error(f"No chunks found for version {target_version}")
                return False

            # Step 3: Delete current chunks from Milvus
            self._delete_from_milvus(document_id)

            # Step 4: Re-insert target version chunks to Milvus
            for chunk in target_chunks:
                # Generate embedding for chunk content
                embeddings = self._encode_embeddings([chunk["content"]])
                chunk_id = chunk.get("chunk_id", "")

                # Prepare Milvus data
                milvus_data = {
                    "id": int(chunk_id.split("_")[-1]),
                    "document_id": document_id,
                    "origin_content": chunk.get("content", ""),
                    "vector": embeddings[0].tolist() if hasattr(embeddings[0], 'tolist') else embeddings[0],
                }

                self._milvus_client.client.insert(
                    collection_name=self.COLLECTION_NAME,
                    data=[milvus_data]
                )

            self._milvus_client.client.flush(self.COLLECTION_NAME)

            # Step 5: Mark target version as current
            self._es_service.mark_version_as_current(document_id, target_version)

            logger.info(f"Successfully rolled back document {document_id} to version {target_version}")
            return True

        except Exception as e:
            logger.error(f"Failed to rollback document {document_id} to version {target_version}: {e}")
            return False

    def _report_progress(self, current: int, total: int, message: str):
        """Report progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(current, total, message)


# ============================================================================
# Export
# ============================================================================

__all__ = [
    "DocumentProcessor",
]
