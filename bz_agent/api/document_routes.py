"""
Document API routes for RAG document processing.

Handles document upload, deletion, and query endpoints.

Author: RAG Team
Created: 2026-03-02
"""

import time
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from bz_agent.api.schemas import (
    ApiResponse,
    ChunkResult,
    DocumentChunksResponse,
    DocumentDeleteResponse,
    DocumentInfoResponse,
    DocumentRollbackResponse,
    DocumentUpdateResponse,
    DocumentUploadResponse,
    DocumentVersionInfoResponse,
    DocumentVersionsResponse,
)
from bz_agent.rag.document_processor import DocumentProcessor
from utils.config_init import application_conf
from utils.logger_config import get_logger

logger = get_logger(__name__)

# Create router
document_router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

# Global document processor instance (initialized on startup)
_processor: Optional[DocumentProcessor] = None


def get_processor() -> DocumentProcessor:
    """Get or create the document processor instance."""
    global _processor
    if _processor is None:
        _processor = DocumentProcessor()
    return _processor


@document_router.post(
    "/upload",
    response_model=ApiResponse,
    summary="Upload a document",
    description="Upload and process a document (PDF, DOCX, MD, TXT) for RAG"
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    split_strategy: str = Form(default="recursive", description="Splitting strategy"),
    chunk_size: int = Form(default=500, ge=100, le=5000, description="Chunk size"),
    chunk_overlap: int = Form(default=50, ge=0, le=500, description="Chunk overlap"),
    enable_refinement: bool = Form(default=True, description="Enable semantic refinement"),
):
    """
    Upload and process a document.

    The document will be parsed, split into chunks, refined, and stored
    in both Milvus (for vector search) and Elasticsearch (for full-text search).
    """
    start_time = time.time()

    # Validate file type
    allowed_extensions = application_conf.get_properties(
        "document.allowed_extensions", [".pdf", ".docx", ".md", ".txt"]
    )
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""

    if file_ext not in allowed_extensions:
        logger.warning(f"Unsupported file type: {file_ext}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )

    # Validate file size
    max_file_size = application_conf.get_properties("document.max_file_size", 10485760)
    file_content = await file.read()

    if len(file_content) > max_file_size:
        logger.warning(f"File too large: {len(file_content)} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {max_file_size} bytes"
        )

    try:
        # Get processor
        processor = get_processor()

        # Process document
        logger.info(f"Processing document: {file.filename}")
        result = processor.process_document(
            file_bytes=file_content,
            filename=file.filename,
            split_strategy=split_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enable_refinement=enable_refinement,
        )

        processing_time = time.time() - start_time
        logger.info(
            f"Document processed successfully: {file.filename} "
            f"(chunks: {result.chunk_count}, time: {processing_time:.2f}s)"
        )

        # Convert chunks to response format
        chunks_response = [
            ChunkResult(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                content=chunk.original_content,
                refined_summary=chunk.refined_summary,
                keywords=chunk.keywords,
                entities=chunk.entities,
                metadata=chunk.metadata,
            )
            for chunk in result.chunks
        ]

        upload_response = DocumentUploadResponse(
            document_id=result.document_id,
            filename=result.filename,
            chunk_count=result.chunk_count,
            status=result.status,
            chunks=chunks_response,
            errors=result.errors,
        )

        return ApiResponse(
            code=0,
            message="Document uploaded and processed successfully",
            data=upload_response.model_dump(),
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to process document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process document: {str(e)}"
        )


@document_router.delete(
    "/{document_id}",
    response_model=ApiResponse,
    summary="Delete a document",
    description="Delete a document and all its chunks from storage"
)
async def delete_document(document_id: str):
    """
    Delete a document and all its associated chunks.

    This will remove the document from both Milvus and Elasticsearch.
    """
    try:
        processor = get_processor()

        logger.info(f"Deleting document: {document_id}")
        deleted = processor.delete_document(document_id)

        if deleted:
            # Get chunk count for response
            es_service = processor._es_service
            deleted_chunks = 0
            if es_service:
                # Since we already deleted, we can't get exact count
                deleted_chunks = -1  # Indicates success but count unknown

            logger.info(f"Document deleted successfully: {document_id}")

            return ApiResponse(
                code=0,
                message="Document deleted successfully",
                data=DocumentDeleteResponse(
                    document_id=document_id,
                    deleted_chunks=deleted_chunks,
                ).model_dump(),
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@document_router.get(
    "/{document_id}",
    response_model=ApiResponse,
    summary="Get document info",
    description="Get information about a specific document"
)
async def get_document(document_id: str):
    """
    Get document information including filename, upload time, and chunk count.
    """
    try:
        processor = get_processor()

        logger.info(f"Getting document info: {document_id}")
        doc_info = processor.get_document_info(document_id)

        if doc_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}"
            )

        return ApiResponse(
            code=0,
            message="Document found",
            data=DocumentInfoResponse(
                document_id=doc_info.document_id,
                filename=doc_info.filename,
                upload_time=doc_info.upload_time if isinstance(doc_info.upload_time, str) else doc_info.upload_time.isoformat(),
                chunk_count=doc_info.chunk_count,
            ).model_dump(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document: {str(e)}"
        )


@document_router.get(
    "/{document_id}/chunks",
    response_model=ApiResponse,
    summary="Get document chunks",
    description="Get all chunks for a specific document"
)
async def get_document_chunks(document_id: str):
    """
    Get all chunks for a document.

    Returns the full content of each chunk along with metadata.
    """
    try:
        processor = get_processor()

        logger.info(f"Getting chunks for document: {document_id}")
        chunks_data = processor.get_document_chunks(document_id)

        if not chunks_data:
            # Check if document exists
            doc_info = processor.get_document_info(document_id)
            if doc_info is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document not found: {document_id}"
                )
            # Document exists but has no chunks
            chunks_data = []

        # Convert to response format
        chunks_response = [
            ChunkResult(
                chunk_id=chunk.get("chunk_id", ""),
                chunk_index=chunk.get("chunk_index", 0),
                content=chunk.get("content", ""),
                refined_summary=chunk.get("refined_summary", ""),
                keywords=chunk.get("keywords", []),
                entities=chunk.get("entities", []),
                metadata=chunk.get("metadata", {}),
            )
            for chunk in chunks_data
        ]

        return ApiResponse(
            code=0,
            message=f"Found {len(chunks_response)} chunks",
            data=DocumentChunksResponse(
                document_id=document_id,
                chunks=chunks_response,
                total=len(chunks_response),
            ).model_dump(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chunks for document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document chunks: {str(e)}"
        )


@document_router.get(
    "/",
    response_model=ApiResponse,
    summary="List documents",
    description="Get a list of all documents (paginated)"
)
async def list_documents(
    limit: int = 100,
    offset: int = 0,
):
    """
    Get a list of all documents.

    This endpoint returns document metadata without the chunk contents.
    """
    try:
        processor = get_processor()
        es_service = processor._es_service

        if es_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Elasticsearch service not available"
            )

        # Get document count
        document_count = es_service.get_document_count()

        # For simplicity, return count only
        # Full pagination would require ES aggregation
        return ApiResponse(
            code=0,
            message=f"Found {document_count} documents",
            data={
                "total": document_count,
                "limit": limit,
                "offset": offset,
                # TODO: Implement actual document listing with ES aggregation
                "documents": []
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list documents: {str(e)}"
        )


@document_router.post(
    "/batch",
    response_model=ApiResponse,
    summary="Batch upload documents",
    description="Upload multiple documents at once"
)
async def batch_upload_documents(
    files: List[UploadFile] = File(..., description="Multiple document files"),
    split_strategy: str = Form(default="recursive", description="Splitting strategy"),
    chunk_size: int = Form(default=500, ge=100, le=5000, description="Chunk size"),
    chunk_overlap: int = Form(default=50, ge=0, le=500, description="Chunk overlap"),
    enable_refinement: bool = Form(default=True, description="Enable semantic refinement"),
):
    """
    Upload and process multiple documents in a single request.

    Processes each document independently and returns individual results.
    """
    start_time = time.time()

    # Validate file count
    max_batch_size = 10
    if len(files) > max_batch_size:
        logger.warning(f"Batch size exceeds limit: {len(files)} > {max_batch_size}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum batch size is {max_batch_size} files"
        )

    # Validate file types
    allowed_extensions = application_conf.get_properties(
        "document.allowed_extensions", [".pdf", ".docx", ".md", ".txt"]
    )
    max_file_size = application_conf.get_properties("document.max_file_size", 10485760)

    # Get processor
    processor = get_processor()

    # Process each file
    results = []
    successful = 0
    failed = 0
    errors = []

    for file in files:
        try:
            # Check file type
            file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""

            if file_ext not in allowed_extensions:
                failed += 1
                errors.append(f"{file.filename}: Unsupported file type")
                results.append({
                    "filename": file.filename,
                    "document_id": None,
                    "status": "failed",
                    "error": "Unsupported file type"
                })
                continue

            # Check file size
            file_content = await file.read()

            if len(file_content) > max_file_size:
                failed += 1
                errors.append(f"{file.filename}: File too large")
                results.append({
                    "filename": file.filename,
                    "document_id": None,
                    "status": "failed",
                    "error": "File too large"
                })
                continue

            # Process document
            logger.info(f"Processing document in batch: {file.filename}")
            result = processor.process_document(
                file_bytes=file_content,
                filename=file.filename,
                split_strategy=split_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                enable_refinement=enable_refinement,
            )

            successful += 1
            results.append({
                "filename": result.filename,
                "document_id": result.document_id,
                "status": result.status,
                "error": ", ".join(result.errors) if result.errors else None
            })

        except Exception as e:
            failed += 1
            error_msg = str(e)
            errors.append(f"{file.filename}: {error_msg}")
            results.append({
                "filename": file.filename,
                "document_id": None,
                "status": "failed",
                "error": error_msg
            })
            logger.error(f"Failed to process {file.filename}: {e}")

    processing_time = time.time() - start_time
    logger.info(
        f"Batch upload completed: {len(files)} files, "
        f"{successful} successful, {failed} failed, time: {processing_time:.2f}s"
    )

    return ApiResponse(
        code=0 if failed == 0 else 1,
        message=f"Batch upload completed: {successful} successful, {failed} failed",
        data={
            "total_files": len(files),
            "successful": successful,
            "failed": failed,
            "results": results
        }
    )


@document_router.put(
    "/{document_id}",
    response_model=ApiResponse,
    summary="Update a document",
    description="Update a document by creating a new version"
)
async def update_document(
    document_id: str,
    file: UploadFile = File(..., description="New document file"),
    split_strategy: str = Form(default="recursive", description="Splitting strategy"),
    chunk_size: int = Form(default=500, ge=100, le=5000, description="Chunk size"),
    chunk_overlap: int = Form(default=50, ge=0, le=500, description="Chunk overlap"),
    enable_refinement: bool = Form(default=True, description="Enable semantic refinement"),
):
    """
    Update a document by creating a new version.

    The old version is retained in Elasticsearch for history.
    Only the current version is kept in Milvus for vector search.
    """
    start_time = time.time()

    # Validate file type
    allowed_extensions = application_conf.get_properties(
        "document.allowed_extensions", [".pdf", ".docx", ".md", ".txt"]
    )
    file_ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""

    if file_ext not in allowed_extensions:
        logger.warning(f"Unsupported file type: {file_ext}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )

    # Validate file size
    max_file_size = application_conf.get_properties("document.max_file_size", 10485760)
    file_content = await file.read()

    if len(file_content) > max_file_size:
        logger.warning(f"File too large: {len(file_content)} bytes")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {max_file_size} bytes"
        )

    try:
        # Get processor
        processor = get_processor()

        # Check if document exists
        doc_info = processor.get_document_info(document_id)
        if doc_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}"
            )

        logger.info(f"Updating document: {document_id}")

        # Process document with new version
        result = processor.update_document(
            document_id=document_id,
            file_bytes=file_content,
            filename=file.filename,
            split_strategy=split_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            enable_refinement=enable_refinement,
        )

        processing_time = time.time() - start_time
        logger.info(
            f"Document updated successfully: {file.filename} "
            f"(version: {result.version}, time: {processing_time:.2f}s)"
        )

        # Convert chunks to response format
        chunks_response = [
            ChunkResult(
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                content=chunk.original_content,
                refined_summary=chunk.refined_summary,
                keywords=chunk.keywords,
                entities=chunk.entities,
                metadata=chunk.metadata,
            )
            for chunk in result.chunks
        ]

        update_response = DocumentUpdateResponse(
            document_id=result.document_id,
            filename=result.filename,
            version=result.version,
            previous_version=result.version - 1,
            chunk_count=result.chunk_count,
            status=result.status,
            chunks=chunks_response,
            errors=result.errors,
        )

        return ApiResponse(
            code=0,
            message="Document updated successfully",
            data=update_response.model_dump(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update document: {str(e)}"
        )


@document_router.get(
    "/{document_id}/versions",
    response_model=ApiResponse,
    summary="Get document versions",
    description="Get all versions of a document"
)
async def get_document_versions(document_id: str):
    """
    Get all versions of a document.

    Returns version history including current status of each version.
    """
    try:
        processor = get_processor()

        logger.info(f"Getting versions for document: {document_id}")

        versions_data = processor.get_document_versions(document_id)

        # Convert to response format
        versions_response = [
            DocumentVersionInfoResponse(
                document_id=v["document_id"],
                version=v["version"],
                filename=v["filename"],
                upload_time=v["upload_time"],
                chunk_count=v["chunk_count"],
                current=v["current"],
            )
            for v in versions_data
        ]

        return ApiResponse(
            code=0,
            message=f"Found {len(versions_response)} versions",
            data=DocumentVersionsResponse(
                document_id=document_id,
                total_versions=len(versions_response),
                versions=versions_response,
            ).model_dump(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document versions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get document versions: {str(e)}"
        )


@document_router.post(
    "/{document_id}/rollback/{version}",
    response_model=ApiResponse,
    summary="Rollback document version",
    description="Rollback a document to a specific version"
)
async def rollback_document(document_id: str, version: int):
    """
    Rollback a document to a specific version.

    Makes the specified version the current active version.
    """
    try:
        processor = get_processor()

        logger.info(f"Rolling back document {document_id} to version {version}")

        # Validate version parameter
        if version < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Version must be greater than 0"
            )

        # Perform rollback
        success = processor.rollback_document(document_id, version)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to rollback document"
            )

        # Get document info after rollback
        doc_info = processor.get_document_info(document_id)

        rollback_response = DocumentRollbackResponse(
            document_id=document_id,
            version=version,
            filename=doc_info.filename if doc_info else "unknown",
            chunk_count=doc_info.chunk_count if doc_info else 0,
            status="success",
        )

        logger.info(f"Document rolled back successfully to version {version}")

        return ApiResponse(
            code=0,
            message="Document rolled back successfully",
            data=rollback_response.model_dump(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rollback document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rollback document: {str(e)}"
        )


# ============================================================================
# Export router
# ============================================================================

__all__ = ["document_router"]
