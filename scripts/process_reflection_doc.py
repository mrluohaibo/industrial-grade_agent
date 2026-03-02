"""
Process 光的反射.docx document and store in vector database.

This script:
1. Reads the Word document
2. Splits it into chunks
3. Stores chunks in Milvus and Elasticsearch
"""

from pathlib import Path

from bz_agent.rag.document_processor import DocumentProcessor
from utils.logger_config import get_logger

logger = get_logger(__name__)


def main():
    """Main function to process the document."""
    # Document path
    doc_path = Path(__file__).parent.parent / "docs" / "temp" / "光的反射.docx"

    if not doc_path.exists():
        logger.error(f"Document not found: {doc_path}")
        return

    logger.info(f"Processing document: {doc_path}")

    # Read file bytes
    with open(doc_path, "rb") as f:
        file_bytes = f.read()

    # Initialize document processor
    try:
        processor = DocumentProcessor()

        # Process document
        result = processor.process_document(
            file_bytes=file_bytes,
            filename="光的反射.docx",
            split_strategy="recursive",
            chunk_size=500,
            chunk_overlap=50,
            enable_refinement=True,
        )

        logger.info(f"Document processed successfully:")
        logger.info(f"  Document ID: {result.document_id}")
        logger.info(f"  Filename: {result.filename}")
        logger.info(f"  Version: {result.version}")
        logger.info(f"  Status: {result.status}")
        logger.info(f"  Chunk count: {result.chunk_count}")

        if result.errors:
            logger.warning(f"  Errors: {result.errors}")

        if result.status == "success":
            logger.info("Document stored in Milvus and Elasticsearch successfully")
        else:
            logger.error(f"Document processing failed with status: {result.status}")

    except Exception as e:
        logger.error(f"Failed to process document: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
