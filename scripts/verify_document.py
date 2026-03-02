"""
Verify that the document was stored correctly in Milvus and Elasticsearch.
"""

from bz_agent.rag.save_embedding_to_milvus import MilvusAndEmbeddingClient
from bz_agent.rag.es_document_store import DocumentESService
from utils.logger_config import get_logger
from pymilvus import utility

logger = get_logger(__name__)


def main():
    """Main function to verify document storage."""
    DOCUMENT_ID = "2028471276590469120"

    # Initialize Milvus client
    try:
        milvus_client = MilvusAndEmbeddingClient(
            milvus_url="http://192.168.99.108:19530"
        )

        # Check if collection exists using the client directly
        has_collection = milvus_client.client.has_collection(
            collection_name="rag_chunks"
        )

        if has_collection:
            logger.info("Milvus collection 'rag_chunks' exists")

            # Load the collection before searching
            logger.info("Loading Milvus collection...")
            milvus_client.load_collection_if_needed("rag_chunks")

            # Use search to find chunks for this document (more compatible API)
            # Create a query embedding from a simple text
            from FlagEmbedding import BGEM3FlagModel
            embedding_model = BGEM3FlagModel(
                model_name_or_path="H:/large_data/modelscope_model/bge_m3",
                use_fp16=True,
            )
            query_embedding = embedding_model.encode(
                sentences=["光的反射"], return_dense=True, return_sparse=False
            )

            search_results = milvus_client.client.search(
                collection_name="rag_chunks",
                data=[query_embedding["dense_vecs"][0]],
                limit=10,
                output_fields=["document_id", "origin_content"]
            )

            # Filter for our specific document ID
            results = [r for r in search_results[0] if r.entity.get("document_id") == DOCUMENT_ID]

            logger.info(f"Milvus: Found {len(results)} chunks for document {DOCUMENT_ID}")
            for result in results:
                logger.info(f"  Chunk content: {result.entity.get('origin_content', '')[:100]}...")

            # Get collection stats
            stats = milvus_client.client.get_collection_stats("rag_chunks")
            logger.info(f"Milvus collection stats: row_count={stats.get('row_count', 0)}")
        else:
            logger.warning("Milvus collection 'rag_chunks' does not exist")

    except Exception as e:
        logger.error(f"Failed to verify Milvus data: {e}", exc_info=True)

    # Verify Elasticsearch data
    try:
        es_service = DocumentESService()

        # Get document info
        doc_info = es_service.get_document_info(DOCUMENT_ID)
        if doc_info:
            logger.info(f"Elasticsearch: Document found - {doc_info.filename}, version={doc_info.version}, chunks={doc_info.chunk_count}")
        else:
            logger.warning(f"Elasticsearch: Document {DOCUMENT_ID} not found")

        # Get document chunks
        chunks = es_service.get_document_chunks(DOCUMENT_ID)
        logger.info(f"Elasticsearch: Found {len(chunks)} chunks")
        for chunk in chunks:
            logger.info(f"  Chunk: {chunk.get('chunk_id', '')}")
            logger.info(f"    Content: {chunk.get('content', '')[:100]}...")
            logger.info(f"    Summary: {chunk.get('refined_summary', '')[:100]}...")
            logger.info(f"    Version: {chunk.get('metadata', {}).get('version', 1)}")

        # Search for a test query
        logger.info("\nTesting search functionality...")
        search_results = es_service.search_chunks(
            query="光的反射",
            document_id=DOCUMENT_ID,
            top_k=5
        )
        logger.info(f"Elasticsearch: Search found {len(search_results)} results")
        for result in search_results:
            logger.info(f"  Score: {result.get('_score', 0)}")
            logger.info(f"  Content: {result.get('content', '')[:100]}...")

    except Exception as e:
        logger.error(f"Failed to verify Elasticsearch data: {e}", exc_info=True)


if __name__ == "__main__":
    main()
