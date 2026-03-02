"""
Query chunks from Milvus and Elasticsearch to show original content.

Document ID: 2028471276590469120
"""

import json
from pathlib import Path

from bz_agent.rag.save_embedding_to_milvus import MilvusAndEmbeddingClient
from bz_agent.rag.es_document_store import DocumentESService
from utils.logger_config import get_logger

logger = get_logger(__name__)

DOCUMENT_ID = "2028471276590469120"
OUTPUT_FILE = Path(__file__).parent.parent / "docs" / "temp" / "chunks_output.txt"


def main():
    """Main function to query chunks."""
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"Query document ID: {DOCUMENT_ID} - Chunk original content")
    output_lines.append("=" * 80)

    # Query from Elasticsearch (has original content)
    output_lines.append("\n[Elasticsearch Query Results]\n")
    es_service = DocumentESService()

    doc_info = es_service.get_document_info(DOCUMENT_ID)
    if doc_info:
        output_lines.append(f"Filename: {doc_info.filename}")
        output_lines.append(f"Version: {doc_info.version}")
        output_lines.append(f"Chunk count: {doc_info.chunk_count}")
    else:
        output_lines.append("Document not found")

    chunks = es_service.get_document_chunks(DOCUMENT_ID)
    output_lines.append(f"\nFound {len(chunks)} chunks:\n")

    for i, chunk in enumerate(chunks, 1):
        content = chunk.get("content", "")
        refined_summary = chunk.get("refined_summary", "")
        keywords = chunk.get("keywords", [])
        entities = chunk.get("entities", [])
        metadata = chunk.get("metadata", {})

        output_lines.append(f"\n--- Chunk {i} ---")
        output_lines.append(f"Chunk ID: {chunk.get('chunk_id', '')}")
        output_lines.append(f"Index: {chunk.get('chunk_index', '')}")
        output_lines.append(f"Original content:")
        output_lines.append(f"  {content}")
        output_lines.append(f"  (Length: {len(content)} chars)")
        output_lines.append(f"Refined summary: {refined_summary[:150]}...")
        if keywords:
            output_lines.append(f"Keywords: {', '.join(keywords[:10])}")
        if entities:
            output_lines.append(f"Entities: {', '.join(entities[:5])}")
        output_lines.append(f"Version: {metadata.get('version', 1)}")

    # Query from Milvus (verify vector storage)
    output_lines.append("\n" + "=" * 80)
    output_lines.append("\n[Milvus Vector Storage Verification]\n")

    try:
        milvus_client = MilvusAndEmbeddingClient(
            milvus_url="http://192.168.99.108:19530"
        )

        # Load collection
        milvus_client.load_collection_if_needed("rag_chunks")

        # Search all chunks for this document
        from FlagEmbedding import BGEM3FlagModel
        embedding_model = BGEM3FlagModel(
            model_name_or_path="H:/large_data/modelscope_model/bge_m3",
            use_fp16=True,
        )
        # Use a broad query to get all chunks
        query_embedding = embedding_model.encode(
            sentences=[""], return_dense=True, return_sparse=False
        )

        search_results = milvus_client.client.search(
            collection_name="rag_chunks",
            data=[query_embedding["dense_vecs"][0]],
            limit=20,
            output_fields=["document_id", "origin_content"]
        )

        # Filter for our document
        results = [r for r in search_results[0] if r.entity.get("document_id") == DOCUMENT_ID]
        output_lines.append(f"Milvus found {len(results)} vectors\n")

        for i, result in enumerate(results, 1):
            content = result.entity.get("origin_content", "")
            output_lines.append(f"\n--- Vector {i} ---")
            output_lines.append(f"Original content:")
            output_lines.append(f"  {content[:200]}..." if len(content) > 200 else f"  {content}")
            if len(content) > 200:
                output_lines.append(f"  (Showing first 200 of {len(content)} chars)")

    except Exception as e:
        logger.error(f"Failed to query Milvus: {e}", exc_info=True)
        output_lines.append(f"\nQuery Milvus failed: {e}")

    output_lines.append("\n" + "=" * 80)

    # Write to file
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Total chunks found: {len(chunks)}")


if __name__ == "__main__":
    main()
