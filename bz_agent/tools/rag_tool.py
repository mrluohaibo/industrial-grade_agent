"""
RAG检索工具

封装MultiCallRagApi为LangChain工具
"""
import asyncio
from typing import Annotated, Dict, Any, List, Optional

from langchain_core.tools import tool

from bz_agent.rag.bm25_es_search import BM25Searcher
from bz_agent.rag.embedding_data_handler import DataEmbeddingOrm, get_milvus_dataEmbeddingOrm
from bz_agent.rag.multi_call_rag_api import MultiCallRagApi
from utils.logger_config import logger


# 全局RAG实例缓存
_rag_instance: Optional[MultiCallRagApi] = None


def get_rag_instance() -> MultiCallRagApi:
    """
    获取或初始化RAG实例（单例模式）

    Returns:
        MultiCallRagApi: RAG实例
    """
    global _rag_instance

    if _rag_instance is None:
        # 从配置读取RAG相关参数（暂时使用硬编码值，后续可迁移到配置文件）
        es_host = "http://192.168.99.108:9200"
        es_basic_auth = ("buz_ac", "123456")
        local_model_path = "H:/large_data/modelscope_model/bge_m3"
        milvus_url = "http://192.168.99.108:19530"

        logger.info("Initializing RAG components...")

        try:
            # 初始化BM25搜索器
            bm25_searcher = BM25Searcher(host=es_host, basic_auth=es_basic_auth)

            # 初始化向量搜索器
            data_embedding_searcher = get_milvus_dataEmbeddingOrm(
                local_tokenizer_model_path=local_model_path,
                milvus_url=milvus_url
            )

            # 创建RAG实例
            _rag_instance = MultiCallRagApi(
                bm25_searcher=bm25_searcher,
                data_embedding_searcher=data_embedding_searcher
            )

            logger.info("RAG components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            raise

    return _rag_instance


def es_package_data(idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
    """包装ES搜索结果"""
    return {
        "rank": idx,
        "data_source": "es_bm5",
        "score": item['score'],
        "document_id": item['source']['document_id'],
        "origin_content": item['source']['origin_content'],
    }


def milvus_package_data(idx: int, hit: Any) -> Dict[str, Any]:
    """包装Milvus搜索结果"""
    return {
        "rank": idx + 1,
        "data_source": "milvus",
        "score": hit.distance,
        "document_id": hit.entity.get('document_id'),
        "origin_content": hit.entity.get('origin_content'),
    }


@tool
def rag_knowledge_retrieval(
    query: Annotated[str, "The search query for knowledge retrieval"],
    top_k: Annotated[int, "Number of top results to retrieve (default: 10)"] = 10,
    es_index_name: Annotated[str, "Elasticsearch index name (default: cmrc2018_train)"] = "cmrc2018_train",
    vec_table_name: Annotated[str, "Milvus vector table name (default: q_content)"] = "q_content"
) -> str:
    """
    Retrieve relevant knowledge using RAG (Retrieval-Augmented Generation).

    This tool combines BM25 keyword search and vector semantic search to find
    the most relevant documents from the knowledge base and uses LLM to generate an answer.

    Args:
        query: The search query for knowledge retrieval.
        top_k: Number of top results to retrieve (default: 10).
        es_index_name: Elasticsearch index name (default: cmrc2018_train).
        vec_table_name: Milvus vector table name (default: q_content).

    Returns:
        str: The retrieved knowledge content or LLM-generated answer.
    """
    try:
        # 获取RAG实例
        rag_instance = get_rag_instance()

        # 构建查询参数
        query_param = {
            "query": query,
            "top_k": top_k,
            "es_param": {
                "index_name": es_index_name,
                "hit_fields": ["origin_content"]
            },
            "vec_param": {
                "table_name": vec_table_name
            }
        }

        # 执行RAG查询（带LLM回答）
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                rag_instance.match_db_and_ask_llm(query_param)
            )
            if result == "NOT_MATCH_CONTENT":
                return "No matching content found in the knowledge base for the given query."
            return result or "No content retrieved from the knowledge base."
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return f"Error during knowledge retrieval: {str(e)}"


@tool
async def rag_knowledge_retrieval_async(
    query: Annotated[str, "The search query for knowledge retrieval"],
    top_k: Annotated[int, "Number of top results to retrieve"] = 10,
    es_index_name: Annotated[str, "Elasticsearch index name"] = "cmrc2018_train",
    vec_table_name: Annotated[str, "Milvus vector table name"] = "q_content"
) -> str:
    """
    Retrieve relevant knowledge using RAG (async version).

    This is the async version of rag_knowledge_retrieval, preferred for use in async contexts.

    Args:
        query: The search query for knowledge retrieval.
        top_k: Number of top results to retrieve (default: 10).
        es_index_name: Elasticsearch index name.
        vec_table_name: Milvus vector table name.

    Returns:
        str: The retrieved knowledge content or LLM-generated answer.
    """
    try:
        # 获取RAG实例
        rag_instance = get_rag_instance()

        # 构建查询参数
        query_param = {
            "query": query,
            "top_k": top_k,
            "es_param": {
                "index_name": es_index_name,
                "hit_fields": ["origin_content"]
            },
            "vec_param": {
                "table_name": vec_table_name
            }
        }

        # 执行RAG查询（带LLM回答）
        result = await rag_instance.match_db_and_ask_llm(query_param)

        if result == "NOT_MATCH_CONTENT":
            return "No matching content found in the knowledge base for the given query."

        return result or "No content retrieved from the knowledge base."

    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return f"Error during knowledge retrieval: {str(e)}"


@tool
def rag_search_only(
    query: Annotated[str, "The search query for knowledge retrieval"],
    top_k: Annotated[int, "Number of top results to retrieve (default: 10)"] = 10,
    es_index_name: Annotated[str, "Elasticsearch index name (default: cmrc2018_train)"] = "cmrc2018_train",
    vec_table_name: Annotated[str, "Milvus vector table name (default: q_content)"] = "q_content"
) -> str:
    """
    Search the knowledge base using RAG without LLM answer generation.

    This tool performs BM25 and vector search to retrieve relevant documents
    from the knowledge base, but returns the raw documents without LLM processing.

    Use this when you want to see the raw retrieved documents for analysis
    or when you want to process them yourself.

    Args:
        query: The search query for knowledge retrieval.
        top_k: Number of top results to retrieve (default: 10).
        es_index_name: Elasticsearch index name (default: cmrc2018_train).
        vec_table_name: Milvus vector table name (default: q_content).

    Returns:
        str: The retrieved documents in a formatted string.
    """
    try:
        # 获取RAG实例
        rag_instance = get_rag_instance()

        # 构建查询参数
        query_param = {
            "query": query,
            "top_k": top_k,
            "es_param": {
                "index_name": es_index_name,
                "hit_fields": ["origin_content"]
            },
            "vec_param": {
                "table_name": vec_table_name
            }
        }

        # 执行RAG查询（不带LLM回答）
        results = rag_instance.query_match(query_param)

        if not results or len(results) == 0:
            return "No matching content found in the knowledge base."

        # 格式化结果
        output_parts = [f"Found {len(results)} matching documents:\n"]
        for i, item in enumerate(results, 1):
            output_parts.append(
                f"[{i}] Source: {item['data_source']}, Score: {item['score']:.4f}\n"
                f"    Document ID: {item['document_id']}\n"
                f"    Content: {item['origin_content'][:300]}...\n"
            )

        return "\n".join(output_parts)

    except Exception as e:
        logger.error(f"RAG search error: {e}")
        return f"Error during knowledge search: {str(e)}"
