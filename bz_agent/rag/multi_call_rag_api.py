import asyncio
from typing import Dict, Any, List

from app.llm import LLM
from app.rag.bm25_es_search import BM25Searcher
from app.rag.embedding_data_handler import DataEmbeddingOrm, get_milvus_dataEmbeddingOrm
from app.schema import ToolChoice, Memory, Message
from embedding_data_handler import DataEmbeddingOrm
from app.logger import logger
from pydantic import Field


class MultiCallRagApi:


    def __init__(self,
                 bm25_searcher :BM25Searcher,
                 data_embedding_searcher:DataEmbeddingOrm,):

        self.bm25_searcher = bm25_searcher
        self.data_embedding_searcher = data_embedding_searcher
        self.memory= Memory()
        self.llm =  LLM()


    def es_package_data(self,idx:int,item:Dict[str, Any]):
        item_pk = {
            "rank":idx,
            "data_source": "es_bm5",
            "score":item['score'],
            "document_id":item['source']['document_id'],
            "origin_content":item['source']['origin_content'],
        }
        return item_pk

    def milvus_package_data(self,idx:int,hit:object):
        print(f"  Rank {idx + 1}:")
        print(f"    ID: {hit.id}")
        print(f"    Distance: {hit.distance:.4f}")
        print(f"    origin_content: {hit.entity.get('origin_content')}")
        item_pk = {
            "rank": idx+1,
            "data_source":"milvus",
            "score": hit.distance,
            "document_id": hit.entity.get('document_id'),
            "origin_content": hit.entity.get('origin_content'),
        }
        return item_pk


    '''
        k 就是取前多少个条目
        该函数计算每个文档的RRF分数并按降序返回文档ID列表
        就是两边同样的文档会加权重
    '''
    def rrf(self,vec_rank_list,bm25_rank_list,k: int = 10,m: int=60):
        doc_score = {}
        for rank,doc in enumerate(vec_rank_list):
            doc_score[doc["document_id"]] = doc_score.get(doc["document_id"],0)+1 / (rank + m)

        for rank, doc in enumerate(bm25_rank_list):
            doc_score[doc["document_id"]] = doc_score.get(doc["document_id"], 0) + 1 / (rank + m)

        sorted_docs = sorted(doc_score.items(), key=lambda x: x[1], reverse=True)[:k]
        return [doc_id for doc_id, _ in sorted_docs]


    def query_match(self,query_param:Dict[str, Any]):
        # results = searcher.search_and_package(index_name = index_name,query=query,hit_fields=["origin_content"], top_k=10,package_fuc=package_data)
        query = query_param["query"]

        es_index_name = query_param["es_param"]["index_name"]
        es_hit_fields = query_param["es_param"]["hit_fields"]

        top_k = query_param.get("top_k",10)

        es_results = self.bm25_searcher.search_and_package(index_name=es_index_name, query=query, hit_fields=es_hit_fields,
                                              top_k=top_k, package_fuc=self.es_package_data)

        # vec result_dict = dataEmbeddingOrm.search_data(table_name="q_content", query=question, limit=10)
        vec_table_name = query_param["vec_param"]["table_name"]

        vec_result = self.data_embedding_searcher.search_and_package(table_name=vec_table_name,query=query,limit=top_k,package_fuc=self.milvus_package_data)

        last_res_list = []
        if len(es_results) > 0 and len(vec_result) > 0:
            doc_id_list_rrf =  self.rrf(vec_rank_list=vec_result,bm25_rank_list=es_results,k=top_k)
            es_results_dict ={ item["document_id"]:item for item in es_results}
            vec_result_dict ={ item["document_id"]:item for item in vec_result}
            if len(doc_id_list_rrf) > 0:
                for doc_id in doc_id_list_rrf:
                    if doc_id in vec_result_dict:
                        last_res_list.append(vec_result_dict[doc_id])
                    else:
                        last_res_list.append(es_results_dict[doc_id])

        elif len(es_results) > 0:
            last_res_list = es_results
        else:
            last_res_list = vec_result


        if len(last_res_list) == 0:
            logger.info(f" query {query} not match any content !!!!!")
            return None

        logger.info(last_res_list)
        return last_res_list

    async def match_db_and_ask_llm(self,query_param:Dict[str, Any]):
        match_contents = self.query_match(query_param)
        last_answer = ""
        if len(match_contents) > 0:
            only_content_list = [item["origin_content"] for item in match_contents]
            last_answer = await self.ask_llm(query_param["query"], only_content_list)
        else:
            logger.info(f">>> not have any content match query for {query_param["query"]}")
            last_answer = "NOT_MATCH_CONTENT"
        logger.info(f" query {query_param["query"]}  and answer is {last_answer}")
        return last_answer


    async def ask_llm(self,query,match_list):
        prompt = '''
        任务目标：根据检索出的文档回答用户问题
        任务要求：
                1、不得脱离检索出的文档回答问题
                2、若检索出的文档不包含用户问题的答案，请仅回答 ANSWER_NOT_IN_CONTEXT
                3、回答中只需包含答案内容，无需其他的描述
        用户问题：
        {}
        检索出的文档:
        {}
        '''
        temp_content = prompt.format(query,"\n".join(match_list))

        user_message = Message.user_message(
            temp_content
        )

        self.memory.add_message(user_message)

        response = await self.llm.ask_tool(
            messages=[user_message],
            system_msgs=None,
            tools=None,
            tool_choice=ToolChoice.AUTO,
        )

        self.memory.add_message(Message.assistant_message(content=response.content))
        return response.content





async def run():
    basic_auth = ("buz_ac", "123456")
    # 初始化搜索器
    bm25_searcher = BM25Searcher(host="http://192.168.99.108:9200", basic_auth=basic_auth)



    # 指向你本地的模型目录
    local_model_path = "H:/large_data/modelscope_model/bge_m3"
    milvus_url = "http://192.168.99.108:19530"
    dataEmbeddingOrm = get_milvus_dataEmbeddingOrm(local_tokenizer_model_path=local_model_path,milvus_url=milvus_url)


    multiCallRagApi = MultiCallRagApi(bm25_searcher =bm25_searcher,data_embedding_searcher=dataEmbeddingOrm)

    es_index_name = "cmrc2018_train"

    query_param = {
        "query":"雅典奥运会是哪一年？",
        "top_k": 10,
        "es_param":{
            "index_name":es_index_name,
            "hit_fields":["origin_content"],

        },
        "vec_param":{
            "table_name":"q_content",

        }
    }


    last_answer = await multiCallRagApi.match_db_and_ask_llm(query_param)
    logger.info("demo end ")


if __name__ == '__main__':
    asyncio.run(run())


