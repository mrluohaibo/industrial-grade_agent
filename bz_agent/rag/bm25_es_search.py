import json
from typing import List, Dict, Any, Optional, Union, Tuple, Callable
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from utils.logger_config import logger


class BM25Searcher:
    def __init__(self, host: str = "http://localhost:9200",basic_auth : Optional[Union[str, Tuple[str, str]]] = None):
        self.es = Elasticsearch(hosts=[host],basic_auth = basic_auth,verify_certs=False )
        if not self.es.ping():
            raise ValueError("Cannot connect to Elasticsearch!")

    def delete_index(self, index_name: str):
        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
            logger.info(f"index {index_name} delete")

    def create_index(self,index_name: str, force_recreate: bool = False ,mappings:Dict[str, Any] = None):
        """创建索引（使用标准 analyzer，BM25 是 ES 默认相似度模型）"""
        if self.es.indices.exists(index=index_name):
            if force_recreate:
                self.es.indices.delete(index=index_name)
            else:
                logger.info(f"Index '{index_name}' already exists.")
                return


        # 定义 mapping（可根据需要调整）
        if mappings is None:
            raise ValueError("mappings can not be null!")

        self.es.indices.create(index=index_name, body={"mappings": mappings})
        logger.info(f"Index '{index_name}' created.")

    def add_documents(self,index_name:str, docs: List[Dict[str, Any]]):
        """批量添加文档"""
        actions = [
            {
                "_index": index_name,
                "_source": doc
            }
            for doc in docs
        ]
        bulk(self.es, actions)
        self.es.indices.refresh(index=index_name)  # 立即刷新使文档可搜
        logger.info(f"Added {len(docs)} documents.")

    def search(self,index_name:str, query: str,hit_fields:List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """执行 BM25 搜索（Elasticsearch 默认就是 BM25！）"""
        body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": hit_fields,
                    "type": "best_fields"  # 也可用 most_fields, cross_fields 等
                }
            },
            "size": top_k
        }

        res = self.es.search(index=index_name, body=body)
        hits = []
        for hit in res["hits"]["hits"]:
            hits.append({
                "id": hit["_id"],
                "score": hit["_score"],
                "source": hit["_source"]
            })
        return hits

    def search_and_package(self,index_name:str, query: str,hit_fields:List[str], top_k: int = 10,package_fuc:Optional[Callable] = None):
        results = self.search(index_name = index_name,query=query, hit_fields= hit_fields,top_k = top_k)
        if package_fuc is None:
            raise ValueError("package_fuc can not be null!")

        print(f"\n🔍 Query: '{query}'\n")
        result_list = []
        if len(results) > 0:
            for i, res in enumerate(results, 1):
                item = package_fuc(i,res)
                result_list.append(item)

        return result_list


# ========================
# 示例使用
# ========================
if __name__ == "__main__":
    # 示例文档（中英文混合）
    # documents = [
    #     {"title": "iPhone 存储空间不足", "content": "苹果手机提示内存不足，无法开机，建议清理照片和应用缓存。"},
    #     {"title": "Android 手机卡顿", "content": "安卓设备运行缓慢时，可重启或卸载不常用 App。"},
    #     {"title": "How to free up iPhone storage", "content": "Delete unused apps, offload photos to iCloud, and clear Safari cache."},
    #     {"title": "Milvus vector search tutorial", "content": "Use Milvus for dense retrieval with cosine similarity."},
    #     {"title": "Elasticsearch BM25 example", "content": "Elasticsearch uses BM25 as default similarity algorithm for text search."}
    # ]
    basic_auth = ("buz_ac","123456")
    index_name =  "cmrc2018_train"
    # 初始化搜索器
    searcher = BM25Searcher(host = "http://192.168.99.108:9200",basic_auth = basic_auth)

    # 创建索引（如果不存在）
    # 如果需要删除之前的索引记得调用如下命令
    # searcher.delete_index(index_name=searcher.index_name)
    # 定义 mapping（可根据需要调整）
    mappings = {
        "properties": {
            "origin_content": {
                "type": "text",
                "analyzer": "ik_max_word"  # 可替换为 ik_smart / ik_max_word（中文需安装 IK 插件）
            },
            "document_id": {
                "type": "text"
            }
        }
    }


    searcher.create_index(index_name = index_name,force_recreate=False,mappings = mappings)

    # with open("H:/large_data/QA_dataset/cmrc2018_train.json",mode="r",encoding="utf-8") as f:
    #     all_content = f.read()
    #     json_content = json.loads(all_content)
    #     data_list = json_content["data"]
    #     temp_list = []
    #     for per_qa in data_list:
    #         paragraphs_list = per_qa["paragraphs"][0]
    #         context = paragraphs_list["context"]
    #         document_id = paragraphs_list["id"]
    #         temp_list.append({
    #                 "origin_content":context,
    #                 "document_id":document_id
    #             })
    #         if len(temp_list) > 0  and len(temp_list) % 5  == 0:
    #             # 添加文档
    #             searcher.add_documents(index_name = index_name,temp_list)
    #             temp_list.clear()
    #     if len(temp_list) > 0:
    #         searcher.add_documents(index_name = index_name,temp_list)

    # 执行搜索
    query = "雅典奥运会是哪一年？"
    def package_data(idx:int,item:Dict[str, Any]):
        item_pk = {
            "idx":idx,
            "score":item['score'],
            "document_id":item['source']['document_id'],
            "origin_content":item['source']['origin_content'],
        }
        return item_pk


    results = searcher.search_and_package(index_name = index_name,query=query,hit_fields=["origin_content"], top_k=10,package_fuc=package_data)

    print(results)