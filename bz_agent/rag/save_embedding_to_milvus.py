import time

from langchain_community.vectorstores import Milvus
from pymilvus import MilvusClient, CollectionSchema, DataType
from typing import List, Union, Dict, Optional
from utils.logger_config import logger


# https://milvus.io/api-reference/pymilvus/v2.4.x/About.md
# milvus api
class MilvusAndEmbeddingClient:


    def __init__(self,milvus_url):
        """
            milvus_url = "http://192.168.99.108:19530"
            embedding_dim 维度数量
        """
        # Authentication not enabled
        self.client = MilvusClient(milvus_url)
        self.table_dict = {}
        self.collection_need_init = ["q_content"]
        self.ready_load_collection()

    def ready_load_collection(self):
        if len(self.collection_need_init) > 0:
            for collection_name in self.collection_need_init:
                self.load_collection_if_needed(collection_name)



    def is_collection_loaded(self, collection_name: str):
        """内部函数：检查集合是否已加载"""
        stats = self.client.get_collection_stats(collection_name=collection_name)
        # 从统计信息中提取加载状态（关键字段：loaded_percent）
        # loaded_percent=100 表示完全加载，0 表示未加载
        loaded_percent = stats.get("loaded_percent", 0)
        return loaded_percent == 100



    # 3. 检查并加载集合
    def load_collection_if_needed(self,collection_name, timeout=60):
        """
        检查集合是否已加载，未加载则自动加载
        :param collection_name: 集合名称
        :param timeout: 加载超时时间（秒）
        :return: 加载后的 Collection 对象
        """
        # 检查集合是否存在
        if not self.client.has_collection(collection_name):
            raise Exception(f"集合 {collection_name} 不存在")

        check_interval: int = 2
        # 检查集合加载状态
        if not self.is_collection_loaded(collection_name):
            logger.info(f"集合 {collection_name} 未加载，开始加载...")
            # 加载集合（可指定加载的副本数，默认全部）
            self.client.load_collection(
                collection_name=collection_name,
               # replica_number=1,  # 集群版可指定多副本，单机版忽略
                timeout=timeout
            )

            # 等待加载完成（轮询检查状态）
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.is_collection_loaded(collection_name):
                    logger.info(f"集合 {collection_name} 加载完成（耗时 {int(time.time() - start_time)} 秒）")
                    return True
                time.sleep(check_interval)

            raise TimeoutError(f"集合 {collection_name} 加载超时（{timeout} 秒）")

    def check_and_create_collection(self,collection_name: str):


        # 检查 Collection 是否存在
        has_collection = self.client.has_collection(collection_name)
        logger.info(f"Collection '{collection_name}' exists: {has_collection}")

        if has_collection:
            # 检查 Collection 是否为空
            stats = self.client.get_collection_stats(collection_name)
            if stats['row_count'] == 0:
                logger.warning(f"Collection '{collection_name}' is empty, need to import data")
            else:
                logger.info(f"Collection '{collection_name}' has {stats['row_count']} documents")
        else:
            # 创建 Collection
            # schema = self.client.create_schema(
            #     field_name="vector",
            #     datatype=DataType.FLOAT_VECTOR,
            #     dim=1024
            # )
            # self.client.create_collection(collection_name, schema=schema)
            # logger.info(f"Created collection: {collection_name}")
            raise Exception(f"Collection '{collection_name}' does not exist")

    def check_collection_status(self, collection_name: str):
        stats = self.client.get_collection_stats(collection_name)
        logger.info(f"Collection '{collection_name}' stats: row_count={stats.get('row_count', 0)}")
        return stats


    def create_table(self,table_name,embedding_dim,table_with_schema:CollectionSchema):

        self.client.create_collection(
            collection_name=table_name,
            dimension=embedding_dim,
            metric_type="COSINE",
            auto_id=False,
            schema=table_with_schema,
        )

    def has_table(self,table_name):
        return self.client.has_collection(collection_name=table_name)

    def insert_row(self,table_name, data: Union[Dict, List[Dict]]):
        # 检查数据是否有问题
        if  isinstance(data, dict):
            if "vector" not in data :
                raise Exception("row not have field vector")

        res_dict = self.client.insert(collection_name=table_name,
                           data = data
                           )
        return res_dict

    def check_and_init_table(self, table_name, embedding_dim, table_with_schema:CollectionSchema):
        if not self.has_table(table_name):
            self.create_table(table_name,embedding_dim,table_with_schema)
            self.table_dict[table_name] = 1
        else:
            self.table_dict[table_name] = 1

    def upsert_row(self,table_name, data: Union[Dict, List[Dict]]):
        res_dict = self.client.upsert(collection_name=table_name,
                                      data=data
                                      )
        return res_dict

    def search_row(self,table_name:str,
                   data: Union[List[list], list],
                   filter: str = "",
                   limit: int = 10,
                   output_fields: Optional[List[str]] = None,
                   search_params: Optional[dict] = None,
                   ):
        if output_fields is None:
            output_fields = ["id","document_id","origin_content"]

        # 检查 Collection 是否存在且可用
        try:
            stats = self.check_collection_status(table_name)
            logger.info(f"Collection stats: {stats}")
            if stats.get('row_count', 0) == 0:
                logger.warning(f"Collection {table_name} is empty, returning empty results")
                return None
        except Exception as e:
            logger.error(f"Failed to check collection: {e}")
            return None
        '''
        你现在遇到的这个 Milvus 异常（code=101），核心问题是你要搜索的集合（collection）没有被加载到内存中，Milvus 只能对已加载的集合执行搜索 / 查询操作，因此导致搜索失败
        
        '''

        # data (Union[List[list], list]): The vector/vectors to search.
        res_dict = self.client.search(collection_name=table_name,
                                      data = data,
                                      filter=filter,
                                      limit = limit,
                                      output_fields = output_fields,
                                      search_params = search_params
                                      )
        return res_dict

    # query 是针对查询指定id，不太需要
    def query(self,tabel_name:str,
              filter: str,
              output_fields: Optional[List[str]] = None,
              ):
        pass

    def delete_row(self,
                   table_name: str,
                   ids: Optional[Union[list, str, int]] = None,
                   filter: Optional[str] = ""
                   ):
        res_dict = self.client.delete(collection_name=table_name,ids=ids,filter=filter)
        return res_dict




if __name__ == "__main__":
    milvus = MilvusAndEmbeddingClient("http://192.168.99.108:19530")
    logger.info("!")