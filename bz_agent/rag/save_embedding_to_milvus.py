import time

from langchain_community.vectorstores import Milvus
from pymilvus import MilvusClient, CollectionSchema, DataType, MilvusException
from typing import List, Union, Dict, Optional

from bz_core.thread_pool_define import handle_daily_stock_data_pool
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
        """
            内部函数：检查集合是否已加载
            beload,need_manual_load return
        """
        stats = self.client.get_load_state(collection_name=collection_name)
        # 从统计信息中提取加载状态（关键字段：loaded_percent）
        # loaded_percent=100 表示完全加载，0 表示未加载
        # NotExist = 0
        # NotLoad = 1
        # Loading = 2
        # Loaded = 3
        state = stats.get("state")
        from pymilvus.client.types import LoadState
        if state == LoadState.Loaded:
            return True,False
        elif state == LoadState.Loading:
            progress = stats.get("progress", 0)
            logger.info(f"collection {collection_name} load progress: {progress}")
            return False,False
        elif state == LoadState.NotLoad:
            return False,True
        elif state == LoadState.NotExist:
            raise Exception(f"集合 {collection_name} 不存在")

    def async_load_collection( self,collection_name: str):
        """
        异步执行加载集合的函数（交给线程池执行）
        """
        try:
            # 执行加载操作（该方法本身是阻塞的，交给线程池后主线程不阻塞）
            self.client.load_collection(collection_name=collection_name,timeout=60)
            return True, "加载成功"
        except MilvusException as e:
            return False, f"加载失败: {str(e)}"


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

        stats = self.client.get_collection_stats(collection_name)
        if stats['row_count'] == 0:
            logger.warning(f"Collection '{collection_name}' is empty, need to import data")
            return

        check_interval: int = 2
        # 检查集合加载状态
        beload,need_manual_load = self.is_collection_loaded(collection_name)

        if not beload:
            logger.info(f"集合 {collection_name} 未加载，开始加载...")
            if need_manual_load:
                handle_daily_stock_data_pool.add_task(self.async_load_collection,collection_name)
            # 等待加载完成（轮询检查状态）
            start_time = time.time()
            while True:
                beload, need_manual_load = self.is_collection_loaded(collection_name)
                if beload:
                    logger.info(f"集合 {collection_name} 加载完成（耗时 {int(time.time() - start_time)} 秒）")
                    return True
                time.sleep(check_interval)
        else:
            logger.info(f"集合 {collection_name} 已经加载")

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

    def emptyCollection(self, collection_name):
        try:
            # 3. 检查 collection 是否存在
            if self.client.has_collection(collection_name=collection_name):
                # 4. 删除 collection 中的所有数据（保留结构）
                # 使用 delete 方法，expr 设置为 "id >= 0" 匹配所有数据
                self.client.delete(
                    collection_name=collection_name,
                    filter="id >= 0"
                )
                logger.info(f"✅ 成功清空 collection: {collection_name}")

                # 可选：验证清空结果（查询数据量）
                stats = self.client.get_collection_stats(collection_name=collection_name)
                logger.info(f"📊 清空后数据量: {stats['row_count']}")
            else:
                logger.info(f"❌ collection {collection_name} 不存在")

        except Exception as e:
            logger.info(f"❌ 清空 collection 失败: {str(e)}")


if __name__ == "__main__":
    milvus = MilvusAndEmbeddingClient("http://192.168.99.108:19530")
    logger.info("!")