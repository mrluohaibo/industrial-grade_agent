from pymilvus import MilvusClient, CollectionSchema
from typing import List, Union, Dict, Optional

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
