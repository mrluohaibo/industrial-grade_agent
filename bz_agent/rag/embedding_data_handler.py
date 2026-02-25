from typing import Union, Dict, List, Callable, Optional
from bz_agent.rag.save_embedding_to_milvus import MilvusAndEmbeddingClient
from FlagEmbedding import BGEM3FlagModel

# 获得向量数据库句柄
def get_milvus_dataEmbeddingOrm(local_tokenizer_model_path:str,milvus_url:str):

    # 指向你本地的模型目录
    # local_tokenizer_model_path = "H:/large_data/modelscope_model/bge_m3"  # 替换为你的实际路径
    tokenizer = BGEM3FlagModel(
        model_name_or_path=local_tokenizer_model_path,
        use_fp16=True  # 如果你的 GPU 支持 FP16，可加速且节省显存
    )

    # tokenizer =  AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh',cache_dir = "H:/large_data/jinaai_zh_embedding",trust_remote_code=True, torch_dtype=torch.bfloat16)

    def embeddings_encode(texts: List[str]):
        result = tokenizer.encode(sentences=texts, return_dense=True, return_sparse=False)
        return result["dense_vecs"]


    # milvus_url = "http://192.168.99.108:19530"
    milvusAndEmbeddingClient = MilvusAndEmbeddingClient(milvus_url=milvus_url)
    dataEmbeddingOrm = DataEmbeddingOrm(embeddings_encode, milvusAndEmbeddingClient)
    return dataEmbeddingOrm


class DataEmbeddingOrm:

    def __init__(self,embeddings_encode,vector_db):
        self.embeddings_encode = embeddings_encode
        self.vector_db = vector_db

    def save_split_data(self,table_name:str,data: Union[Dict, List[Dict]]):
        # 将所有的内容变成向量 {content,document_id}
        save_data = None
        if isinstance(data, dict):

            save_data = self.parse_vec_json_item(data)
        elif isinstance(data,list):
            save_data = [self.parse_vec_json_item(item) for item in data]


        if save_data is not None:
            result_orm = self.vector_db.upsert_row(table_name,save_data)
            print(result_orm)
            return result_orm
        return None

    def search_data(self,table_name:str,query:str,limit:int):
        query_vector = self.embeddings_encode([query])[0].tolist()
        column_filter = ["document_id","origin_content","vector"]
        result_dict = self.vector_db.search_row(table_name,data=[query_vector],output_fields=column_filter,limit = limit)
        return result_dict

    def search_and_package(self,table_name:str,query:str,limit:int,package_fuc:Optional[Callable] ):
        result = self.search_data(table_name,query,limit)
        if package_fuc is None:
            raise ValueError("package_fuc can not be null!")
        result_list = []
        if result is not None:
            for i, hits in enumerate(result):  # 每个 hits 对应一个 query vector
                print(f"--- Query {i} Top-{len(hits)} results ---")
                for j, hit in enumerate(hits):
                    item = package_fuc(j,hit)
                    result_list.append(item)
        return result_list

    def parse_vec_json_item(self,data:Dict):
        save_data = {}

        document_id = data["document_id"]
        content = data["origin_content"]

        save_data["id"] = data["id"]
        save_data["document_id"] = document_id
        save_data["origin_content"] = content
        save_data["vector"] = self.embeddings_encode([content])[0].tolist()
        return save_data

