import json
from typing import List

from pymilvus import FieldSchema, DataType, CollectionSchema

from embedding_data_handler import DataEmbeddingOrm
from save_embedding_to_milvus import MilvusAndEmbeddingClient
from FlagEmbedding import BGEM3FlagModel
from utils.snow_flake import snowflake


def q_content_schema():
    # 2. 定义字段
    id_field = FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=False  # 手动指定 ID（如用雪花算法）
    )

    text_field = FieldSchema(
        name="origin_content",
        dtype=DataType.VARCHAR,
        max_length=65535  # 最大支持 65535 字节（约 2 万中文字符）
    )

    document_id_field = FieldSchema(
        name="document_id",
        dtype=DataType.VARCHAR,
        max_length=128  # 最大支持 65535 字节（约 2 万中文字符）
    )

    vector_field = FieldSchema(
        name="vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=1024  # 根据你的模型调整，BGE-M3 默认是 1024
    )

    # 3. 创建 Schema
    schema = CollectionSchema(
        fields=[id_field, document_id_field ,text_field, vector_field],
        description="Collection for RAG with text and embedding"
    )

    return schema


if __name__ == '__main__':
    init_table = ["q_content"]
    table_schema = {
        "q_content": q_content_schema(),
    }

    # 指向你本地的模型目录
    local_model_path = "H:/large_data/modelscope_model/bge_m3"  # 替换为你的实际路径

    tokenizer = BGEM3FlagModel(
        model_name_or_path=local_model_path,
        use_fp16=True  # 如果你的 GPU 支持 FP16，可加速且节省显存
    )

    # tokenizer =  AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-zh',cache_dir = "H:/large_data/jinaai_zh_embedding",trust_remote_code=True, torch_dtype=torch.bfloat16)
    embedding_dim = 1024

    def embeddings_encode(texts: List[str]):
        result = tokenizer.encode(sentences = texts, return_dense=True, return_sparse=False)
        return result["dense_vecs"]


    milvus_url = "http://192.168.99.108:19530"

    milvusAndEmbeddingClient = MilvusAndEmbeddingClient(milvus_url = milvus_url)
    # milvusAndEmbeddingClient.client.drop_collection(collection_name="q_content")
    if init_table is not None:
        temp = [milvusAndEmbeddingClient.check_and_init_table(table_name, embedding_dim, table_schema[table_name]) for table_name in init_table]

    dataEmbeddingOrm = DataEmbeddingOrm(embeddings_encode,milvusAndEmbeddingClient)


    # 初始化保存
    count = 0
    # with open("H:/large_data/QA_dataset/cmrc2018_train.json",mode="r",encoding="utf-8") as f:
    #     all_content = f.read()
    #     json_content = json.loads(all_content)
    #     data_list = json_content["data"]
    #     for per_qa in data_list:
    #         paragraphs_list = per_qa["paragraphs"][0]
    #         context = paragraphs_list["context"]
    #         document_id = paragraphs_list["id"]
    #         data = {}
    #         data["document_id"] = document_id
    #         data["origin_content"] = context
    #         data["id"] = snowflake.generate_id()
    #         result_orm = dataEmbeddingOrm.save_split_data("q_content",data)
    #         print(f"after save res is {result_orm} count is {count}")
    #         count = count + 1



    def milvus_package_data(idx:int,hit:object):
        print(f"  Rank {idx + 1}:")
        print(f"    ID: {hit.id}")
        print(f"    Distance: {hit.distance:.4f}")
        print(f"    origin_content: {hit.entity.get('origin_content')}")
        item_pk = {
            "idx": idx+1,
            "score": hit.distance,
            "document_id": hit.entity.get('document_id'),
            "origin_content": hit.entity.get('origin_content'),
        }
        return item_pk


    question = "雅典奥运会是哪一年？"
    result_dict = dataEmbeddingOrm.search_and_package(table_name = "q_content",query=question,limit=10,package_fuc=milvus_package_data)



    print("1")



