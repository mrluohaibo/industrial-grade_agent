# mongo_client.py
from typing import Optional, Dict, Any, List, Union
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging

from utils.StringUtil import StringUtil


class MongoManager:
    """
    MongoDB 客户端管理器（适配 pymongo 4.x+）
    """

    def __init__(
            self,
            uri: str ,
            host: Optional[str] = None,
            port: int = 27017,
            username: Optional[str] = None,
            password: Optional[str] = None,
            auth_source: str = "admin",
            database_name: str = "test",
            server_selection_timeout_ms: int = 5000,
            **kwargs
    ):
        """
        初始化 MongoDB 连接

        :param uri: MongoDB 连接字符串（优先使用）  "mongodb://localhost:27017/"
        :param host: 主机地址（若未提供 uri）
        :param port: 端口
        :param username: 用户名
        :param password: 密码
        :param auth_source: 认证数据库
        :param database_name: 默认数据库名
        :param server_selection_timeout_ms: 连接超时（毫秒）
        """
        self.database_name = database_name
        self._client: Optional[MongoClient] = None

        if uri:
            self._client = MongoClient(
                uri,
                serverSelectionTimeoutMS=server_selection_timeout_ms,
                **kwargs
            )
        else:
            # 构建连接参数
            connect_kwargs = {
                "host": host or "localhost",
                "port": port,
                "serverSelectionTimeoutMS": server_selection_timeout_ms,
                **kwargs
            }
            if username and password:
                connect_kwargs.update({
                    "username": username,
                    "password": password,
                    "authSource": auth_source
                })
            self._client = MongoClient(**connect_kwargs)

        # 验证连接
        self._ping()

    def _ping(self) -> None:
        """测试数据库连接"""
        try:
            self._client.admin.command('ping')
            logging.info("✅ 成功连接到 MongoDB")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logging.error(f"❌ 无法连接到 MongoDB: {e}")
            raise

    @property
    def client(self) -> MongoClient:
        """获取原始 MongoClient 实例"""
        return self._client

    @property
    def db(self) -> Database:
        """获取默认数据库"""
        return self._client[self.database_name]

    def get_collection(self, name: str) -> Collection:
        """获取指定集合"""
        return self.db[name]

    # --- 常用操作封装 ---
    def insert_one(self, collection: str, document: Dict[str, Any]) -> str:
        """插入单条文档"""
        result = self.get_collection(collection).insert_one(document)
        return str(result.inserted_id)

    def insert_many(self, collection: str, documents: List[Dict[str, Any]]) -> List[str]:
        """批量插入文档"""
        result = self.get_collection(collection).insert_many(documents)
        return [str(oid) for oid in result.inserted_ids]

    def find_one(self, collection: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """查找单条文档"""
        doc = self.get_collection(collection).find_one(query)
        if doc and "_id" in doc:
            doc["_id"] = str(doc["_id"])  # 转为字符串便于 JSON 序列化
        return doc

    def find(
            self,
            collection: str,
            query: Optional[Dict[str, Any]] = None,
            projection: Optional[Dict[str, Any]] = None,
            limit: int = 0
    ) -> List[Dict[str, Any]]:
        """查找多条文档"""
        cursor = self.get_collection(collection).find(
            filter=query or {},
            projection=projection
        )
        if limit > 0:
            cursor = cursor.limit(limit)

        results = []
        for doc in cursor:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])
            results.append(doc)
        return results

    def get_paginated_data(self,
                           collection: str,
                           query: Optional[Dict[str, Any]] ,
                           page: int,
                           page_size: int):
        """
        分页查询数据
        :param page: 页码（从1开始）
        :param page_size: 每页数量
        :return: 当前页的数据列表
        """
        skip = (page - 1) * page_size
        cursor = self.get_collection(collection).find(filter=query or {},).skip(skip).limit(page_size)
        return list(cursor)

    def get_cursor_paginated_data(self,
                               collection: str,
                               query: Optional[Dict[str, Any]] ,
                               last_id: str,
                               page_size: int = 10):
        """
        使用 _id 游标分页（假设 _id 是 ObjectId）
        :param last_id: 上一页最后一个文档的 _id（字符串形式）
        :param page_size: 每页数量
        :return: 当前页数据
        """
        from bson import ObjectId

        if last_id:
            query['_id'] = {'$gt': ObjectId(last_id)}

        cursor = self.get_collection(collection).find(filter=query or {},).sort('_id', 1).limit(page_size)
        return list(cursor)

    def update_one(
            self,
            collection: str,
            query: Dict[str, Any],
            update: Dict[str, Any],
            upsert: bool = False
    ) -> int:
        """更新单条文档，返回修改数量"""
        result = self.get_collection(collection).update_one(
            query, update, upsert=upsert
        )
        return result.modified_count

    def delete_one(self, collection: str, query: Dict[str, Any]) -> int:
        """删除单条文档，返回删除数量"""
        result = self.get_collection(collection).delete_one(query)
        return result.deleted_count

    def close(self):
        """关闭连接"""
        if self._client:
            self._client.close()
            logging.info("🔌 MongoDB 连接已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()




if __name__ == "__main__":
    # 方式 1：使用 URI
    # mongo = MongoManager(uri="mongodb://user:pass@localhost:27017/mydb?authSource=admin")

    # 方式 2：分项配置
    mongo = MongoManager(
        host="192.168.99.108",
        port=27017,
        database_name="stock_db"
    )

    # 插入数据
    doc_id = mongo.insert_one("stocks", {
        "code": "600000",
        "name": "浦发银行",
        "price": 9.85
    })
    print(f"Inserted ID: {doc_id}")

    # 查询数据
    stock = mongo.find_one("stocks", {"code": "600000"})
    print(stock)

    # 使用 with 自动关闭
    with MongoManager(database_name="test") as db:
        docs = db.find("logs", limit=5)
        print(docs)