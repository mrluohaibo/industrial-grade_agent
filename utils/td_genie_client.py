from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

from taos import TaosConnection

from utils.logger_config import  logger



class TDEngineClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6030,
        user: str = "root",
        password: str = "taosdata",
        database: Optional[str] = None,
        timeout: int = 30,
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.timeout = timeout
        self._conn = None
        self.connect()

    def connect(self):
        """建立 TDengine 原生连接"""
        try:
            self._conn = TaosConnection(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                timeout=self.timeout,
            )
            logger.info(f"✅ Connected to TDengine at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect: {e}")
            raise

    def close(self):
        """关闭连接"""
        if self._conn:
            self._conn.close()
            logger.info("🔌 Connection closed")

    def execute(self, sql: str) -> int:
        """执行非查询语句（CREATE, INSERT, ALTER 等）"""
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql)
            affected = cursor.rowcount
            logger.debug(f"✅ Executed: {sql} | Affected rows: {affected}")
            return affected
        finally:
            cursor.close()

    def query(self, sql: str) -> List[Dict[str, Any]]:
        """执行查询，返回字典列表"""
        cursor = self._conn.cursor()
        try:
            cursor.execute(sql)
            columns = [col[0] for col in cursor.description]
            rows = cursor.fetchall()
            result = [dict(zip(columns, row)) for row in rows]
            logger.debug(f"✅ Query returned {len(result)} rows")
            return result
        finally:
            cursor.close()

    def insert_many(self, sql: str, data: List[Tuple]) -> int:
        """
        批量插入（参数化，防 SQL 注入）
        示例: insert_many("INSERT INTO t VALUES (?, ?)", [(ts1, val1), (ts2, val2)])
        """
        cursor = self._conn.cursor()
        try:
            cursor.executemany(sql, data)
            affected = cursor.rowcount
            logger.debug(f"✅ Batch inserted {affected} rows")
            return affected
        finally:
            cursor.close()



    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



