# transactional_mysql_client.py

import MySQLdb
from dbutils.pooled_db  import PooledDB
import logging
from typing import Optional, List, Dict, Any, Union

logger = logging.getLogger(__name__)


class TransactionalMySQLClient:
    def __init__(
            self,
            host: str,
            port: int = 3306,
            user: str = 'root',
            password: str = '',
            database: str = '',
            charset: str = 'utf8mb4',
            mincached: int = 2,
            maxcached: int = 10,
            maxconnections: int = 20,
            blocking: bool = True,
            autocommit_default: bool = False,  # 控制 execute 是否自动提交
            **kwargs
    ):
        """
        初始化带事务支持的 MySQL 客户端

        :param autocommit_default:
            - True: execute() 执行后自动 commit（类似普通模式）
            - False: execute() 在内部开启事务，成功则 commit，失败则 rollback（推荐生产使用）
        """
        self.pool = PooledDB(
            creator=MySQLdb,
            host=host,
            port=port,
            user=user,
            passwd=password,
            db=database,
            charset=charset,
            mincached=mincached,
            maxcached=maxcached,
            maxconnections=maxconnections,
            blocking=blocking,
            autocommit=False,  # 底层连接始终关闭 autocommit，由我们控制事务
            **kwargs
        )
        self.autocommit_default = autocommit_default

    def _get_conn(self):
        """获取新连接（不带自动重连，供内部使用）"""
        return self.pool.connection()

    def _safe_execute(self, conn, sql: str, args: Optional[tuple] = None, fetch: bool = False, as_dict: bool = True):
        """在给定连接上安全执行 SQL"""
        cursor_class = MySQLdb.cursors.DictCursor if as_dict else MySQLdb.cursors.Cursor
        cursor = conn.cursor(cursor_class)
        try:
            cursor.execute(sql, args or ())
            if fetch:
                result = cursor.fetchall()
                return list(result) if result else []
            else:
                return cursor.rowcount
        finally:
            cursor.close()

    def execute(self, sql: str, args: Optional[tuple] = None) -> int:
        """
        执行 INSERT/UPDATE/DELETE，默认开启事务：
        - 成功 → commit
        - 异常 → rollback 并抛出
        """
        conn = self._get_conn()
        try:
            # 显式开始事务（虽然 autocommit=False 已隐式开启）
            # 执行语句
            rowcount = self._safe_execute(conn, sql, args, fetch=False)

            if self.autocommit_default:
                conn.commit()
            else:
                # 默认行为：单条 execute 视为一个完整事务
                conn.commit()
            return rowcount
        except Exception as e:
            conn.rollback()
            logger.error(f"Execute failed and rolled back. SQL: {sql}, Args: {args}, Error: {e}")
            raise
        finally:
            conn.close()

    def executemany(self, sql: str, args_list: List[tuple]) -> int:
        """批量执行，默认开启事务"""
        conn = self._get_conn()
        cursor = None
        try:
            cursor = conn.cursor()
            rowcount = cursor.executemany(sql, args_list)
            if self.autocommit_default:
                conn.commit()
            else:
                conn.commit()
            return rowcount
        except Exception as e:
            conn.rollback()
            logger.error(f"Executemany failed and rolled back. SQL: {sql}, Error: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def query(self, sql: str, args: Optional[tuple] = None, as_dict: bool = True) -> List[Dict[str, Any]]:
        """执行 SELECT 查询（只读，不涉及事务）"""
        conn = self._get_conn()
        try:
            return self._safe_execute(conn, sql, args, fetch=True, as_dict=as_dict)
        finally:
            conn.close()

    # ========================
    # 手动事务控制（高级用法）
    # ========================

    def begin(self):
        """开始一个手动事务（返回连接对象，需自行管理 commit/rollback）"""
        conn = self._get_conn()
        # autocommit 已为 False，连接即处于事务中
        return conn

    def commit(self, conn):
        """提交事务"""
        conn.commit()

    def rollback(self, conn):
        """回滚事务"""
        conn.rollback()

    def close_conn(self, conn):
        """关闭连接（归还到池）"""
        conn.close()

    # ========================
    # 事务上下文管理器（推荐方式）
    # ========================

    from contextlib import contextmanager

    @contextmanager
    def transaction(self):
        """
        使用示例：
        with db.transaction() as conn:
            db.execute_in_conn(conn, "UPDATE ...")
            db.execute_in_conn(conn, "INSERT ...")
        """
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Manual transaction rolled back: {e}")
            raise
        finally:
            conn.close()

    def execute_in_conn(self, conn, sql: str, args: Optional[tuple] = None) -> int:
        """在已有连接上执行（用于手动事务）"""
        return self._safe_execute(conn, sql, args, fetch=False)

    def query_in_conn(self, conn, sql: str, args: Optional[tuple] = None, as_dict: bool = True):
        """在已有连接上查询（用于手动事务中的中间查询）"""
        return self._safe_execute(conn, sql, args, fetch=True, as_dict=as_dict)


# =================== 使用示例 ===================

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    db = TransactionalMySQLClient(
        host='192.168.99.108',
        user='root',
        password='root',
        database='stock_info',
        autocommit_default=False  # 推荐：让 execute 自身成为一个原子事务
    )

    # ✅ 示例1：单条 execute 自动事务（成功则提交，失败则回滚）
    try:
        db.execute("INSERT INTO test11 (name, age) VALUES (%s, %s)", ("Tom", 11))
        print("Insert succeeded.")
    except Exception as e:
        print("Insert failed:", e)

    # ✅ 示例2：多语句手动事务
    try:
        with db.transaction() as conn:
            db.execute_in_conn(conn, "UPDATE test11 SET age = age - 1 WHERE id = %s", (1,))
            db.execute_in_conn(conn, "UPDATE test11 SET age = age + 100 WHERE id = %s", (1,))
            # 如果中间出错，整个事务回滚
    except Exception as e:
        print("Transfer failed:", e)

    # ✅ 示例3：查询
    users = db.query("SELECT * FROM test11 WHERE name = %s", ("Tom",))
    print(users)