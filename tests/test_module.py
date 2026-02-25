import unittest
from utils.db_tool_init import td_engine_client

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_temp1(self):
        data = [
            {"name": "Alice", "age": 30, "city": "Beijing"},
            {"name": "Bob", "age": 25, "city": "Shanghai"}
        ]

        # 指定字段顺序
        keys = ["name", "age", "city"]
        result = [tuple(item[k] for k in keys) for item in data]
        print(result)

    def test_td_engine(self):
        # 使用上下文管理器自动关闭连接

        td_client = td_engine_client
        with td_client as client:
            # # 创建数据库
            # client.execute("CREATE DATABASE IF NOT EXISTS test")
            #
            # # 切换数据库（或通过构造函数指定）
            client.execute("USE test")
            #
            # # 创建超级表
            # client.execute("""
            #        CREATE STABLE IF NOT EXISTS meters (
            #            ts TIMESTAMP,
            #            current FLOAT,
            #            voltage INT,
            #            phase FLOAT
            #        ) TAGS (location BINARY(64))
            #        """)
            #
            # # 创建子表
            # client.execute("CREATE TABLE d1001 USING meters TAGS ('California.SF')")

            # 插入单条数据
            client.execute("""
                           INSERT INTO d1001
                           VALUES ('2026-01-06 21:00:00', 10.3, 219, 0.31)
                           """)

            # 批量插入（推荐方式，高效且安全）
            data = [
                ("2026-01-06 21:01:00", 12.6, 218, 0.33),
                ("2026-01-06 21:02:00", 11.2, 220, 0.32),
            ]
            client.insert_many(
                "INSERT INTO d1001 VALUES (?, ?, ?, ?)",
                data
            )

            # 查询数据
            rows = client.query("SELECT * FROM meters LIMIT 5")
            for row in rows:
                print(row)



if __name__ == '__main__':
    unittest.main()
