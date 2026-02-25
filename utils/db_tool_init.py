
from utils.config_init import application_conf
from utils.mongo_util import MongoManager
from utils.mysql_client import TransactionalMySQLClient
from utils.redis_client import RedisClient
from utils.logger_config import logger

from utils.td_genie_client import  TDEngineClient


def init_mysql():
    mysql_host = application_conf.get_properties("mysql.host")
    mysql_port = application_conf.get_properties("mysql.port")
    mysql_username = application_conf.get_properties("mysql.user")
    mysql_password = application_conf.get_properties("mysql.password")
    mysql_dbname = application_conf.get_properties("mysql.database")
    db = TransactionalMySQLClient(
        host=mysql_host,
        port=int(mysql_port),
        user=mysql_username,
        password=mysql_password,
        database=mysql_dbname,
        autocommit_default=False  # 推荐：让 execute 自身成为一个原子事务
    )
    return db


def init_mongo_db():
    mongo_host = application_conf.get_properties("mongo.host")
    mongo_port = application_conf.get_properties("mongo.port")
    mongo_db_name = application_conf.get_properties("mongo.db_name")
    mongo = MongoManager(
        uri = None,
        host=mongo_host,
        port=mongo_port,
        database_name=mongo_db_name
    )
    return mongo


def init_redis():
    redis_host = application_conf.get_properties("redis.host")
    redis_port = application_conf.get_properties("redis.port")
    redis_db = application_conf.get_properties("redis.db")
    redis_password = application_conf.get_properties("redis.password")
    redis_client = RedisClient(host=redis_host, port=redis_port, db=redis_db, password=redis_password)
    return redis_client

def init_td_engine_client():
    td_engine_host = application_conf.get_properties("td_engine.host")
    td_engine_port = application_conf.get_properties("td_engine.port")
    td_engine_user = application_conf.get_properties("td_engine.user")
    td_engine_password = application_conf.get_properties("td_engine.password")
    td_engine_database = application_conf.get_properties("td_engine.database")
    td_engine_timeout = application_conf.get_properties("td_engine.timeout")
    td_engine_client = TDEngineClient(host=td_engine_host,
                                            port=td_engine_port,
                                            user=td_engine_user,
                                            password=td_engine_password,
                                            timeout=td_engine_timeout,
                                            database=td_engine_database)
    return td_engine_client

mysql_client = init_mysql()
mongo_client = init_mongo_db()
redis_client = init_redis()
td_engine_client = init_td_engine_client()



logger.info("✔ ✔db  mysql mongo  redis init !!!")


