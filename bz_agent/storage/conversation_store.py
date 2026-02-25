"""
会话上下文存储管理模块

数据结构设计:
MongoDB Collection: `conversations`

Document Schema:
{
    "_id": ObjectId,
    "session_id": str,           # 会话唯一标识
    "messages": List[Dict],       # 消息历史列表 (LangChain Message格式)
    "created_at": datetime,       # 会话创建时间
    "updated_at": datetime,       # 最后更新时间
    "metadata": Dict,             # 可选: 额外元数据
    "user_id": Optional[str],     # 可选: 用户标识
    "status": str                 # 会话状态: "active", "closed", "archived"
}
"""
from datetime import datetime
from typing import List, Dict, Optional, Any

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

from utils.db_tool_init import mongo_client
from utils.logger_config import logger


class ConversationStore:
    """会话上下文MongoDB存储管理"""

    COLLECTION_NAME = "conversations"
    MAX_MESSAGES_PER_SESSION = 1000  # 单会话最大消息数限制

    def __init__(self):
        self.collection = mongo_client.get_collection(self.COLLECTION_NAME)
        self._ensure_indexes()

    def _ensure_indexes(self):
        """确保必要的索引存在"""
        try:
            # 为 session_id 创建索引以提高查询性能
            self.collection.create_index([("session_id", 1)], unique=True)
            # 为 user_id 创建索引以便按用户查询
            self.collection.create_index([("user_id", 1)])
            # 为 updated_at 创建索引以便排序
            self.collection.create_index([("updated_at", -1)])
            logger.debug("Conversation indexes ensured")
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")

    def generate_session_id(self) -> str:
        """生成新的会话ID"""
        import uuid
        return str(uuid.uuid4())

    def create_session(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        创建新会话

        Args:
            session_id: 可选的会话ID，如果不提供则自动生成
            user_id: 可选的用户标识
            metadata: 可选的元数据

        Returns:
            session_id: 会话ID
        """
        if session_id is None:
            session_id = self.generate_session_id()

        # 检查会话是否已存在
        existing = self.collection.find_one({"session_id": session_id})
        if existing:
            logger.debug(f"Session {session_id} already exists, returning existing")
            return session_id

        now = datetime.utcnow()
        doc = {
            "session_id": session_id,
            "messages": [],
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {},
            "user_id": user_id,
            "status": "active"
        }

        result = self.collection.insert_one(doc)
        logger.info(f"Created new session: {session_id}, _id: {result.inserted_id}")
        return session_id

    def add_message(
        self,
        session_id: str,
        message: BaseMessage,
        user_id: Optional[str] = None
    ) -> bool:
        """
        向会话添加消息

        Args:
            session_id: 会话ID
            message: LangChain BaseMessage对象
            user_id: 可选的用户标识

        Returns:
            bool: 是否成功添加
        """
        message_dict = message_to_dict(message)

        update_result = self.collection.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": message_dict},
                "$set": {
                    "updated_at": datetime.utcnow(),
                    "status": "active"
                }
            }
        )

        if update_result.modified_count > 0:
            logger.debug(f"Added message to session {session_id}")

            # 检查消息数量限制
            session = self.collection.find_one({"session_id": session_id})
            if len(session["messages"]) > self.MAX_MESSAGES_PER_SESSION:
                self._trim_messages(session_id, self.MAX_MESSAGES_PER_SESSION)

            return True
        else:
            logger.warning(f"Failed to add message to session {session_id}")
            return False

    def add_messages(
        self,
        session_id: str,
        messages: List[BaseMessage],
        user_id: Optional[str] = None
    ) -> int:
        """
        批量向会话添加消息

        Args:
            session_id: 会话ID
            messages: LangChain BaseMessage对象列表
            user_id: 可选的用户标识

        Returns:
            int: 成功添加的消息数量
        """
        message_dicts = [message_to_dict(msg) for msg in messages]

        update_result = self.collection.update_one(
            {"session_id": session_id},
            {
                "$push": {"messages": {"$each": message_dicts}},
                "$set": {
                    "updated_at": datetime.utcnow(),
                    "status": "active"
                }
            }
        )

        if update_result.modified_count > 0:
            added_count = len(message_dicts)
            logger.debug(f"Added {added_count} messages to session {session_id}")

            # 检查消息数量限制
            session = self.collection.find_one({"session_id": session_id})
            if len(session["messages"]) > self.MAX_MESSAGES_PER_SESSION:
                self._trim_messages(session_id, self.MAX_MESSAGES_PER_SESSION)

            return added_count
        else:
            logger.warning(f"Failed to add messages to session {session_id}")
            return 0

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[BaseMessage]:
        """
        获取会话的消息历史

        Args:
            session_id: 会话ID
            limit: 可选，限制返回的消息数量（返回最后N条）

        Returns:
            List[BaseMessage]: LangChain Message对象列表
        """
        session = self.collection.find_one({"session_id": session_id})

        if not session:
            logger.debug(f"Session {session_id} not found")
            return []

        messages = session.get("messages", [])

        if limit and limit < len(messages):
            messages = messages[-limit:]

        # 转换为LangChain Message对象
        return messages_from_dict(messages)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取完整会话信息

        Args:
            session_id: 会话ID

        Returns:
            会话字典，包含消息、元数据等
        """
        session = self.collection.find_one({"session_id": session_id})

        if session and "_id" in session:
            session["_id"] = str(session["_id"])

        return session

    def update_metadata(
        self,
        session_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        更新会话元数据

        Args:
            session_id: 会话ID
            metadata: 要更新的元数据字典

        Returns:
            bool: 是否成功更新
        """
        update_result = self.collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "metadata": metadata,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        return update_result.modified_count > 0

    def close_session(self, session_id: str) -> bool:
        """
        关闭会话

        Args:
            session_id: 会话ID

        Returns:
            bool: 是否成功关闭
        """
        update_result = self.collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "status": "closed",
                    "updated_at": datetime.utcnow()
                }
            }
        )
        if update_result.modified_count > 0:
            logger.info(f"Closed session {session_id}")
        return update_result.modified_count > 0

    def delete_session(self, session_id: str) -> int:
        """
        删除会话

        Args:
            session_id: 会话ID

        Returns:
            int: 删除的文档数量
        """
        result = self.collection.delete_one({"session_id": session_id})
        if result.deleted_count > 0:
            logger.info(f"Deleted session {session_id}")
        return result.deleted_count

    def _trim_messages(self, session_id: str, keep_count: int) -> None:
        """
        裁剪消息历史，保留最新的keep_count条消息

        Args:
            session_id: 会话ID
            keep_count: 保留的消息数量
        """
        session = self.collection.find_one({"session_id": session_id})
        if not session:
            return

        messages = session.get("messages", [])
        if len(messages) <= keep_count:
            return

        # 保留最新的keep_count条消息
        trimmed_messages = messages[-keep_count:]

        self.collection.update_one(
            {"session_id": session_id},
            {
                "$set": {
                    "messages": trimmed_messages,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        logger.info(f"Trimmed session {session_id} messages to {keep_count}")

    def list_sessions(
        self,
        user_id: Optional[str] = None,
        status: str = "active",
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        列出会话

        Args:
            user_id: 可选的用户标识
            status: 会话状态过滤
            limit: 返回数量限制

        Returns:
            会话列表
        """
        query = {"status": status}
        if user_id:
            query["user_id"] = user_id

        sessions = list(
            self.collection.find(query)
            .sort("updated_at", -1)
            .limit(limit)
        )

        for session in sessions:
            if "_id" in session:
                session["_id"] = str(session["_id"])

        return sessions


# 全局单例
conversation_store = ConversationStore()
