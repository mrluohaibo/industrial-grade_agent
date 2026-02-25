"""
会话上下文中间件

在workflow执行前后自动处理会话上下文的加载和保存
"""
from typing import Callable, Dict, Any, Optional, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from bz_agent.storage import conversation_store
from utils.logger_config import logger


class ContextMiddleware:
    """会话上下文中间件"""

    def __init__(self):
        self.store = conversation_store

    def pre_process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Workflow执行前的预处理

        - 如果没有session_id，创建新会话
        - 如果有session_id，从MongoDB加载历史消息

        Args:
            state: 初始状态字典

        Returns:
            更新后的状态字典
        """
        session_id = state.get("session_id")

        # 创建新会话
        if not session_id:
            session_id = self.store.create_session()
            state["session_id"] = session_id
            logger.info(f"Created new session: {session_id}")
        else:
            # 验证会话是否存在，不存在则创建
            session = self.store.get_session(session_id)
            if not session:
                self.store.create_session(session_id=session_id)
                logger.info(f"Created session with provided ID: {session_id}")
            else:
                # 加载历史消息
                existing_messages = self.store.get_messages(session_id)
                if existing_messages:
                    # 合并消息（避免重复）
                    state_messages = state.get("messages", [])

                    # 将state中的消息转换为统一格式
                    normalized_state_messages = self._normalize_messages(state_messages)

                    # 去重：只添加不在历史消息中的消息
                    new_messages = self._filter_new_messages(existing_messages, normalized_state_messages)

                    # 合并：历史消息 + 新消息
                    state["messages"] = existing_messages + new_messages
                    logger.info(f"Loaded {len(existing_messages)} messages from session {session_id}, added {len(new_messages)} new messages")

        return state

    def post_process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Workflow执行后的后处理

        - 保存新消息到MongoDB

        Args:
            state: workflow执行后的状态字典

        Returns:
            更新后的状态字典
        """
        session_id = state.get("session_id")
        messages = state.get("messages", [])

        if session_id and messages:
            # 获取保存前的消息数量
            session = self.store.get_session(session_id)
            if session:
                saved_count = len(session.get("messages", []))
            else:
                saved_count = 0

            # 保存所有消息
            normalized_messages = self._normalize_messages(messages)
            count = self.store.add_messages(session_id, normalized_messages)

            logger.info(f"Saved {count} messages to session {session_id} (total: {saved_count + count})")

        return state

    def _normalize_messages(self, messages: List[Any]) -> List[BaseMessage]:
        """
        将消息列表标准化为LangChain BaseMessage对象列表

        Args:
            messages: 可能是字典或BaseMessage对象的列表

        Returns:
            标准化后的BaseMessage列表
        """
        normalized = []
        for msg in messages:
            if isinstance(msg, BaseMessage):
                normalized.append(msg)
            elif isinstance(msg, dict):
                # 尝试从字典创建消息对象
                content = msg.get("content", "")
                role = msg.get("role", "user")

                if role == "user":
                    normalized.append(HumanMessage(content=content))
                elif role == "assistant":
                    normalized.append(AIMessage(content=content))
                else:
                    # 其他类型，默认用HumanMessage
                    normalized.append(HumanMessage(content=content))
            else:
                logger.warning(f"Unsupported message type: {type(msg)}")

        return normalized

    def _filter_new_messages(
        self,
        existing_messages: List[BaseMessage],
        new_messages: List[BaseMessage]
    ) -> List[BaseMessage]:
        """
        过滤出新消息，排除与历史消息重复的内容

        Args:
            existing_messages: 已存在的消息列表
            new_messages: 新消息列表

        Returns:
            不在existing_messages中的新消息列表
        """
        if not existing_messages:
            return new_messages

        # 简单的去重逻辑：比较内容
        existing_contents = {msg.content for msg in existing_messages}
        filtered = [msg for msg in new_messages if msg.content not in existing_contents]

        return filtered


# 全局单例
context_middleware = ContextMiddleware()


def with_context_middleware(workflow_func: Callable) -> Callable:
    """
    装饰器：为workflow函数添加会话上下文处理

    Usage:
        @with_context_middleware
        def run_workflow(initial_state: Dict[str, Any]) -> Dict[str, Any]:
            # workflow logic here
            pass
    """
    def wrapper(initial_state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # 预处理
        state = context_middleware.pre_process(initial_state)

        # 执行workflow
        result = workflow_func(state, **kwargs)

        # 后处理
        result = context_middleware.post_process(result)

        return result

    return wrapper
