"""
Prompt 模板存储管理模块

数据结构设计:
MongoDB Collection: `prompts`

Document Schema:
{
    "_id": ObjectId,
    "prompt_name": "planner",        # prompt 名称（与文件名对应）
    "template": "...",              # prompt 模板内容（使用 <<VAR>> 语法）
    "description": "Planner agent prompt",  # 描述
    "active": true,                 # 是否激活（使用数据库版本时）
    "version": 1,                   # 版本号
    "created_at": datetime,         # 创建时间
    "updated_at": datetime,         # 更新时间
    "created_by": "user",           # 创建者（可选）
    "tags": ["agent", "planner"]    # 标签（可选）
}
"""
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from utils.db_tool_init import mongo_client
from utils.logger_config import logger


class PromptStore:
    """Prompt 模板 MongoDB 存储管理"""

    COLLECTION_NAME = "prompts"

    def __init__(self):
        self.collection = mongo_client.get_collection(self.COLLECTION_NAME)
        self._ensure_indexes()

    def _ensure_indexes(self):
        """确保必要的索引存在"""
        try:
            # 为 prompt_name 创建索引以提高查询性能
            self.collection.create_index([("prompt_name", 1)])
            # 为 active 创建索引以便快速查询激活的 prompt
            self.collection.create_index([("active", 1)])
            # 为 updated_at 创建索引以便排序
            self.collection.create_index([("updated_at", -1)])
            logger.debug("Prompt indexes ensured")
        except Exception as e:
            logger.warning(f"Failed to create indexes: {e}")

    def get_prompt(self, prompt_name: str, version: Optional[int] = None) -> Optional[str]:
        """
        获取 prompt 模板

        Args:
            prompt_name: prompt 名称
            version: 可选版本号，不指定则获取最新激活版本

        Returns:
            prompt 模板内容，如果不存在则返回 None
        """
        query = {"prompt_name": prompt_name}

        if version is not None:
            # 指定版本号
            query["version"] = version
        else:
            # 获取最新激活版本
            query["active"] = True

        doc = self.collection.find_one(query, sort=[("version", -1)])

        if doc:
            return doc.get("template")

        return None

    def save_prompt(
        self,
        prompt_name: str,
        template: str,
        description: str = "",
        created_by: str = "system"
    ) -> str:
        """
        保存或更新 prompt 模板

        如果同名 prompt 已存在，则创建新版本；
        如果不存在，则创建版本 1。

        Args:
            prompt_name: prompt 名称
            template: 模板内容
            description: 描述
            created_by: 创建者

        Returns:
            文档 ID
        """
        now = datetime.utcnow()

        # 检查是否已存在
        existing = self.collection.find_one(
            {"prompt_name": prompt_name},
            sort=[("version", -1)]
        )

        if existing:
            # 创建新版本
            new_version = existing.get("version", 0) + 1

            # 取消激活旧版本
            self.collection.update_many(
                {"prompt_name": prompt_name},
                {"$set": {"active": False}}
            )

            doc = {
                "prompt_name": prompt_name,
                "template": template,
                "description": description,
                "active": True,
                "version": new_version,
                "created_at": now,
                "updated_at": now,
                "created_by": created_by,
                "tags": self._infer_tags(prompt_name)
            }
        else:
            # 首次创建
            doc = {
                "prompt_name": prompt_name,
                "template": template,
                "description": description,
                "active": True,
                "version": 1,
                "created_at": now,
                "updated_at": now,
                "created_by": created_by,
                "tags": self._infer_tags(prompt_name)
            }

        result = self.collection.insert_one(doc)
        logger.info(f"Saved prompt '{prompt_name}' version {doc['version']}, _id: {result.inserted_id}")
        return str(result.inserted_id)

    def update_prompt(
        self,
        prompt_name: str,
        template: str,
        description: Optional[str] = None
    ) -> bool:
        """
        更新现有的 prompt 模板（创建新版本）

        Args:
            prompt_name: prompt 名称
            template: 模板内容
            description: 描述（可选，不提供则保持原描述）

        Returns:
            bool: 是否成功更新
        """
        # 获取当前版本
        existing = self.collection.find_one(
            {"prompt_name": prompt_name},
            sort=[("version", -1)]
        )

        if not existing:
            logger.warning(f"Prompt '{prompt_name}' not found, cannot update")
            return False

        now = datetime.utcnow()
        new_version = existing.get("version", 1) + 1

        # 取消激活旧版本
        self.collection.update_many(
            {"prompt_name": prompt_name},
            {"$set": {"active": False}}
        )

        # 创建新版本
        new_doc = {
            "prompt_name": prompt_name,
            "template": template,
            "description": description if description is not None else existing.get("description", ""),
            "active": True,
            "version": new_version,
            "created_at": now,
            "updated_at": now,
            "created_by": existing.get("created_by", "system"),
            "tags": existing.get("tags", [])
        }

        result = self.collection.insert_one(new_doc)
        logger.info(f"Updated prompt '{prompt_name}' to version {new_version}, _id: {result.inserted_id}")
        return True

    def list_prompts(
        self,
        active_only: bool = False,
        tag: Optional[str] = None,
        prompt_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        列出 prompt 模板

        Args:
            active_only: 是否只返回激活的 prompt
            tag: 按标签过滤
            prompt_name: 按 prompt 名称过滤
            limit: 返回数量限制

        Returns:
            prompt 列表
        """
        query = {}

        if active_only:
            query["active"] = True

        if tag:
            query["tags"] = {"$in": [tag]}

        if prompt_name:
            query["prompt_name"] = {"$regex": prompt_name, "$options": "i"}

        prompts = list(
            self.collection.find(query)
            .sort("updated_at", -1)
            .limit(limit)
        )

        for prompt in prompts:
            if "_id" in prompt:
                prompt["_id"] = str(prompt["_id"])

        return prompts

    def delete_prompt(self, prompt_name: str) -> int:
        """
        删除指定名称的所有版本

        Args:
            prompt_name: prompt 名称

        Returns:
            int: 删除的文档数量
        """
        result = self.collection.delete_many({"prompt_name": prompt_name})
        if result.deleted_count > 0:
            logger.info(f"Deleted all versions of prompt '{prompt_name}'")
        return result.deleted_count

    def activate_prompt(self, prompt_name: str, version: Optional[int] = None) -> bool:
        """
        激活指定 prompt（设置 active=true，其他同名版本的设置为 false）

        Args:
            prompt_name: prompt 名称
            version: 可选版本号，不指定则激活最新版本

        Returns:
            bool: 是否成功激活
        """
        if version is None:
            # 激活最新版本
            latest = self.collection.find_one(
                {"prompt_name": prompt_name},
                sort=[("version", -1)]
            )
            if not latest:
                return False
            version = latest.get("version")

        # 取消激活所有版本
        self.collection.update_many(
            {"prompt_name": prompt_name},
            {"$set": {"active": False}}
        )

        # 激活指定版本
        result = self.collection.update_one(
            {"prompt_name": prompt_name, "version": version},
            {"$set": {"active": True, "updated_at": datetime.utcnow()}}
        )

        if result.modified_count > 0:
            logger.info(f"Activated prompt '{prompt_name}' version {version}")
            return True

        return False

    def import_from_file(self, prompt_name: str, prompts_dir: str) -> Optional[Dict[str, Any]]:
        """
        从文件导入 prompt 到数据库

        用于初始化或迁移现有 prompt 文件到数据库

        Args:
            prompt_name: prompt 名称（不带 .md 扩展名）
            prompts_dir: prompt 文件目录

        Returns:
            导入的 prompt 文档字典，如果文件不存在则返回 None
        """
        file_path = os.path.join(prompts_dir, f"{prompt_name}.md")

        if not os.path.exists(file_path):
            logger.warning(f"Prompt file not found: {file_path}")
            return None

        with open(file_path, "r", encoding="utf-8") as f:
            template = f.read()

        # 检查是否已存在
        existing = self.collection.find_one({"prompt_name": prompt_name})

        if existing:
            logger.info(f"Prompt '{prompt_name}' already exists in database, skipping import")
            return existing

        # 创建新文档
        now = datetime.utcnow()
        doc = {
            "prompt_name": prompt_name,
            "template": template,
            "description": f"Imported from file: {file_path}",
            "active": True,
            "version": 1,
            "created_at": now,
            "updated_at": now,
            "created_by": "import_script",
            "tags": self._infer_tags(prompt_name)
        }

        result = self.collection.insert_one(doc)
        logger.info(f"Imported prompt '{prompt_name}' from file, _id: {result.inserted_id}")

        doc["_id"] = str(result.inserted_id)
        return doc

    def import_all_from_directory(self, prompts_dir: str) -> int:
        """
        批量导入目录下所有 .md prompt 文件

        Args:
            prompts_dir: prompt 文件目录

        Returns:
            导入的 prompt 数量
        """
        if not os.path.exists(prompts_dir):
            logger.error(f"Prompts directory not found: {prompts_dir}")
            return 0

        count = 0
        for filename in os.listdir(prompts_dir):
            if filename.endswith(".md"):
                prompt_name = filename[:-3]  # 去掉 .md 扩展名
                if self.import_from_file(prompt_name, prompts_dir):
                    count += 1

        logger.info(f"Imported {count} prompts from directory: {prompts_dir}")
        return count

    def _infer_tags(self, prompt_name: str) -> List[str]:
        """
        根据名称推断标签

        Args:
            prompt_name: prompt 名称

        Returns:
            标签列表
        """
        tags = []

        # 根据名称推断一些基础标签
        if prompt_name in ["planner", "supervisor", "coder", "browser", "reporter",
                      "url_to_markdown", "rag", "file_manager"]:
            tags.append("agent")

        if prompt_name == "planner":
            tags.append("planning")
        elif prompt_name == "supervisor":
            tags.append("coordination")
        elif prompt_name == "rag":
            tags.append("retrieval")
            tags.append("knowledge")
        elif prompt_name == "coder":
            tags.append("code")
        elif prompt_name == "browser":
            tags.append("web")
            tags.append("vision")

        return tags


# 全局单例
prompt_store = PromptStore()
