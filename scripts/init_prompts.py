#!/usr/bin/env python3
"""
初始化脚本：将现有 prompt 文件导入到 MongoDB

使用方法:
    python -m scripts.init_prompts

功能:
1. 扫描 bz_agent/prompts 目录下所有 .md 文件
2. 将每个文件作为 prompt 模板导入到 MongoDB
3. 如果 prompt 已存在则跳过（避免覆盖）
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bz_agent.prompts.template import import_prompts_to_mongo
from utils.logger_config import logger


def main():
    """主函数：导入所有 prompt 文件到 MongoDB"""
    logger.info("=" * 60)
    logger.info("Starting prompt import to MongoDB...")
    logger.info("=" * 60)

    # 导入 prompts
    count = import_prompts_to_mongo()

    logger.info("=" * 60)
    logger.info(f"Prompt import completed. Total imported: {count}")
    logger.info("=" * 60)

    # 验证导入
    from bz_agent.storage import prompt_store
    all_prompts = prompt_store.list_prompts()
    active_prompts = prompt_store.list_prompts(active_only=True)

    logger.info(f"Total prompts in MongoDB: {len(all_prompts)}")
    logger.info(f"Active prompts in MongoDB: {len(active_prompts)}")

    # 打印所有导入的 prompt
    logger.info("Imported prompts:")
    for prompt in all_prompts:
        status = " [ACTIVE]" if prompt.get("active") else " [INACTIVE]"
        logger.info(f"  - {prompt['prompt_name']}: version {prompt.get('version')}{status}")

    return count


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nImport interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during prompt import: {e}")
        sys.exit(1)
