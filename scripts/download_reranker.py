#!/usr/bin/env python3
"""
BGE-Reranker 模型下载脚本

下载 BAAI/bge-reranker-large 模型到 model/ 目录
"""
import os
import sys
from huggingface_hub import snapshot_download

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger_config import logger


def download_bge_reranker(
    model_name: str = "BAAI/bge-reranker-large",
    local_dir: str = "model"
) -> str:
    """
    从 HuggingFace 下载 BGE-Reranker 模型

    Args:
        model_name: HuggingFace 模型 ID
        local_dir: 本地保存目录

    Returns:
        模型保存路径
    """
    logger.info(f"Downloading {model_name} to {local_dir}...")

    try:
        # 创建模型目录
        os.makedirs(local_dir, exist_ok=True)

        # 下载模型
        model_path = snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )

        logger.info(f"Model downloaded successfully to: {model_path}")
        logger.info(f"Please update config/application.yaml with:")
        logger.info(f"  milvus.bge_reranker_path: '{os.path.join(local_dir, model_name.split('/')[-1])}'")

        return model_path

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Download BGE-Reranker model for RAG re-ranking',
        epilog='''
示例:
  # 下载 large 模型（默认）
  python -m scripts.download_reranker

  # 下载 base 模型（更小更快）
  python -m scripts.download_reranker --model BAAI/bge-reranker-base
  python -m scripts.download_reranker --model bge-reranker-base

  # 指定下载目录
  python -m scripts.download_reranker --output-dir /path/to/models
        '''
    )

    parser.add_argument(
        '--model',
        '-m',
        default='BAAI/bge-reranker-large',
        help='Model name to download (default: BAAI/bge-reranker-large)'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        default='model',
        help='Local directory to save model (default: model)'
    )

    args = parser.parse_args()

    try:
        download_bge_reranker(args.model, args.output_dir)
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
