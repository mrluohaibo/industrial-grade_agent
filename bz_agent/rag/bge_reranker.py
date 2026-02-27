"""
BGE-Reranker 重排序实现

使用 BAAI/bge-reranker-large 模型对 RAG 检索结果进行重排序
"""
from typing import List, Dict
import torch

from FlagEmbedding import BGEM3FlagModel
from utils.logger_config import logger


class BGEReranker:
    """BGE-Reranker 重排序器

    使用跨编码器（cross-encoder）计算 query 与 passage 的语义相似度
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化 BGE-Reranker 模型

        Args:
            model_path: 模型路径（本地路径或 HuggingFace ID）
            device: 'cuda' | 'cpu' | 'auto'
        """
        # 自动检测设备
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading BGE-Reranker from: {model_path} on device: {self.device}")

        try:
            # 加载模型，使用 FP16 加速
            self.model = BGEM3FlagModel(
                model_name_or_path=model_path,
                use_fp16=True,
                device=self.device
            )
            logger.info(f"BGE-Reranker loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BGE-Reranker: {e}")
            raise

    def rerank(
        self,
        query: str,
        passages: List[Dict[str, str]],
        top_k: int = 10
    ) -> List[Dict]:
        """
        对候选文档进行重排序

        Args:
            query: 用户查询
            passages: [{"id": doc_id, "text": content}]
            top_k: 返回前 k 个

        Returns:
            [{"id": doc_id, "score": score}]，按 score 降序
        """
        if not passages:
            logger.warning("No passages to rerank")
            return []

        # 1. 编码 query
        try:
            query_emb = self.model.encode_queries(
                [query],
                return_dense=True
            )["dense_vecs"][0]  # [1024]
        except Exception as e:
            logger.error(f"Failed to encode query: {e}")
            return []

        # 2. 编码 passages
        passage_texts = [p["text"] for p in passages]
        try:
            # FlagEmbedding 使用 encode 方法，传入 passages 列表
            passage_embs = self.model.encode(
                passage_texts,
                return_dense=True
            )["dense_vecs"]  # [N, 1024]
        except Exception as e:
            logger.error(f"Failed to encode passages: {e}")
            return []

        # 3. 计算相似度分数 (query 与每个 passage 的点积)
        # query_emb: [1024], passage_embs: [N, 1024]
        # 需要转置: [1024, N]
        try:
            scores = torch.matmul(
                query_emb.unsqueeze(0),  # [1, 1024]
                passage_embs.T  # [1024, N]
            ).squeeze(0)  # [N]
        except Exception as e:
            logger.error(f"Failed to compute scores: {e}")
            return []

        # 4. 排序并返回
        doc_ids = [p["id"] for p in passages]
        score_list = scores.tolist()

        # 组合 ID 和分数
        score_pairs = list(zip(doc_ids, score_list))

        # 按分数降序排序
        sorted_pairs = sorted(score_pairs, key=lambda x: x[1], reverse=True)

        # 取前 top_k 个
        result = [{"id": doc_id, "score": float(score)}
                  for doc_id, score in sorted_pairs[:top_k]]

        logger.info(f"Reranked {len(passages)} passages, returning top {len(result)}")
        for i, r in enumerate(result[:5]):
            logger.info(f"  Rank {i+1}: ID={r['id']}, Score={r['score']:.4f}")

        return result
