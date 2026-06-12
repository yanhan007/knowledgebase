"""结果重排模块

支持多种重排策略：
- 交叉编码器重排（需要 sentence-transformers）
- 多样性重排（MMR）
- 元数据加权重排
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.rag_retriever.hybrid_retriever import HybridSearchResult

logger = logging.getLogger(__name__)


class RerankStrategy(str, Enum):
    """重排策略"""

    CROSS_ENCODER = "cross_encoder"
    MMR = "mmr"  # Maximal Marginal Relevance
    METADATA_BOOST = "metadata_boost"
    NONE = "none"


@dataclass
class RerankConfig:
    """重排配置"""

    strategy: RerankStrategy = RerankStrategy.MMR
    diversity_weight: float = 0.3  # MMR 多样性权重
    metadata_boost_field: Optional[str] = None  # 元数据加权字段
    metadata_boost_weight: float = 0.2
    cross_encoder_model: str = "BAAI/bge-reranker-base"
    max_rerank_docs: int = 20  # 交叉编码器最大重排文档数


class Reranker:
    """结果重排器"""

    def __init__(self, config: Optional[RerankConfig] = None):
        self.config = config or RerankConfig()
        self._cross_encoder = None
        self._cross_encoder_loaded = False

    @property
    def cross_encoder(self):
        """延迟加载交叉编码器"""
        if self._cross_encoder is None and not self._cross_encoder_loaded:
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"加载交叉编码器: {self.config.cross_encoder_model}")
                self._cross_encoder = CrossEncoder(
                    self.config.cross_encoder_model, max_length=512
                )
                self._cross_encoder_loaded = True
                logger.info("交叉编码器加载完成")
            except Exception as e:
                self._cross_encoder_loaded = True
                logger.error(f"交叉编码器加载失败: {e}")
                raise
        return self._cross_encoder

    def rerank(
        self,
        query: str,
        results: List[HybridSearchResult],
        top_k: Optional[int] = None,
    ) -> List[HybridSearchResult]:
        """重排检索结果"""
        if not results:
            return []

        strategy = self.config.strategy
        if strategy == RerankStrategy.CROSS_ENCODER:
            reranked = self._cross_encoder_rerank(query, results)
        elif strategy == RerankStrategy.MMR:
            reranked = self._mmr_rerank(query, results)
        elif strategy == RerankStrategy.METADATA_BOOST:
            reranked = self._metadata_boost_rerank(results)
        else:
            reranked = results

        if top_k:
            reranked = reranked[:top_k]

        return reranked

    def _cross_encoder_rerank(
        self, query: str, results: List[HybridSearchResult]
    ) -> List[HybridSearchResult]:
        """使用交叉编码器重排"""
        try:
            # 限制重排文档数
            to_rerank = results[: self.config.max_rerank_docs]
            rest = results[self.config.max_rerank_docs :]

            pairs = [(query, r.document.page_content) for r in to_rerank]
            scores = self.cross_encoder.predict(pairs)

            for i, result in enumerate(to_rerank):
                result.fused_score = float(scores[i])

            to_rerank.sort(key=lambda x: x.fused_score, reverse=True)
            logger.info(
                f"交叉编码器重排完成，重排 {len(to_rerank)} / {len(results)} 个文档"
            )

            return to_rerank + rest

        except Exception as e:
            logger.warning(f"交叉编码器重排失败，使用原始排序: {e}")
            return results

    def _mmr_rerank(
        self, query: str, results: List[HybridSearchResult]
    ) -> List[HybridSearchResult]:
        """MMR（最大边际相关性）重排，平衡相关性和多样性"""
        if len(results) <= 1:
            return results

        lambda_param = 1.0 - self.config.diversity_weight
        selected: List[HybridSearchResult] = []
        candidates = list(results)

        candidates.sort(key=lambda x: x.fused_score, reverse=True)
        selected.append(candidates.pop(0))

        while candidates:
            best_score = -float("inf")
            best_idx = -1

            for i, candidate in enumerate(candidates):
                relevance = candidate.fused_score
                max_similarity = max(
                    self._compute_similarity(candidate.document, s.document)
                    for s in selected
                )
                mmr_score = lambda_param * relevance - (
                    1 - lambda_param
                ) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx >= 0:
                selected.append(candidates.pop(best_idx))

        logger.info(f"MMR 重排完成，结果数: {len(selected)}")
        return selected

    def _compute_similarity(self, doc1: Document, doc2: Document) -> float:
        """计算两个文档的相似度（基于字符级 Jaccard）"""
        set1 = set(doc1.page_content)
        set2 = set(doc2.page_content)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _metadata_boost_rerank(
        self, results: List[HybridSearchResult]
    ) -> List[HybridSearchResult]:
        """基于元数据加权重排"""
        field = self.config.metadata_boost_field
        if not field:
            return results

        boost_weight = self.config.metadata_boost_weight

        for result in results:
            metadata_value = result.document.metadata.get(field)
            if metadata_value is not None:
                boost = self._calculate_metadata_boost(metadata_value)
                result.fused_score *= 1.0 + boost_weight * boost

        results.sort(key=lambda x: x.fused_score, reverse=True)
        logger.info(f"元数据加权重排完成，字段: {field}")
        return results

    def _calculate_metadata_boost(self, value: Any) -> float:
        """计算元数据加权值"""
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            return min(abs(value), 1.0)
        elif isinstance(value, str):
            return 1.0 if value else 0.0
        return 0.0
