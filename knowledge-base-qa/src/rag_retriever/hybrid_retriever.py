"""混合检索器

结合向量检索和 BM25 关键词检索，通过分数融合返回最相关结果
"""

import hashlib
import json
import logging
import os
import pickle
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from src.vector_store import VectorStore, SearchResult, ChangeEvent, ChangeType

logger = logging.getLogger(__name__)


class FusionMethod(str, Enum):
    """分数融合方式"""

    RRF = "rrf"  # Reciprocal Rank Fusion
    WEIGHTED = "weighted"  # 加权平均


@dataclass
class HybridSearchResult:
    """混合检索结果"""

    document: Document
    vector_score: float
    bm25_score: float
    fused_score: float
    source_id: str


class HybridRetriever:
    """混合检索器：向量 + BM25

    支持：
    - BM25 索引持久化
    - 向量库变更自动同步
    - 增量更新（无需全量重建）
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        fusion_method: FusionMethod = FusionMethod.RRF,
        rrf_k: int = 60,
        persist_dir: Optional[str] = None,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.fusion_method = fusion_method
        self.rrf_k = rrf_k

        # BM25 持久化路径
        self._persist_dir = persist_dir
        self._persist_path = (
            Path(persist_dir) / "bm25_index.pkl" if persist_dir else None
        )

        # BM25 索引状态
        self._bm25_index: Optional[BM25Okapi] = None
        self._bm25_docs: List[Document] = []
        self._bm25_doc_hashes: List[str] = []  # 内容哈希，用于增量同步
        self._tokenized_corpus: List[List[str]] = []

        # 监听向量库变更
        self.vector_store.on_change(self._on_vector_store_change)

    @staticmethod
    def _content_hash(doc: Document) -> str:
        """文档内容哈希"""
        key = f"{doc.metadata.get('source', '')}||{doc.page_content}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _tokenize(self, text: str) -> List[str]:
        """中英文分词"""
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.lower().split()
        cn_chars = re.findall(r"[一-鿿]", text)
        tokens.extend(cn_chars)
        return [t for t in tokens if t.strip()]

    # ==================== 索引持久化 ====================

    def _save_index(self) -> None:
        """保存 BM25 索引到磁盘"""
        if not self._persist_path:
            return

        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "docs": self._bm25_docs,
            "hashes": self._bm25_doc_hashes,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(self._persist_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"BM25 索引已保存: {self._persist_path}")

    def _load_index(self) -> bool:
        """从磁盘加载 BM25 索引"""
        if not self._persist_path or not self._persist_path.exists():
            return False

        try:
            with open(self._persist_path, "rb") as f:
                data = pickle.load(f)
            self._bm25_docs = data["docs"]
            self._bm25_doc_hashes = data["hashes"]
            self._tokenized_corpus = data["tokenized_corpus"]
            self._bm25_index = BM25Okapi(self._tokenized_corpus)
            logger.info(f"BM25 索引已从磁盘加载，文档数: {len(self._bm25_docs)}")
            return True
        except Exception as e:
            logger.warning(f"BM25 索引加载失败: {e}")
            return False

    # ==================== 索引构建与同步 ====================

    def _build_bm25_index(self) -> None:
        """构建 BM25 索引（优先从磁盘加载）"""
        if self._load_index():
            # 校验与向量库一致性
            vector_count = self.vector_store.count()
            if len(self._bm25_docs) == vector_count:
                logger.info("BM25 索引与向量库一致，跳过重建")
                return
            logger.warning(
                f"BM25 索引({len(self._bm25_docs)}) "
                f"与向量库({vector_count})不一致，重建"
            )

        all_docs = self.vector_store.get_all()
        if not all_docs:
            logger.warning("向量存储中无文档，无法构建 BM25 索引")
            return

        self._bm25_docs = all_docs
        self._bm25_doc_hashes = [self._content_hash(d) for d in all_docs]
        self._tokenized_corpus = [self._tokenize(d.page_content) for d in all_docs]
        self._bm25_index = BM25Okapi(self._tokenized_corpus)
        self._save_index()
        logger.info(f"BM25 索引构建完成，文档数: {len(all_docs)}")

    def _on_vector_store_change(self, event: ChangeEvent) -> None:
        """向量库变更回调，增量同步 BM25 索引"""
        if self._bm25_index is None:
            return  # 索引尚未构建，延迟到首次检索时处理

        if event.change_type == ChangeType.CLEAR:
            self._bm25_index = None
            self._bm25_docs = []
            self._bm25_doc_hashes = []
            self._tokenized_corpus = []
            self._save_index()
            logger.info("向量库已清空，BM25 索引已同步清空")
            return

        if event.change_type == ChangeType.DELETE:
            # 按 ID 删除：由于 BM25 不支持按 ID 删除，
            # 标记为脏，下次检索时重建
            self._bm25_index = None
            logger.info("检测到删除操作，BM25 索引标记为待重建")
            return

        if event.change_type in (ChangeType.ADD, ChangeType.UPSERT):
            # 增量添加新文档
            if event.affected_docs:
                for doc in event.affected_docs:
                    doc_hash = self._content_hash(doc)
                    if doc_hash not in self._bm25_doc_hashes:
                        self._bm25_docs.append(doc)
                        self._bm25_doc_hashes.append(doc_hash)
                        self._tokenized_corpus.append(self._tokenize(doc.page_content))
                # 重建 BM25 索引（BM25Okapi 不支持增量添加）
                if self._tokenized_corpus:
                    self._bm25_index = BM25Okapi(self._tokenized_corpus)
                self._save_index()
                logger.info(f"BM25 索引增量更新完成，文档数: {len(self._bm25_docs)}")
            return

        if event.change_type == ChangeType.UPDATE:
            # 更新操作：标记为脏，下次检索时重建
            self._bm25_index = None
            logger.info("检测到更新操作，BM25 索引标记为待重建")

    def _bm25_search(self, query: str, k: int) -> List[SearchResult]:
        """BM25 关键词检索"""
        if self._bm25_index is None:
            self._build_bm25_index()

        if not self._bm25_docs:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokenized_query)

        top_indices = scores.argsort()[-k:][::-1]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append(
                    SearchResult(
                        document=self._bm25_docs[idx],
                        score=float(scores[idx]),
                        id=self._bm25_doc_hashes[idx],
                    )
                )

        return results

    # ==================== 分数融合 ====================

    def _rrf_fusion(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
    ) -> List[HybridSearchResult]:
        """Reciprocal Rank Fusion 分数融合"""
        doc_ranks: Dict[str, Dict[str, Any]] = {}

        for rank, result in enumerate(vector_results):
            doc_id = result.id
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = {
                    "document": result.document,
                    "vector_rank": rank + 1,
                    "bm25_rank": None,
                    "vector_score": result.score,
                    "bm25_score": 0.0,
                }
            doc_ranks[doc_id]["vector_rank"] = rank + 1
            doc_ranks[doc_id]["vector_score"] = result.score

        for rank, result in enumerate(bm25_results):
            doc_id = result.id
            if doc_id not in doc_ranks:
                doc_ranks[doc_id] = {
                    "document": result.document,
                    "vector_rank": None,
                    "bm25_rank": rank + 1,
                    "vector_score": 0.0,
                    "bm25_score": result.score,
                }
            doc_ranks[doc_id]["bm25_rank"] = rank + 1
            doc_ranks[doc_id]["bm25_score"] = result.score

        results = []
        for doc_id, info in doc_ranks.items():
            vector_rrf = (
                1.0 / (self.rrf_k + info["vector_rank"])
                if info["vector_rank"]
                else 0.0
            )
            bm25_rrf = (
                1.0 / (self.rrf_k + info["bm25_rank"])
                if info["bm25_rank"]
                else 0.0
            )
            fused_score = (
                self.vector_weight * vector_rrf + self.bm25_weight * bm25_rrf
            )

            results.append(
                HybridSearchResult(
                    document=info["document"],
                    vector_score=info["vector_score"],
                    bm25_score=info["bm25_score"],
                    fused_score=fused_score,
                    source_id=doc_id,
                )
            )

        results.sort(key=lambda x: x.fused_score, reverse=True)
        return results

    def _weighted_fusion(
        self,
        vector_results: List[SearchResult],
        bm25_results: List[SearchResult],
    ) -> List[HybridSearchResult]:
        """加权平均分数融合"""
        vector_scores = [r.score for r in vector_results]
        bm25_scores = [r.score for r in bm25_results]

        vector_max = max(vector_scores) if vector_scores else 1.0
        bm25_max = max(bm25_scores) if bm25_scores else 1.0

        doc_info: Dict[str, Dict[str, Any]] = {}

        for result in vector_results:
            doc_id = result.id
            normalized_score = result.score / vector_max if vector_max > 0 else 0
            doc_info[doc_id] = {
                "document": result.document,
                "vector_score": normalized_score,
                "bm25_score": 0.0,
            }

        for result in bm25_results:
            doc_id = result.id
            normalized_score = result.score / bm25_max if bm25_max > 0 else 0
            if doc_id in doc_info:
                doc_info[doc_id]["bm25_score"] = normalized_score
            else:
                doc_info[doc_id] = {
                    "document": result.document,
                    "vector_score": 0.0,
                    "bm25_score": normalized_score,
                }

        results = []
        for doc_id, info in doc_info.items():
            fused_score = (
                self.vector_weight * info["vector_score"]
                + self.bm25_weight * info["bm25_score"]
            )
            results.append(
                HybridSearchResult(
                    document=info["document"],
                    vector_score=info["vector_score"],
                    bm25_score=info["bm25_score"],
                    fused_score=fused_score,
                    source_id=doc_id,
                )
            )

        results.sort(key=lambda x: x.fused_score, reverse=True)
        return results

    # ==================== 检索 ====================

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[HybridSearchResult]:
        """混合检索"""
        k = k or self.top_k
        fetch_k = k * 2

        vector_results = self.vector_store.similarity_search_with_score(
            query, k=fetch_k, where=where
        )

        bm25_results = self._bm25_search(query, k=fetch_k)

        if self.fusion_method == FusionMethod.RRF:
            fused_results = self._rrf_fusion(vector_results, bm25_results)
        else:
            fused_results = self._weighted_fusion(vector_results, bm25_results)

        return fused_results[:k]

    def rebuild_index(self) -> None:
        """强制重建 BM25 索引"""
        self._bm25_index = None
        self._bm25_docs = []
        self._bm25_doc_hashes = []
        self._tokenized_corpus = []
        logger.info("BM25 索引已清除，下次检索时将重建")
