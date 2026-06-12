"""RAG 检索管线

整合混合检索、重排、上下文管理和引用溯源
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain_core.documents import Document

from src.vector_store import VectorStore, VectorStoreProtocol
from src.rag_retriever.hybrid_retriever import HybridRetriever, HybridSearchResult
from src.rag_retriever.reranker import Reranker, RerankConfig, RerankStrategy
from src.rag_retriever.context_manager import (
    ContextManager,
    ContextConfig,
    ContextWindow,
    default_token_counter,
)
from src.rag_retriever.citation import CitationManager, Citation, CitationContext

logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """RAG 配置"""

    # 检索配置
    retrieval_top_k: int = 10
    vector_weight: float = 0.7
    bm25_weight: float = 0.3

    # 重排配置
    rerank_strategy: RerankStrategy = RerankStrategy.MMR
    rerank_top_k: int = 5
    diversity_weight: float = 0.3
    max_rerank_docs: int = 20  # 交叉编码器最大重排文档数

    # 上下文配置
    max_context_tokens: int = 4000
    context_overlap_chars: int = 100  # 截断时的重叠字符数
    context_overlap_sentences: int = 1

    # 引用配置
    preview_length: int = 100

    # BM25 持久化
    bm25_persist_dir: Optional[str] = None

    # 自定义 token 计数器
    token_counter: Optional[Callable[[str], int]] = None


@dataclass
class RAGResult:
    """RAG 检索结果"""

    context: str  # 拼接后的上下文
    context_window: ContextWindow  # 上下文窗口信息
    citation_context: CitationContext  # 引用上下文
    search_results: List[HybridSearchResult]  # 检索结果
    citations: List[Citation]  # 引用列表


class RAGPipeline:
    """RAG 检索管线

    接收 VectorStoreProtocol 兼容的向量存储实例，
    不依赖具体实现。
    """

    def __init__(
        self,
        vector_store: VectorStoreProtocol,
        config: Optional[RAGConfig] = None,
    ):
        self.config = config or RAGConfig()
        self.vector_store = vector_store

        # 初始化组件
        self.hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            top_k=self.config.retrieval_top_k,
            vector_weight=self.config.vector_weight,
            bm25_weight=self.config.bm25_weight,
            persist_dir=self.config.bm25_persist_dir,
        )

        self.reranker = Reranker(
            config=RerankConfig(
                strategy=self.config.rerank_strategy,
                diversity_weight=self.config.diversity_weight,
                max_rerank_docs=self.config.max_rerank_docs,
            )
        )

        self.context_manager = ContextManager(
            config=ContextConfig(
                max_tokens=self.config.max_context_tokens,
                context_overlap_chars=self.config.context_overlap_chars,
                context_overlap_sentences=self.config.context_overlap_sentences,
            ),
            token_counter=self.config.token_counter or default_token_counter,
        )

        self.citation_manager = CitationManager(
            preview_length=self.config.preview_length
        )

    def retrieve(
        self,
        query: str,
        where: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> RAGResult:
        """执行完整的 RAG 检索流程"""
        logger.info(f"开始 RAG 检索: {query[:50]}...")

        # 1. 混合检索
        search_results = self.hybrid_retriever.retrieve(
            query=query,
            k=self.config.retrieval_top_k,
            where=where,
        )
        logger.info(f"混合检索完成，结果数: {len(search_results)}")

        # 2. 重排
        reranked_results = self.reranker.rerank(
            query=query,
            results=search_results,
            top_k=self.config.rerank_top_k,
        )
        logger.info(f"重排完成，结果数: {len(reranked_results)}")

        # 3. 提取文档和分数
        documents = [r.document for r in reranked_results]
        scores = [r.fused_score for r in reranked_results]

        # 4. 构建上下文（带重叠）
        context_window = self.context_manager.build_context_with_overlap(
            documents=documents,
            max_tokens=max_tokens or self.config.max_context_tokens,
        )
        context = self.context_manager.join_context(context_window)
        logger.info(
            f"上下文构建完成，token 数: {context_window.total_tokens}，"
            f"截断: {context_window.truncated}"
        )

        # 5. 生成引用
        citation_context = self.citation_manager.format_context_with_citations(
            documents=documents,
            scores=scores,
        )
        logger.info(f"引用生成完成，数量: {len(citation_context.citations)}")

        return RAGResult(
            context=context,
            context_window=context_window,
            citation_context=citation_context,
            search_results=reranked_results,
            citations=citation_context.citations,
        )

    def retrieve_and_format(
        self,
        query: str,
        where: Optional[Dict[str, Any]] = None,
    ) -> str:
        """检索并返回格式化的上下文（可直接用于 LLM 提示）"""
        result = self.retrieve(query, where)
        return result.citation_context.text_with_citations

    def rebuild_index(self) -> None:
        """重建索引"""
        self.hybrid_retriever.rebuild_index()
        self.citation_manager.clear()
        logger.info("索引重建完成")
