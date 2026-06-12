"""RAG 检索模块

功能：
- 混合检索（向量 + BM25 关键词）+ 索引持久化 + 自动同步
- 结果重排（交叉编码器 / MMR / 元数据加权）+ 可配置启用/禁用
- 上下文窗口管理 + 自定义 token 计数器 + 可配置重叠窗口
- 引用溯源 + 基于内容哈希的稳定 ID
"""

from .hybrid_retriever import HybridRetriever, HybridSearchResult, FusionMethod
from .reranker import Reranker, RerankConfig, RerankStrategy
from .context_manager import (
    ContextManager,
    ContextConfig,
    ContextWindow,
    default_token_counter,
)
from .citation import CitationManager, Citation, CitationContext
from .rag_pipeline import RAGPipeline, RAGConfig, RAGResult

__all__ = [
    # 混合检索
    "HybridRetriever",
    "HybridSearchResult",
    "FusionMethod",
    # 重排
    "Reranker",
    "RerankConfig",
    "RerankStrategy",
    # 上下文管理
    "ContextManager",
    "ContextConfig",
    "ContextWindow",
    "default_token_counter",
    # 引用溯源
    "CitationManager",
    "Citation",
    "CitationContext",
    # RAG 管线
    "RAGPipeline",
    "RAGConfig",
    "RAGResult",
]
