"""问答系统主流程引擎

整合所有模块，提供完整的问答流程：
用户输入 -> 查询分析 -> 检索 -> 生成 -> 输出

功能：
- 结构化日志记录
- 性能监控（耗时统计）
- 异常处理链
"""

import logging
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple
from contextlib import contextmanager

from src.document_loader import DocumentLoader
from src.embeddings import TextEmbedder
from src.vector_store import VectorStore
from src.rag_retriever import RAGPipeline, RAGConfig, RAGResult
from src.llm import create_llm, BaseLLM, StreamChunk, PromptManager

logger = logging.getLogger(__name__)


# ==================== 日志配置 ====================


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """配置结构化日志"""
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )


# ==================== 性能监控 ====================


@dataclass
class PerformanceMetrics:
    """性能指标"""

    query_analysis_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    context_build_ms: float = 0.0
    llm_first_token_ms: float = 0.0
    llm_total_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "query_analysis_ms": round(self.query_analysis_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "rerank_ms": round(self.rerank_ms, 2),
            "context_build_ms": round(self.context_build_ms, 2),
            "llm_first_token_ms": round(self.llm_first_token_ms, 2),
            "llm_total_ms": round(self.llm_total_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }

    def summary(self) -> str:
        return (
            f"性能统计: "
            f"查询分析={self.query_analysis_ms:.0f}ms, "
            f"检索={self.retrieval_ms:.0f}ms, "
            f"重排={self.rerank_ms:.0f}ms, "
            f"上下文={self.context_build_ms:.0f}ms, "
            f"LLM首token={self.llm_first_token_ms:.0f}ms, "
            f"LLM总耗时={self.llm_total_ms:.0f}ms, "
            f"总耗时={self.total_ms:.0f}ms"
        )


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self):
        self._metrics = PerformanceMetrics()
        self._start_time: float = 0.0
        self._stage_start: float = 0.0

    def start(self):
        """开始计时"""
        self._metrics = PerformanceMetrics()
        self._start_time = time.perf_counter()

    def start_stage(self, stage: str):
        """开始阶段计时"""
        self._stage_start = time.perf_counter()
        logger.debug(f"开始阶段: {stage}")

    def end_stage(self, stage: str):
        """结束阶段计时"""
        elapsed = (time.perf_counter() - self._stage_start) * 1000
        logger.debug(f"阶段 {stage} 耗时: {elapsed:.2f}ms")
        return elapsed

    def finish(self) -> PerformanceMetrics:
        """完成计时"""
        self._metrics.total_ms = (time.perf_counter() - self._start_time) * 1000
        return self._metrics

    @property
    def metrics(self) -> PerformanceMetrics:
        return self._metrics


@contextmanager
def measure_stage(monitor: PerformanceMonitor, stage: str):
    """阶段计时上下文管理器"""
    monitor.start_stage(stage)
    try:
        yield
    finally:
        elapsed = monitor.end_stage(stage)
        # 将耗时写入对应字段
        if stage == "query_analysis":
            monitor.metrics.query_analysis_ms = elapsed
        elif stage == "retrieval":
            monitor.metrics.retrieval_ms = elapsed
        elif stage == "rerank":
            monitor.metrics.rerank_ms = elapsed
        elif stage == "context_build":
            monitor.metrics.context_build_ms = elapsed
        elif stage == "llm_generate":
            monitor.metrics.llm_total_ms = elapsed


# ==================== 异常处理 ====================


class ErrorCode(str, Enum):
    """错误码"""

    QUERY_EMPTY = "QUERY_EMPTY"
    QUERY_TOO_LONG = "QUERY_TOO_LONG"
    RETRIEVAL_FAILED = "RETRIEVAL_FAILED"
    NO_RESULTS = "NO_RESULTS"
    LLM_ERROR = "LLM_ERROR"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_FALLBACK = "LLM_FALLBACK"  # LLM 失败，返回检索片段兜底
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass
class QAError:
    """问答错误"""

    code: ErrorCode
    message: str
    details: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class QueryAnalysis:
    """查询分析结果"""

    original_query: str
    cleaned_query: str
    intent: str = "qa"  # qa / summarize / clarify
    keywords: List[str] = field(default_factory=list)
    is_valid: bool = True
    error: Optional[QAError] = None


@dataclass
class QAResult:
    """问答结果"""

    answer: str
    query_analysis: QueryAnalysis
    rag_result: Optional[RAGResult] = None
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    error: Optional[QAError] = None

    @property
    def success(self) -> bool:
        return self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "answer": self.answer,
            "query": self.query_analysis.original_query,
            "intent": self.query_analysis.intent,
            "citations": [
                {
                    "index": c.index,
                    "source": c.source,
                    "score": c.relevance_score,
                }
                for c in (self.rag_result.citations if self.rag_result else [])
            ],
            "metrics": self.metrics.to_dict(),
            "error": self.error.to_dict() if self.error else None,
        }


# ==================== 查询分析器 ====================


class QueryAnalyzer:
    """查询分析器"""

    MAX_QUERY_LENGTH = 1000

    def analyze(self, query: str) -> QueryAnalysis:
        """分析用户查询"""
        # 清理查询
        cleaned = query.strip()

        # 空查询检查
        if not cleaned:
            return QueryAnalysis(
                original_query=query,
                cleaned_query="",
                is_valid=False,
                error=QAError(
                    code=ErrorCode.QUERY_EMPTY,
                    message="查询不能为空",
                ),
            )

        # 长度检查
        if len(cleaned) > self.MAX_QUERY_LENGTH:
            return QueryAnalysis(
                original_query=query,
                cleaned_query=cleaned[:self.MAX_QUERY_LENGTH],
                is_valid=False,
                error=QAError(
                    code=ErrorCode.QUERY_TOO_LONG,
                    message=f"查询超过最大长度限制 ({self.MAX_QUERY_LENGTH} 字符)",
                ),
            )

        # 提取关键词（简单实现）
        keywords = self._extract_keywords(cleaned)

        # 判断意图
        intent = self._detect_intent(cleaned)

        return QueryAnalysis(
            original_query=query,
            cleaned_query=cleaned,
            intent=intent,
            keywords=keywords,
            is_valid=True,
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        import re
        # 移除标点
        text = re.sub(r"[^\w\s]", " ", query)
        # 英文单词
        en_words = re.findall(r"[a-zA-Z]{2,}", text)
        # 中文词（简单按 2-4 字切分）
        cn_chars = re.findall(r"[一-鿿]+", text)
        cn_words = []
        for seg in cn_chars:
            for i in range(len(seg)):
                for length in [2, 3, 4]:
                    if i + length <= len(seg):
                        cn_words.append(seg[i:i+length])

        return list(set(en_words + cn_words))[:10]

    def _detect_intent(self, query: str) -> str:
        """检测查询意图"""
        summarize_keywords = ["总结", "概括", "摘要", "summarize", "summary"]
        clarify_keywords = ["什么是", "解释", "定义", "what is", "explain", "define"]

        query_lower = query.lower()

        for kw in summarize_keywords:
            if kw in query_lower:
                return "summarize"

        for kw in clarify_keywords:
            if kw in query_lower:
                return "clarify"

        return "qa"


# ==================== 问答引擎 ====================


class QAEngine:
    """问答系统主引擎

    完整流程：
    1. 查询分析 -> 2. 混合检索 -> 3. 结果重排 -> 4. 上下文构建 -> 5. LLM 生成 -> 6. 引用附加
    """

    def __init__(
        self,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        rag_config: Optional[RAGConfig] = None,
    ):
        # 初始化各模块
        self.document_loader = DocumentLoader()
        self.text_embedder = TextEmbedder()
        self.vector_store = VectorStore()
        self.rag_pipeline = RAGPipeline(
            vector_store=self.vector_store,
            config=rag_config or RAGConfig(),
        )
        self.llm = create_llm(provider=llm_provider, model=llm_model)
        self.prompt_manager = PromptManager()
        self.query_analyzer = QueryAnalyzer()
        self.performance_monitor = PerformanceMonitor()

        logger.info(
            f"QAEngine 初始化完成: "
            f"provider={llm_provider or 'default'}, "
            f"model={llm_model or 'default'}"
        )

    # ==================== 文档导入 ====================

    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """导入单个文档"""
        logger.info(f"开始导入文档: {file_path}")
        start = time.perf_counter()

        try:
            documents = self.document_loader.load(file_path)
            split_docs = self.text_embedder.split_documents(documents)
            self.vector_store.add_documents(split_docs)
            self.rag_pipeline.rebuild_index()

            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"文档导入完成: {file_path}, {len(split_docs)} 块, {elapsed:.0f}ms")

            return {
                "success": True,
                "file": file_path,
                "chunks": len(split_docs),
                "elapsed_ms": round(elapsed, 2),
            }

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(f"文档导入失败: {file_path}, {e}")
            return {
                "success": False,
                "file": file_path,
                "error": str(e),
                "elapsed_ms": round(elapsed, 2),
            }

    def ingest_directory(self, dir_path: str) -> Dict[str, Any]:
        """导入目录下所有文档"""
        logger.info(f"开始导入目录: {dir_path}")
        start = time.perf_counter()

        try:
            documents, load_result = self.document_loader.load_directory(dir_path)
            if not documents:
                return {"success": True, "message": "目录中无文档", "chunks": 0}

            split_docs = self.text_embedder.split_documents(documents)
            self.vector_store.add_documents(split_docs)
            self.rag_pipeline.rebuild_index()

            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"目录导入完成: {dir_path}, {len(split_docs)} 块, {elapsed:.0f}ms")

            return {
                "success": True,
                "dir": dir_path,
                "chunks": len(split_docs),
                "load_summary": load_result.summary,
                "elapsed_ms": round(elapsed, 2),
            }

        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(f"目录导入失败: {dir_path}, {e}")
            return {
                "success": False,
                "dir": dir_path,
                "error": str(e),
                "elapsed_ms": round(elapsed, 2),
            }

    # ==================== 问答流程 ====================

    def _format_fallback_answer(
        self, rag_result: RAGResult, error: Exception
    ) -> str:
        """LLM 失败时，将检索到的片段格式化为兜底答案"""
        fallback_parts = [
            "⚠️ **LLM 生成失败，以下为检索到的相关内容片段：**\n",
            f"*错误原因: {str(error)[:100]}*\n",
        ]

        for i, citation in enumerate(rag_result.citations[:5], 1):
            source_info = f"[{citation.source}]"
            if citation.page is not None:
                source_info += f" 第{citation.page + 1}页"
            fallback_parts.append(f"**{i}. {source_info}**")

            # 使用 content_preview 属性
            preview = citation.content_preview or ""
            snippet = preview[:200]
            if len(preview) > 200:
                snippet += "..."
            fallback_parts.append(f"> {snippet}\n")

        fallback_parts.append(
            "*以上内容为知识库检索结果，请参考原文获取完整信息。*"
        )
        return "\n".join(fallback_parts)

    def ask(self, question: str) -> QAResult:
        """同步问答

        流程：查询分析 -> 检索 -> 重排 -> 上下文 -> 生成 -> 引用
        """
        self.performance_monitor.start()
        logger.info(f"收到问题: {question[:50]}...")

        # 1. 查询分析
        with measure_stage(self.performance_monitor, "query_analysis"):
            analysis = self.query_analyzer.analyze(question)

        if not analysis.is_valid:
            logger.warning(f"查询分析失败: {analysis.error.message}")
            return QAResult(
                answer=analysis.error.message,
                query_analysis=analysis,
                metrics=self.performance_monitor.finish(),
                error=analysis.error,
            )

        # 2-4. RAG 检索（混合检索 + 重排 + 上下文构建）
        try:
            with measure_stage(self.performance_monitor, "retrieval"):
                rag_result = self.rag_pipeline.retrieve(analysis.cleaned_query)

            if not rag_result.citations:
                logger.info("未找到相关文档")
                return QAResult(
                    answer="知识库中暂无相关内容",
                    query_analysis=analysis,
                    rag_result=rag_result,
                    metrics=self.performance_monitor.finish(),
                    error=QAError(
                        code=ErrorCode.NO_RESULTS,
                        message="知识库中暂无相关内容",
                    ),
                )

        except Exception as e:
            logger.error(f"检索失败: {e}")
            return QAResult(
                answer="检索过程中发生错误",
                query_analysis=analysis,
                metrics=self.performance_monitor.finish(),
                error=QAError(
                    code=ErrorCode.RETRIEVAL_FAILED,
                    message="检索失败",
                    details=str(e),
                ),
            )

        # 5. LLM 生成
        try:
            with measure_stage(self.performance_monitor, "llm_generate"):
                prompt = self.prompt_manager.render(
                    "rag_qa",
                    context=rag_result.context,
                    question=analysis.cleaned_query,
                )
                response = self.llm.chat(prompt)
                answer = response.content

            # 6. 添加引用
            answer_with_citations = (
                self.rag_pipeline.citation_manager.format_response_with_citations(
                    answer=answer,
                    citations=rag_result.citations,
                )
            )

            metrics = self.performance_monitor.finish()
            logger.info(f"问答完成: {metrics.summary()}")

            return QAResult(
                answer=answer_with_citations,
                query_analysis=analysis,
                rag_result=rag_result,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"LLM 生成失败，启用兜底策略: {e}")

            # 兜底：返回检索到的片段
            fallback_answer = self._format_fallback_answer(rag_result, e)

            # 添加引用
            answer_with_citations = (
                self.rag_pipeline.citation_manager.format_response_with_citations(
                    answer=fallback_answer,
                    citations=rag_result.citations,
                )
            )

            return QAResult(
                answer=answer_with_citations,
                query_analysis=analysis,
                rag_result=rag_result,
                metrics=self.performance_monitor.finish(),
                error=QAError(
                    code=ErrorCode.LLM_FALLBACK,
                    message="LLM 生成失败，已返回检索片段",
                    details=str(e),
                ),
            )

    def ask_stream(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Generator[Tuple[str, QAResult], None, None]:
        """流式问答

        Yields:
            (chunk_content, partial_result)
        """
        self.performance_monitor.start()
        logger.info(f"收到流式问题: {question[:50]}...")

        # 1. 查询分析
        with measure_stage(self.performance_monitor, "query_analysis"):
            analysis = self.query_analyzer.analyze(question)

        if not analysis.is_valid:
            yield analysis.error.message, QAResult(
                answer=analysis.error.message,
                query_analysis=analysis,
                metrics=self.performance_monitor.finish(),
                error=analysis.error,
            )
            return

        # 2-4. RAG 检索
        try:
            with measure_stage(self.performance_monitor, "retrieval"):
                rag_result = self.rag_pipeline.retrieve(analysis.cleaned_query)

            if not rag_result.citations:
                yield "知识库中暂无相关内容", QAResult(
                    answer="知识库中暂无相关内容",
                    query_analysis=analysis,
                    rag_result=rag_result,
                    metrics=self.performance_monitor.finish(),
                    error=QAError(
                        code=ErrorCode.NO_RESULTS,
                        message="知识库中暂无相关内容",
                    ),
                )
                return

        except Exception as e:
            logger.error(f"检索失败: {e}")
            yield "检索过程中发生错误", QAResult(
                answer="检索过程中发生错误",
                query_analysis=analysis,
                metrics=self.performance_monitor.finish(),
                error=QAError(
                    code=ErrorCode.RETRIEVAL_FAILED,
                    message="检索失败",
                    details=str(e),
                ),
            )
            return

        # 5. LLM 流式生成
        try:
            # 构建提示词
            history_str = ""
            if history:
                for msg in history:
                    role = "用户" if msg.get("role") == "user" else "助手"
                    history_str += f"{role}: {msg.get('content', '')}\n"

            if history_str:
                prompt = self.prompt_manager.render(
                    "rag_qa_with_history",
                    history=history_str,
                    context=rag_result.context,
                    question=analysis.cleaned_query,
                )
            else:
                prompt = self.prompt_manager.render(
                    "rag_qa",
                    context=rag_result.context,
                    question=analysis.cleaned_query,
                )

            # 流式生成
            full_answer = ""
            first_token = True
            for chunk in self.llm.stream_chat(prompt):
                if chunk.content:
                    if first_token:
                        self.performance_monitor.metrics.llm_first_token_ms = (
                            self.performance_monitor.end_stage("llm_first_token")
                        )
                        first_token = False
                    full_answer += chunk.content
                    yield chunk.content, QAResult(
                        answer=full_answer,
                        query_analysis=analysis,
                        rag_result=rag_result,
                        metrics=self.performance_monitor.metrics,
                    )

            # 添加引用
            citations_text = "\n\n---\n**参考来源：**\n"
            for citation in rag_result.citations:
                citations_text += f"[{citation.index}] {citation.source}"
                if citation.page is not None:
                    citations_text += f" (第{citation.page + 1}页)"
                citations_text += f" | 相关度: {citation.relevance_score:.1%}\n"

            yield citations_text, QAResult(
                answer=full_answer + citations_text,
                query_analysis=analysis,
                rag_result=rag_result,
                metrics=self.performance_monitor.finish(),
            )

            logger.info(f"流式问答完成: {self.performance_monitor.metrics.summary()}")

        except Exception as e:
            logger.error(f"LLM 流式生成失败，启用兜底策略: {e}")

            # 兜底：返回检索到的片段
            fallback_answer = self._format_fallback_answer(rag_result, e)

            # 添加引用
            citations_text = "\n\n---\n**参考来源：**\n"
            for citation in rag_result.citations:
                citations_text += f"[{citation.index}] {citation.source}"
                if citation.page is not None:
                    citations_text += f" (第{citation.page + 1}页)"
                citations_text += f" | 相关度: {citation.relevance_score:.1%}\n"

            full_answer = fallback_answer + citations_text

            yield full_answer, QAResult(
                answer=full_answer,
                query_analysis=analysis,
                rag_result=rag_result,
                metrics=self.performance_monitor.finish(),
                error=QAError(
                    code=ErrorCode.LLM_FALLBACK,
                    message="LLM 生成失败，已返回检索片段",
                    details=str(e),
                ),
            )

    # ==================== 工具方法 ====================

    def get_stats(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "vector_store_count": self.vector_store.count(),
            "llm_provider": self.llm.config.provider.value,
            "llm_model": self.llm.config.model,
            "templates": self.prompt_manager.list_templates(),
        }

    def health_check(self) -> Dict[str, bool]:
        """健康检查"""
        return {
            "llm": self.llm.health_check(),
            "vector_store": self.vector_store.count() >= 0,
        }
