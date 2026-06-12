"""引用溯源模块

功能：
- 生成文档引用标记
- 追踪引用来源
- 格式化引用输出
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """引用信息"""

    id: str  # 唯一标识
    index: int  # 引用序号
    source: str  # 来源文件
    page: Optional[int] = None  # 页码
    chunk_index: Optional[int] = None  # 文本块索引
    content_preview: str = ""  # 内容预览
    relevance_score: float = 0.0  # 相关性分数
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """转换为 Markdown 格式"""
        parts = [f"[{self.index}] {self.source}"]
        if self.page is not None:
            parts.append(f"第 {self.page + 1} 页")
        if self.chunk_index is not None:
            parts.append(f"段落 {self.chunk_index + 1}")
        return " - ".join(parts)

    def to_footnote(self) -> str:
        """转换为脚注格式"""
        return f"[^{self.index}]: {self.source}" + (
            f", 第 {self.page + 1} 页" if self.page else ""
        )


@dataclass
class CitationContext:
    """引用上下文"""

    citations: List[Citation]
    text_with_citations: str
    sources_summary: str


class CitationManager:
    """引用管理器"""

    def __init__(self, preview_length: int = 100):
        self.preview_length = preview_length
        self._citation_counter = 0
        self._citations: Dict[str, Citation] = {}

    def _generate_id(self, document: Document) -> str:
        """生成文档唯一 ID（基于内容哈希，保证稳定性）

        使用文档路径 + 完整内容的 SHA256 哈希，
        重新分割后只要内容不变，ID 不变。
        """
        source = document.metadata.get("source", "unknown")
        content = document.page_content
        key = f"{source}||{content}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _extract_preview(self, content: str) -> str:
        """提取内容预览"""
        # 移除多余空白
        content = " ".join(content.split())
        if len(content) <= self.preview_length:
            return content
        return content[: self.preview_length] + "..."

    def create_citation(
        self,
        document: Document,
        score: float = 0.0,
    ) -> Citation:
        """创建引用"""
        doc_id = self._generate_id(document)

        # 检查是否已存在
        if doc_id in self._citations:
            return self._citations[doc_id]

        self._citation_counter += 1

        citation = Citation(
            id=doc_id,
            index=self._citation_counter,
            source=document.metadata.get("source", "未知来源"),
            page=document.metadata.get("page"),
            chunk_index=document.metadata.get("chunk_index"),
            content_preview=self._extract_preview(document.page_content),
            relevance_score=score,
            metadata=document.metadata,
        )

        self._citations[doc_id] = citation
        return citation

    def create_citations(
        self,
        documents: List[Document],
        scores: Optional[List[float]] = None,
    ) -> List[Citation]:
        """批量创建引用"""
        if scores is None:
            scores = [0.0] * len(documents)

        citations = []
        for doc, score in zip(documents, scores):
            citation = self.create_citation(doc, score)
            citations.append(citation)

        return citations

    def format_context_with_citations(
        self,
        documents: List[Document],
        scores: Optional[List[float]] = None,
    ) -> CitationContext:
        """格式化带引用标记的上下文"""
        citations = self.create_citations(documents, scores)

        # 构建带引用标记的文本
        text_parts = []
        for i, (doc, citation) in enumerate(zip(documents, citations)):
            text_parts.append(f"[{citation.index}] {doc.page_content}")

        text_with_citations = "\n\n".join(text_parts)

        # 生成来源摘要
        sources_summary = self._generate_sources_summary(citations)

        return CitationContext(
            citations=citations,
            text_with_citations=text_with_citations,
            sources_summary=sources_summary,
        )

    def _generate_sources_summary(self, citations: List[Citation]) -> str:
        """生成来源摘要"""
        if not citations:
            return "无引用来源"

        lines = ["**引用来源：**"]
        for citation in citations:
            lines.append(citation.to_markdown())

        return "\n".join(lines)

    def get_citation_by_id(self, doc_id: str) -> Optional[Citation]:
        """根据 ID 获取引用"""
        return self._citations.get(doc_id)

    def get_all_citations(self) -> List[Citation]:
        """获取所有引用"""
        return list(self._citations.values())

    def clear(self) -> None:
        """清空引用"""
        self._citation_counter = 0
        self._citations.clear()

    def format_response_with_citations(
        self,
        answer: str,
        citations: List[Citation],
    ) -> str:
        """在回答中添加引用标记"""
        if not citations:
            return answer

        # 在回答末尾添加引用列表
        citation_lines = ["\n\n---\n**参考来源：**"]
        for citation in citations:
            citation_lines.append(
                f"[{citation.index}] {citation.source}"
                + (f" (第{citation.page + 1}页)" if citation.page else "")
                + f" | 相关度: {citation.relevance_score:.1%}"
            )

        return answer + "\n".join(citation_lines)
