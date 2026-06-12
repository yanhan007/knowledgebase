"""上下文窗口管理器

功能：
- Token 计数（支持自定义计数器）
- 上下文截断与拼接
- 可配置重叠窗口
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class ContextWindow:
    """上下文窗口"""

    texts: List[str]
    total_tokens: int
    truncated: bool = False
    removed_count: int = 0


@dataclass
class ContextConfig:
    """上下文配置"""

    max_tokens: int = 4000  # 最大 token 数
    separator: str = "\n\n---\n\n"  # 文档分隔符
    preserve_head: int = 1  # 保留头部文档数
    preserve_tail: int = 1  # 保留尾部文档数
    context_overlap_chars: int = 100  # 重叠窗口字符数
    context_overlap_sentences: int = 1  # 重叠窗口句子数


def default_token_counter(text: str) -> int:
    """默认 token 估算

    中文字符约 1.5 token，英文单词约 1 token，数字约 1 token
    """
    cn_chars = len(re.findall(r"[一-鿿]", text))
    en_words = len(re.findall(r"[a-zA-Z]+", text))
    numbers = len(re.findall(r"\d+", text))
    return int(cn_chars * 1.5 + en_words + numbers)


class ContextManager:
    """上下文窗口管理器"""

    def __init__(
        self,
        config: Optional[ContextConfig] = None,
        token_counter: Optional[Callable[[str], int]] = None,
    ):
        self.config = config or ContextConfig()
        self._token_counter = token_counter or default_token_counter

    def estimate_tokens(self, text: str) -> int:
        """估算 token 数量（使用可配置的计数器）"""
        return self._token_counter(text)

    def build_context(
        self,
        documents: List[Document],
        max_tokens: Optional[int] = None,
    ) -> ContextWindow:
        """构建上下文窗口"""
        max_tokens = max_tokens or self.config.max_tokens

        if not documents:
            return ContextWindow(texts=[], total_tokens=0)

        doc_tokens = [
            (doc, self.estimate_tokens(doc.page_content)) for doc in documents
        ]

        sep_tokens = self.estimate_tokens(self.config.separator)

        selected_texts: List[str] = []
        total_tokens = 0
        truncated = False
        removed_count = 0

        # 先处理必须保留的头部文档
        head_docs = doc_tokens[: self.config.preserve_head]
        remaining_docs = doc_tokens[self.config.preserve_head :]

        for doc, tokens in head_docs:
            if total_tokens + tokens <= max_tokens:
                selected_texts.append(doc.page_content)
                total_tokens += tokens
                if selected_texts:
                    total_tokens += sep_tokens

        # 处理尾部必须保留的文档
        tail_docs = (
            remaining_docs[-self.config.preserve_tail :]
            if self.config.preserve_tail > 0
            else []
        )
        middle_docs = (
            remaining_docs[: -self.config.preserve_tail]
            if self.config.preserve_tail > 0
            else remaining_docs
        )

        for doc, tokens in middle_docs:
            needed = tokens + (sep_tokens if selected_texts else 0)
            if total_tokens + needed <= max_tokens:
                selected_texts.append(doc.page_content)
                total_tokens += needed
            else:
                truncated = True
                removed_count += 1

        for doc, tokens in tail_docs:
            needed = tokens + (sep_tokens if selected_texts else 0)
            if total_tokens + needed <= max_tokens:
                selected_texts.append(doc.page_content)
                total_tokens += needed
            else:
                truncated = True
                removed_count += 1

        if truncated:
            logger.warning(f"上下文被截断，移除了 {removed_count} 个文档")

        return ContextWindow(
            texts=selected_texts,
            total_tokens=total_tokens,
            truncated=truncated,
            removed_count=removed_count,
        )

    def join_context(self, context_window: ContextWindow) -> str:
        """将上下文窗口拼接为字符串"""
        return self.config.separator.join(context_window.texts)

    def truncate_text(self, text: str, max_tokens: int) -> str:
        """截断单个文本到指定 token 数"""
        estimated = self.estimate_tokens(text)
        if estimated <= max_tokens:
            return text

        ratio = max_tokens / estimated
        target_length = int(len(text) * ratio * 0.9)

        truncated = text[:target_length]
        last_period = max(
            truncated.rfind("。"),
            truncated.rfind("！"),
            truncated.rfind("？"),
            truncated.rfind("."),
            truncated.rfind("!"),
            truncated.rfind("?"),
        )
        if last_period > target_length * 0.5:
            truncated = truncated[: last_period + 1]

        return truncated + "..."

    def split_into_chunks(
        self,
        text: str,
        chunk_tokens: int = 500,
        overlap_tokens: int = 50,
    ) -> List[str]:
        """将长文本分割成重叠的块"""
        estimated_total = self.estimate_tokens(text)
        if estimated_total <= chunk_tokens:
            return [text]

        chunks = []
        chars_per_token = len(text) / estimated_total
        chunk_size = int(chunk_tokens * chars_per_token)
        overlap_size = int(overlap_tokens * chars_per_token)

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]

            if end < len(text):
                last_period = max(
                    chunk.rfind("。"),
                    chunk.rfind("！"),
                    chunk.rfind("？"),
                    chunk.rfind("."),
                    chunk.rfind("!"),
                    chunk.rfind("?"),
                )
                if last_period > chunk_size * 0.5:
                    chunk = chunk[: last_period + 1]
                    end = start + last_period + 1

            chunks.append(chunk)
            start = end - overlap_size

        return chunks

    def build_context_with_overlap(
        self,
        documents: List[Document],
        max_tokens: Optional[int] = None,
    ) -> ContextWindow:
        """构建带重叠的上下文窗口

        当上下文被截断时，使用配置的重叠参数在截断边界添加
        邻近文本片段，保持语义连贯性。
        """
        window = self.build_context(documents, max_tokens)

        if not window.truncated or not documents:
            return window

        # 找到最后一个被包含的文档
        last_included_text = window.texts[-1] if window.texts else ""
        last_included_idx = -1
        for i, doc in enumerate(documents):
            if doc.page_content == last_included_text:
                last_included_idx = i
                break

        if last_included_idx < 0:
            return window

        # 从下一个文档中截取重叠片段
        if last_included_idx + 1 < len(documents):
            next_doc = documents[last_included_idx + 1].page_content
            overlap_chars = self.config.context_overlap_chars
            overlap_text = next_doc[:overlap_chars]

            # 尝试在句子边界截断
            for sep in ["。", "！", "？", ".", "!", "?"]:
                last_sep = overlap_text.rfind(sep)
                if last_sep > overlap_chars * 0.5:
                    overlap_text = overlap_text[: last_sep + 1]
                    break

            if overlap_text:
                window.texts.append(f"[...] {overlap_text}")
                window.total_tokens += self.estimate_tokens(overlap_text)

        return window
