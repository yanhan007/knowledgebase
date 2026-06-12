"""智能文本分割器

特性：
- 按句子边界分割，保留语义完整性
- 支持中英文混合文本
- 可配置块大小和重叠
- 避免在单词中间分割
"""

import re
from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document


@dataclass
class SplitterConfig:
    """分割器配置"""

    chunk_size: int = 500
    chunk_overlap: int = 50
    min_chunk_size: int = 50


class SentenceSplitter:
    """句子分割器，处理中英文混合文本"""

    # 中文句子结束符
    CN_SENTENCE_END = re.compile(r"([。！？；\n]+)")
    # 英文句子结束符
    EN_SENTENCE_END = re.compile(r"([.!?;]\s+|\n+)")
    # 中文标点
    CN_PUNCTUATION = set("。！？；，、：""''（）【】《》")
    # 英文单词边界
    EN_WORD_BOUNDARY = re.compile(r"\b")

    def split_sentences(self, text: str) -> List[str]:
        """将文本分割为句子列表"""
        if not text.strip():
            return []

        # 先按段落分割
        paragraphs = text.split("\n")
        sentences = []

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 检测是否包含中文
            if self._contains_chinese(para):
                sentences.extend(self._split_cn_sentences(para))
            else:
                sentences.extend(self._split_en_sentences(para))

        return [s.strip() for s in sentences if s.strip()]

    def _contains_chinese(self, text: str) -> bool:
        """检测文本是否包含中文"""
        return bool(re.search(r"[一-鿿]", text))

    def _split_cn_sentences(self, text: str) -> List[str]:
        """分割中文句子"""
        parts = self.CN_SENTENCE_END.split(text)
        sentences = []
        current = ""

        for part in parts:
            current += part
            if self.CN_SENTENCE_END.match(part):
                if current.strip():
                    sentences.append(current)
                current = ""

        if current.strip():
            sentences.append(current)

        return sentences

    def _split_en_sentences(self, text: str) -> List[str]:
        """分割英文句子"""
        parts = self.EN_SENTENCE_END.split(text)
        sentences = []
        current = ""

        for part in parts:
            current += part
            if self.EN_SENTENCE_END.match(part):
                sentences.append(current)
                current = ""

        if current.strip():
            sentences.append(current)

        return sentences


class SmartTextSplitter:
    """智能文本分割器"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 50,
    ):
        self.config = SplitterConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )
        self.sentence_splitter = SentenceSplitter()

    def split_text(self, text: str) -> List[str]:
        """将文本分割为语义完整的块"""
        if not text.strip():
            return []

        sentences = self.sentence_splitter.split_sentences(text)
        if not sentences:
            return []

        return self._merge_sentences(sentences)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档列表"""
        split_docs = []

        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                metadata = {**doc.metadata, "chunk_index": i}
                split_docs.append(Document(page_content=chunk, metadata=metadata))

        return split_docs

    def _merge_sentences(self, sentences: List[str]) -> List[str]:
        """将句子合并为满足大小要求的块"""
        chunks = []
        current_chunk: List[str] = []
        current_length = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # 单个句子超过块大小，需要进一步分割
            if sentence_len > self.config.chunk_size:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                long_chunks = self._split_long_sentence(sentence)
                chunks.extend(long_chunks)
                # 用最后一块的重叠部分作为下一个块的起始
                if long_chunks:
                    overlap_text = self._get_overlap_text(long_chunks[-1])
                    if overlap_text:
                        current_chunk = [overlap_text]
                        current_length = len(overlap_text)
                continue

            # 当前块加上新句子会超过大小限制
            if current_length + sentence_len > self.config.chunk_size:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    # 计算重叠部分
                    overlap = self._get_overlap(current_chunk)
                    current_chunk = overlap
                    current_length = sum(len(s) for s in overlap)

            current_chunk.append(sentence)
            current_length += sentence_len

        # 处理最后一个块
        if current_chunk:
            last_chunk = "".join(current_chunk)
            if len(last_chunk) >= self.config.min_chunk_size:
                chunks.append(last_chunk)
            elif chunks:
                # 太短的块合并到前一个
                chunks[-1] += last_chunk

        return chunks

    def _split_long_sentence(self, sentence: str) -> List[str]:
        """分割过长的句子，按词边界分割"""
        chunks = []
        words = self._split_by_words(sentence)
        current_chunk: List[str] = []
        current_length = 0

        for word in words:
            word_len = len(word)

            if current_length + word_len > self.config.chunk_size:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    # 计算重叠
                    overlap = self._get_overlap_text("".join(current_chunk))
                    current_chunk = [overlap] if overlap else []
                    current_length = len(overlap)
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(word)
            current_length += word_len

        if current_chunk:
            chunks.append("".join(current_chunk))

        return chunks

    # 中文字符范围（预编译）
    _CN_CHAR_RE = re.compile(r"[一-鿿]")

    def _split_by_words(self, text: str) -> List[str]:
        """按词边界分割文本（中英文混合）"""
        tokens = []
        current = []

        for char in text:
            # 中文字符单独成 token
            if self._CN_CHAR_RE.match(char):
                if current:
                    tokens.append("".join(current))
                    current = []
                tokens.append(char)
            # 英文数字连续拼接
            elif char.isalnum():
                current.append(char)
            # 标点等分隔符：断开当前词
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
                tokens.append(char)

        if current:
            tokens.append("".join(current))

        return tokens

    def _get_overlap(self, chunk_sentences: List[str]) -> List[str]:
        """获取重叠部分的句子"""
        if not chunk_sentences:
            return []

        overlap_length = 0
        overlap_sentences: List[str] = []

        for sentence in reversed(chunk_sentences):
            if overlap_length + len(sentence) > self.config.chunk_overlap:
                break
            overlap_sentences.insert(0, sentence)
            overlap_length += len(sentence)

        return overlap_sentences

    def _get_overlap_text(self, text: str) -> str:
        """从文本末尾获取重叠部分，保证不在单词中间截断"""
        if len(text) <= self.config.chunk_overlap:
            return text

        # 从 chunk_overlap 位置向前找安全截断点
        candidate = text[-self.config.chunk_overlap :]

        # 优先在句子边界截断
        match = re.search(r"[。！？.!?\s]", candidate)
        if match:
            return candidate[match.end() :]

        # 其次在中文字符后截断（中文字符本身就是完整语义单元）
        cn_match = re.search(r"[一-鿿]", candidate)
        if cn_match:
            return candidate[cn_match.end() :]

        # 英文场景：回退到空格位置，避免切断单词
        space_idx = candidate.find(" ")
        if space_idx != -1:
            return candidate[space_idx + 1 :]

        # 无安全截断点，返回完整重叠（极端情况）
        return candidate
