"""文本向量化模块

使用 shibing624/text2vec-base-chinese 嵌入模型
"""

import logging
from typing import List

from langchain_core.documents import Document

from config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class TextEmbedder:
    """文本向量化处理"""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model_name = model_name
        self._embeddings = None

    @property
    def embeddings(self):
        """延迟加载 embedding 模型"""
        if self._embeddings is None:
            from langchain_huggingface import HuggingFaceEmbeddings

            logger.info(f"加载嵌入模型: {self.model_name}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("嵌入模型加载完成")
        return self._embeddings

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档分割成小块"""
        from src.text_splitter import SmartTextSplitter

        splitter = SmartTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        return splitter.split_documents(documents)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将文本列表转换为向量"""
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """将查询文本转换为向量"""
        return self.embeddings.embed_query(text)
