"""问答链

整合文档加载、向量化、RAG 检索和 LLM 生成
"""

import logging
from typing import Generator, List, Optional, Tuple

from langchain_core.documents import Document

from src.document_loader import DocumentLoader
from src.embeddings import TextEmbedder
from src.vector_store import VectorStore, SearchResult
from src.rag_retriever import RAGPipeline, RAGConfig, RAGResult
from src.llm import create_llm, BaseLLM, StreamChunk, PromptManager

logger = logging.getLogger(__name__)


class QAChain:
    """知识库问答链"""

    def __init__(
        self,
        rag_config: Optional[RAGConfig] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        self.loader = DocumentLoader()
        self.embedder = TextEmbedder()
        self.vector_store = VectorStore()
        self.rag_pipeline = RAGPipeline(
            vector_store=self.vector_store,
            config=rag_config or RAGConfig(),
        )
        self.llm = create_llm(provider=llm_provider, model=llm_model)
        self.prompt_manager = PromptManager()

    def ingest_document(self, file_path: str) -> int:
        """导入单个文档"""
        documents = self.loader.load(file_path)
        split_docs = self.embedder.split_documents(documents)
        self.vector_store.add_documents(split_docs)
        self.rag_pipeline.rebuild_index()
        return len(split_docs)

    def ingest_directory(self, dir_path: str) -> Tuple[int, str]:
        """导入目录下所有文档"""
        documents, result = self.loader.load_directory(dir_path)
        if not documents:
            return 0, result.summary
        split_docs = self.embedder.split_documents(documents)
        self.vector_store.add_documents(split_docs)
        self.rag_pipeline.rebuild_index()
        return len(split_docs), result.summary

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

    def ask(self, question: str) -> Tuple[str, RAGResult]:
        """提问并获取回答"""
        rag_result = self.rag_pipeline.retrieve(question)

        if not rag_result.citations:
            return "知识库中暂无相关内容", rag_result

        # 使用提示词模板
        prompt = self.prompt_manager.render(
            "rag_qa",
            context=rag_result.context,
            question=question,
        )

        try:
            response = self.llm.chat(prompt)
            answer = response.content
        except Exception as e:
            logger.error(f"LLM 生成失败，启用兜底策略: {e}")
            answer = self._format_fallback_answer(rag_result, e)

        answer_with_citations = (
            self.rag_pipeline.citation_manager.format_response_with_citations(
                answer=answer,
                citations=rag_result.citations,
            )
        )

        return answer_with_citations, rag_result

    def ask_stream(
        self,
        question: str,
        history: Optional[List[dict]] = None,
    ) -> Generator[Tuple[str, RAGResult], None, None]:
        """流式提问

        Yields:
            (chunk_content, rag_result)
        """
        rag_result = self.rag_pipeline.retrieve(question)

        if not rag_result.citations:
            yield "知识库中暂无相关内容", rag_result
            return

        # 构建历史消息字符串
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
                question=question,
            )
        else:
            prompt = self.prompt_manager.render(
                "rag_qa",
                context=rag_result.context,
                question=question,
            )

        # 流式生成
        full_answer = ""
        try:
            for chunk in self.llm.stream_chat(prompt):
                if chunk.content:
                    full_answer += chunk.content
                    yield chunk.content, rag_result
        except Exception as e:
            logger.error(f"LLM 流式生成失败，启用兜底策略: {e}")
            fallback_answer = self._format_fallback_answer(rag_result, e)
            full_answer = fallback_answer
            yield fallback_answer, rag_result

        # 最后追加引用
        citations_text = "\n\n---\n**参考来源：**\n"
        for citation in rag_result.citations:
            citations_text += f"[{citation.index}] {citation.source}"
            if citation.page is not None:
                citations_text += f" (第{citation.page + 1}页)"
            citations_text += f" | 相关度: {citation.relevance_score:.1%}\n"

        yield citations_text, rag_result

    def get_vector_store(self) -> VectorStore:
        """获取向量存储"""
        return self.vector_store

    def get_rag_pipeline(self) -> RAGPipeline:
        """获取 RAG 管线"""
        return self.rag_pipeline

    def get_llm(self) -> BaseLLM:
        """获取 LLM 实例"""
        return self.llm

    def get_prompt_manager(self) -> PromptManager:
        """获取提示词管理器"""
        return self.prompt_manager
