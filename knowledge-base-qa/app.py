"""知识库问答系统 - 主程序入口"""

import streamlit as st
from pathlib import Path

from src.qa_engine import QAEngine, setup_logging
from src.rag_retriever import RAGConfig, RerankStrategy
from config import DATA_DIR

# 初始化日志
setup_logging(level="INFO")


def init_session_state():
    """初始化会话状态"""
    if "engine" not in st.session_state:
        st.session_state.engine = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


def save_uploaded_file(uploaded_file) -> str:
    """保存上传的文件"""
    file_path = DATA_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)


def create_engine(provider: str, model: str) -> QAEngine:
    """创建 QAEngine 实例"""
    rag_config = RAGConfig(
        retrieval_top_k=10,
        rerank_top_k=5,
        rerank_strategy=RerankStrategy.MMR,
        max_context_tokens=4000,
    )
    return QAEngine(
        llm_provider=provider,
        llm_model=model,
        rag_config=rag_config,
    )


def main():
    st.set_page_config(
        page_title="知识库问答系统",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 知识库问答系统")

    init_session_state()

    # 侧边栏
    with st.sidebar:
        st.header("⚙️ 系统配置")

        provider = st.selectbox(
            "LLM 提供商",
            options=["openai", "deepseek", "ollama"],
            index=0,
        )

        model_map = {
            "openai": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            "deepseek": ["deepseek-chat", "deepseek-reasoner"],
            "ollama": ["qwen2:7b", "qwen2:14b", "llama3:8b"],
        }

        model = st.selectbox(
            "模型",
            options=model_map.get(provider, []),
            index=0,
        )

        if st.button("初始化系统", type="primary"):
            with st.spinner("正在初始化..."):
                st.session_state.engine = create_engine(provider, model)
                stats = st.session_state.engine.get_stats()
                st.success(f"初始化完成: {provider} - {model}")
                st.json(stats)

        st.divider()

        # 健康检查
        if st.button("健康检查"):
            if st.session_state.engine:
                health = st.session_state.engine.health_check()
                for key, status in health.items():
                    icon = "✅" if status else "❌"
                    st.write(f"{icon} {key}")
            else:
                st.warning("请先初始化系统")

        st.divider()

        st.header("📁 文档管理")

        uploaded_files = st.file_uploader(
            "上传文档",
            type=["txt", "md", "pdf", "docx"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("导入选定文档"):
            if st.session_state.engine is None:
                st.error("请先初始化系统")
            else:
                with st.spinner("正在导入..."):
                    for uploaded_file in uploaded_files:
                        file_path = save_uploaded_file(uploaded_file)
                        result = st.session_state.engine.ingest_document(file_path)
                        if result["success"]:
                            st.success(
                                f"✓ {uploaded_file.name} - "
                                f"{result['chunks']} 块 - "
                                f"{result['elapsed_ms']:.0f}ms"
                            )
                        else:
                            st.error(f"✗ {uploaded_file.name} - {result['error']}")

        st.divider()

        # 知识库状态
        st.subheader("📊 系统状态")
        if st.session_state.engine:
            stats = st.session_state.engine.get_stats()
            st.metric("文档数量", stats["vector_store_count"])
            st.metric("LLM", f"{stats['llm_provider']} - {stats['llm_model']}")
        else:
            st.info("未初始化")

        if st.button("清空知识库", type="secondary"):
            if st.session_state.engine:
                st.session_state.engine.vector_store.delete_all()
                st.session_state.messages = []
                st.success("已清空")
                st.rerun()

    # 主界面
    if st.session_state.engine is None:
        st.info("👈 请在左侧选择 LLM 提供商并点击「初始化系统」")
        return

    st.header("💬 对话")

    # 显示历史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            if "metrics" in message and message["metrics"]:
                with st.expander("📊 性能指标"):
                    st.json(message["metrics"])

    # 用户输入
    if question := st.chat_input("请输入您的问题"):
        # 显示用户问题
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # 流式生成回答
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            final_result = None

            history = st.session_state.messages[:-1]

            for chunk, result in st.session_state.engine.ask_stream(
                question, history=history
            ):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
                final_result = result

            message_placeholder.markdown(full_response)

            # 显示性能指标
            if final_result and final_result.metrics:
                with st.expander("📊 性能指标"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("检索耗时", f"{final_result.metrics.retrieval_ms:.0f}ms")
                    col2.metric("LLM 首 Token", f"{final_result.metrics.llm_first_token_ms:.0f}ms")
                    col3.metric("总耗时", f"{final_result.metrics.total_ms:.0f}ms")

            # 显示引用来源
            if final_result and final_result.rag_result and final_result.rag_result.citations:
                with st.expander("📎 参考来源"):
                    for citation in final_result.rag_result.citations:
                        st.markdown(f"**[{citation.index}]** {citation.source}")
                        if citation.page is not None:
                            st.caption(f"页码: {citation.page + 1}")
                        st.caption(f"相关度: {citation.relevance_score:.1%}")

            # 保存助手回复
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "metrics": final_result.metrics.to_dict() if final_result else None,
            })


if __name__ == "__main__":
    main()
