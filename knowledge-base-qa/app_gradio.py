"""知识库问答系统 - Gradio Web 界面

功能：
- 文件上传与文档管理
- 聊天对话界面（支持流式输出）
- 知识库管理面板
- 系统配置与监控
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Tuple, Optional

import gradio as gr

from src.qa_engine import QAEngine, setup_logging
from config import settings

# 日志配置
setup_logging(level="INFO")
logger = logging.getLogger(__name__)

# 全局引擎实例
engine: Optional[QAEngine] = None


def get_engine(
    provider: str = "deepseek",
    model: str = "deepseek-chat",
) -> QAEngine:
    """获取或创建 QAEngine 单例"""
    global engine
    if engine is None:
        engine = QAEngine(llm_provider=provider, llm_model=model)
    return engine


# ==================== 核心功能 ====================


def upload_files(files: List[str]) -> str:
    """上传并导入文件到知识库"""
    if not files:
        return "请选择要上传的文件"

    eng = get_engine()
    results = []
    total_chunks = 0

    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            result = eng.ingest_document(file_path)
            if result.get("success"):
                chunks = result.get("chunks", 0)
                elapsed = result.get("elapsed_ms", 0)
                total_chunks += chunks
                results.append(f"✅ {filename}: {chunks} 块 ({elapsed:.0f}ms)")
            else:
                error = result.get("error", "未知错误")
                results.append(f"❌ {filename}: {error}")
        except Exception as e:
            results.append(f"❌ {filename}: {str(e)}")

    summary = f"\n📊 导入完成: {len(files)} 个文件, {total_chunks} 个文本块"
    return "\n".join(results) + summary


def chat_response(
    message: str,
    history: List[Tuple[str, str]],
    provider: str,
    model: str,
    use_stream: bool,
) -> str:
    """处理聊天消息"""
    if not message.strip():
        return ""

    eng = get_engine(provider, model)

    try:
        if use_stream:
            # 流式输出
            full_answer = ""
            for chunk, result in eng.ask_stream(message):
                if chunk:
                    full_answer += chunk
            return full_answer
        else:
            # 同步输出
            result = eng.ask(message)
            return result.answer
    except Exception as e:
        logger.error(f"问答失败: {e}")
        return f"❌ 错误: {str(e)}"


def chat_response_stream(
    message: str,
    history: List[Tuple[str, str]],
    provider: str,
    model: str,
):
    """流式聊天响应"""
    if not message.strip():
        yield ""
        return

    eng = get_engine(provider, model)

    try:
        full_answer = ""
        for chunk, result in eng.ask_stream(message):
            if chunk:
                full_answer += chunk
                yield full_answer
    except Exception as e:
        logger.error(f"流式问答失败: {e}")
        yield f"❌ 错误: {str(e)}"


def get_knowledge_base_info() -> str:
    """获取知识库信息"""
    eng = get_engine()

    try:
        count = eng.vector_store.count()
        health = eng.health_check()

        info = f"""📚 **知识库状态**

| 项目 | 状态 |
|------|------|
| 文档数量 | {count} |
| LLM 服务 | {'✅ 正常' if health.get('llm') else '❌ 异常'} |
| 向量存储 | {'✅ 正常' if health.get('vector_store') else '❌ 异常'} |

**配置信息**
- 嵌入模型: {settings.EMBEDDING_MODEL}
- 向量维度: {settings.EMBEDDING_DIMENSION}
- 分块大小: {settings.CHUNK_SIZE}
- 分块重叠: {settings.CHUNK_OVERLAP}
"""
        return info
    except Exception as e:
        return f"❌ 获取信息失败: {str(e)}"


def clear_knowledge_base() -> str:
    """清空知识库"""
    eng = get_engine()

    try:
        eng.vector_store.reset()
        return "✅ 知识库已清空"
    except Exception as e:
        return f"❌ 清空失败: {str(e)}"


def update_settings(
    provider: str,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
) -> str:
    """更新系统设置"""
    global engine

    try:
        # 更新配置
        settings.LLM_PROVIDER = provider
        settings.LLM_MODEL = model
        settings.CHUNK_SIZE = chunk_size
        settings.CHUNK_OVERLAP = chunk_overlap
        settings.TOP_K = top_k

        # 重置引擎以应用新配置
        engine = None

        return f"""✅ 设置已更新

- LLM 提供商: {provider}
- LLM 模型: {model}
- 分块大小: {chunk_size}
- 分块重叠: {chunk_overlap}
- 检索数量: {top_k}

⚠️ 新配置将在下次问答时生效
"""
    except Exception as e:
        return f"❌ 更新失败: {str(e)}"


def run_health_check() -> str:
    """运行健康检查"""
    eng = get_engine()

    try:
        health = eng.health_check()
        stats = eng.get_stats()

        report = f"""🏥 **系统健康检查**

| 组件 | 状态 |
|------|------|
| LLM 服务 | {'✅ 正常' if health.get('llm') else '❌ 异常'} |
| 向量存储 | {'✅ 正常' if health.get('vector_store') else '❌ 异常'} |

📊 **系统统计**
- 向量库文档数: {stats.get('vector_store_count', 0)}
- LLM 提供商: {stats.get('llm_provider', 'N/A')}
- LLM 模型: {stats.get('llm_model', 'N/A')}
"""
        return report
    except Exception as e:
        return f"❌ 健康检查失败: {str(e)}"


# ==================== Gradio 界面 ====================


def create_ui() -> gr.Blocks:
    """创建 Gradio 界面"""

    with gr.Blocks(
        title="知识库问答系统",
        theme=gr.themes.Soft(),
    ) as app:

        # ==================== 标题 ====================
        gr.Markdown(
            """
            # 📚 知识库问答系统
            **基于 RAG 的智能文档问答系统** | 支持 PDF/Word/TXT/Markdown
            """,
            elem_classes="main-header",
        )

        with gr.Tabs():
            # ==================== Tab 1: 聊天 ====================
            with gr.Tab("💬 聊天", id="chat"):
                with gr.Row():
                    with gr.Column(scale=4):
                        chatbot = gr.Chatbot(
                            label="对话历史",
                            height=500,
                            type="messages",
                        )

                        with gr.Row():
                            msg_input = gr.Textbox(
                                label="输入问题",
                                placeholder="请输入您的问题...",
                                lines=2,
                                scale=4,
                            )
                            submit_btn = gr.Button(
                                "发送",
                                variant="primary",
                                scale=1,
                            )

                        with gr.Row():
                            clear_btn = gr.Button("🗑️ 清空对话")
                            retry_btn = gr.Button("🔄 重试")

                    with gr.Column(scale=1):
                        gr.Markdown("### ⚙️ 聊天设置")

                        stream_toggle = gr.Checkbox(
                            label="流式输出",
                            value=True,
                            info="启用流式响应",
                        )

                        provider_select = gr.Dropdown(
                            choices=["deepseek", "openai", "ollama"],
                            value="deepseek",
                            label="LLM 提供商",
                        )

                        model_input = gr.Textbox(
                            value="deepseek-chat",
                            label="模型名称",
                            info="DeepSeek: deepseek-chat",
                        )

                        gr.Markdown("### 📝 对话提示")
                        gr.Markdown(
                            """
                            - 支持中英文问答
                            - 基于知识库内容回答
                            - 自动引用来源文档
                            """
                        )

            # ==================== Tab 2: 文档管理 ====================
            with gr.Tab("📁 文档管理", id="docs"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📤 上传文档")

                        file_upload = gr.File(
                            label="选择文件",
                            file_count="multiple",
                            file_types=[".pdf", ".docx", ".txt", ".md"],
                            type="filepath",
                        )

                        upload_btn = gr.Button(
                            "📤 导入文档",
                            variant="primary",
                        )

                        upload_output = gr.Textbox(
                            label="导入结果",
                            lines=10,
                            interactive=False,
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 知识库状态")

                        kb_info_btn = gr.Button("🔄 刷新状态")
                        kb_info_output = gr.Markdown(
                            value="点击按钮刷新状态",
                        )

                        gr.Markdown("### ⚠️ 危险操作")

                        clear_kb_btn = gr.Button(
                            "🗑️ 清空知识库",
                            variant="stop",
                        )
                        clear_kb_output = gr.Textbox(
                            label="操作结果",
                            interactive=False,
                        )

            # ==================== Tab 3: 设置 ====================
            with gr.Tab("⚙️ 设置", id="settings"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🤖 LLM 配置")

                        setting_provider = gr.Dropdown(
                            choices=["deepseek", "openai", "ollama"],
                            value="deepseek",
                            label="LLM 提供商",
                        )

                        setting_model = gr.Textbox(
                            value="deepseek-chat",
                            label="模型名称",
                        )

                        gr.Markdown("### 📄 文本分割")

                        setting_chunk_size = gr.Slider(
                            minimum=100,
                            maximum=2000,
                            value=settings.CHUNK_SIZE,
                            step=50,
                            label="分块大小",
                            info="每个文本块的最大字符数",
                        )

                        setting_chunk_overlap = gr.Slider(
                            minimum=0,
                            maximum=500,
                            value=settings.CHUNK_OVERLAP,
                            step=10,
                            label="分块重叠",
                            info="相邻文本块的重叠字符数",
                        )

                        gr.Markdown("### 🔍 检索配置")

                        setting_top_k = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=settings.TOP_K,
                            step=1,
                            label="检索数量",
                            info="返回的相关文档数量",
                        )

                        save_settings_btn = gr.Button(
                            "💾 保存设置",
                            variant="primary",
                        )

                        settings_output = gr.Textbox(
                            label="保存结果",
                            lines=8,
                            interactive=False,
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### 🏥 系统状态")

                        health_check_btn = gr.Button("🔍 运行健康检查")

                        health_output = gr.Markdown(
                            value="点击按钮运行健康检查",
                        )

                        gr.Markdown("### 📖 使用说明")
                        gr.Markdown(
                            """
                            **首次使用**

                            1. 在「文档管理」上传知识文档
                            2. 等待文档导入完成
                            3. 在「聊天」中提问

                            **配置说明**

                            - **分块大小**: 影响检索精度，越小越精确
                            - **分块重叠**: 保证上下文连贯性
                            - **检索数量**: 返回的相关片段数量

                            **支持格式**

                            - PDF 文档
                            - Word 文档 (.docx)
                            - 纯文本 (.txt)
                            - Markdown (.md)
                            """
                        )

        # ==================== 事件绑定 ====================

        # 聊天功能
        def user_input(message, history):
            return "", history + [{"role": "user", "content": message}]

        def bot_response(history, provider, model, stream):
            # 从 history 中取最后一条用户消息
            if not history:
                yield history
                return

            last = history[-1]
            # 兼容多种格式
            if isinstance(last, dict):
                message = last.get("content", "")
            elif isinstance(last, (list, tuple)):
                message = last[0] if last and last[0] else ""
            else:
                message = str(last) if last else ""

            if not isinstance(message, str) or not message.strip():
                yield history
                return

            if stream:
                # 流式输出
                eng = get_engine(provider, model)
                full_answer = ""
                for chunk, result in eng.ask_stream(message):
                    if chunk:
                        full_answer += chunk
                        yield history + [{"role": "assistant", "content": full_answer}]
            else:
                # 同步输出
                eng = get_engine(provider, model)
                result = eng.ask(message)
                yield history + [{"role": "assistant", "content": result.answer}]

        msg_input.submit(
            user_input,
            [msg_input, chatbot],
            [msg_input, chatbot],
        ).then(
            bot_response,
            [chatbot, provider_select, model_input, stream_toggle],
            chatbot,
        )

        submit_btn.click(
            user_input,
            [msg_input, chatbot],
            [msg_input, chatbot],
        ).then(
            bot_response,
            [chatbot, provider_select, model_input, stream_toggle],
            chatbot,
        )

        clear_btn.click(lambda: [], outputs=chatbot, queue=False)
        retry_btn.click(lambda h: h[:-2] if len(h) >= 2 else h, chatbot, chatbot, queue=False)

        # 文档管理功能
        upload_btn.click(
            upload_files,
            file_upload,
            upload_output,
        )

        kb_info_btn.click(
            get_knowledge_base_info,
            outputs=kb_info_output,
        )

        clear_kb_btn.click(
            clear_knowledge_base,
            outputs=clear_kb_output,
        )

        # 设置功能
        save_settings_btn.click(
            update_settings,
            [
                setting_provider,
                setting_model,
                setting_chunk_size,
                setting_chunk_overlap,
                setting_top_k,
            ],
            settings_output,
        )

        health_check_btn.click(
            run_health_check,
            outputs=health_output,
        )

    return app


# ==================== 启动入口 ====================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="知识库问答系统 Web 界面")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="端口号")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    parser.add_argument("--debug", action="store_true", help="调试模式")

    args = parser.parse_args()

    # 创建界面
    app = create_ui()

    # 启动服务
    app.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        debug=args.debug,
    )
