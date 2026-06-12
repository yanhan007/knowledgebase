# 📚 知识库问答系统

基于 **RAG（Retrieval-Augmented Generation）** 的本地知识库问答系统，支持多格式文档导入、智能检索、多模型调用、流式输出。

## ✨ 功能特性

### 文档管理
- 📄 支持 TXT、MD、PDF、DOCX 四种格式
- 📤 批量上传与导入
- 📊 知识库状态监控

### 智能处理
- ✂️ 智能文本分割（中英文混合，句子边界保持）
- 🔢 本地嵌入模型向量化（text2vec-base-chinese，768 维）
- 💾 ChromaDB 持久化存储

### 检索能力
- 🔍 向量相似度检索
- 📝 BM25 关键词检索
- 🔀 RRF 分数融合（混合检索）
- 🎯 MMR / 交叉编码器重排
- 📎 引用溯源（来源、页码、段落）

### 模型支持
- 🤖 OpenAI / DeepSeek / Ollama 多提供商
- ⚡ 流式输出
- 🔄 自动重试机制
- 📊 Token 精确计数

### 界面
- 🖥️ **Streamlit** 主界面（完整功能）
- 💬 **Gradio** 聊天界面（流式对话）

## 🛠️ 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| Web 框架 | Streamlit 1.28+ | 主界面 |
| | Gradio 5.50+ | 聊天界面 |
| 向量数据库 | ChromaDB 0.4+ | 向量存储与检索 |
| 嵌入模型 | sentence-transformers | 本地文本向量化 |
| | text2vec-base-chinese | 中文嵌入（768 维） |
| 关键词检索 | rank_bm25 0.2+ | BM25 检索 |
| LLM | OpenAI SDK 1.0+ | OpenAI / DeepSeek API |
| | Ollama | 本地大模型 |
| 文档解析 | PyPDF2 3.0+ | PDF 解析 |
| | python-docx 0.8+ | Word 解析 |
| Token 计数 | tiktoken 0.5+ | OpenAI Token 精确计算 |

## 📁 项目结构

```
knowledge-base-qa/
├── app.py                         # Streamlit 主界面
├── app_gradio.py                  # Gradio 聊天界面
├── config/
│   ├── __init__.py
│   └── settings.py                # 全局配置
├── src/
│   ├── qa_engine.py               # 问答引擎（核心入口）
│   ├── document_loader/           # 文档加载模块
│   │   └── loader.py              # 多格式解析器
│   ├── text_splitter/             # 文本分割模块
│   │   └── splitter.py            # 智能分割器
│   ├── embeddings/                # 向量化模块
│   │   └── embedder.py            # 嵌入模型封装
│   ├── vector_store/              # 向量存储模块
│   │   └── store.py               # ChromaDB 封装
│   ├── retriever/                 # 基础检索模块
│   │   └── retriever.py           # 简单相似度检索
│   ├── rag_retriever/             # RAG 高级检索模块
│   │   ├── hybrid_retriever.py    # 混合检索（向量 + BM25）
│   │   ├── reranker.py            # 结果重排
│   │   ├── context_manager.py     # 上下文管理
│   │   ├── citation.py            # 引用溯源
│   │   └── rag_pipeline.py        # RAG 管线整合
│   ├── llm/                       # 大模型调用模块
│   │   ├── base.py                # 抽象基类
│   │   ├── ollama_client.py       # Ollama 客户端
│   │   ├── openai_client.py       # OpenAI/DeepSeek 客户端
│   │   ├── prompts.py             # 提示词模板
│   │   └── factory.py             # LLM 工厂
│   └── qa_chain/                  # 问答链模块
│       └── chain.py               # 端到端问答链
├── data/                          # 文档存储（不进版本控制）
├── vector_store/                  # ChromaDB 持久化（不进版本控制）
├── tests/                         # 测试目录
├── requirements.txt               # 依赖清单
├── CLAUDE.md                      # 项目规则
└── ARCHITECTURE.md                # 架构说明
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

编辑 `config/settings.py`，填入你的 API Key：

```python
# DeepSeek 配置
DEEPSEEK_API_KEY = "your_deepseek_api_key"

# OpenAI 配置（可选）
OPENAI_API_KEY = "your_openai_api_key"
```

### 3. 启动应用

**Streamlit 界面（推荐）：**
```bash
streamlit run app.py
```

**Gradio 界面：**
```bash
python app_gradio.py
```

## 📖 使用方法

### Streamlit 界面

1. 在左侧边栏选择 LLM 提供商（OpenAI / DeepSeek / Ollama）
2. 点击「初始化系统」
3. 上传文档（支持多文件）
4. 点击「导入选定文档」
5. 在对话框输入问题
6. 查看回答和引用来源

### Gradio 界面

1. 切换到「📁 文档管理」标签
2. 上传并导入文档
3. 切换到「💬 聊天」标签
4. 输入问题，支持流式输出

## ⚙️ 配置说明

### LLM 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_PROVIDER` | `deepseek` | 提供商：openai / deepseek / ollama |
| `LLM_MODEL` | `deepseek-chat` | 模型名称 |
| `LLM_TEMPERATURE` | `0.7` | 生成温度 |
| `LLM_MAX_TOKENS` | `2000` | 最大 Token 数 |

### 文本分割配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `CHUNK_SIZE` | `500` | 每个文本块的最大字符数 |
| `CHUNK_OVERLAP` | `50` | 相邻文本块的重叠字符数 |

### 检索配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `TOP_K` | `3` | 返回的相关文档数量 |

### 嵌入模型配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `EMBEDDING_MODEL` | `shibing624/text2vec-base-chinese` | 本地嵌入模型 |
| `EMBEDDING_DIMENSION` | `768` | 向量维度 |

## 🔄 数据流

### 文档导入流程

```
文档文件 → DocumentLoader → TextSplitter → Embedder → VectorStore + BM25 Index
```

### 问答流程

```
用户问题 → HybridRetriever → Reranker → ContextManager → LLM → 回答 + 引用
```

## 🧩 扩展点

| 扩展点 | 方式 |
|--------|------|
| 新文档格式 | 继承 `BaseParser`，注册到 `DocumentLoader` |
| 新嵌入模型 | 修改 `TextEmbedder` 或实现新 Embeddings 类 |
| 新向量库 | 实现 `VectorStoreProtocol` 协议 |
| 新重排策略 | 添加 `RerankStrategy` 枚举，实现对应方法 |
| 新 LLM 提供商 | 继承 `BaseLLM`，添加到 `LLMProvider` 和工厂 |
| 新提示词模板 | `PromptManager.register()` 注册 |

## 📌 当前限制

- ❌ 不支持多模态（图像/音频）
- ❌ 不支持文档增量更新（需全量重建）
- ❌ 不支持对话记忆（单轮问答）

## 📄 许可证

MIT License
