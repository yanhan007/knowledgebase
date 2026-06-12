# 知识库问答系统 - 架构说明

## 项目概述

基于 RAG（Retrieval-Augmented Generation）的本地知识库问答系统，支持多格式文档导入、智能检索、多模型调用。

## 目录结构

```
knowledge-base-qa/
├── app.py                    # Streamlit 主界面入口
├── config/
│   ├── __init__.py
│   └── settings.py           # 全局配置（模型、路径、参数）
├── src/
│   ├── document_loader/      # 文档加载模块
│   │   ├── __init__.py
│   │   └── loader.py         # 多格式文档解析器
│   ├── text_splitter/        # 文本分割模块
│   │   ├── __init__.py
│   │   └── splitter.py       # 智能文本分割器
│   ├── embeddings/           # 向量化模块
│   │   ├── __init__.py
│   │   └── embedder.py       # BGE 嵌入模型封装
│   ├── vector_store/         # 向量存储模块
│   │   ├── __init__.py
│   │   └── store.py          # ChromaDB 向量数据库
│   ├── retriever/            # 基础检索模块
│   │   ├── __init__.py
│   │   └── retriever.py      # 简单相似度检索
│   ├── rag_retriever/        # RAG 高级检索模块
│   │   ├── __init__.py
│   │   ├── hybrid_retriever.py  # 混合检索（向量 + BM25）
│   │   ├── reranker.py          # 结果重排（MMR/交叉编码器）
│   │   ├── context_manager.py   # 上下文窗口管理
│   │   ├── citation.py          # 引用溯源
│   │   └── rag_pipeline.py      # RAG 管线整合
│   ├── llm/                  # 大模型调用模块
│   │   ├── __init__.py
│   │   ├── base.py           # 抽象基类 + 数据结构
│   │   ├── ollama_client.py  # Ollama 本地模型客户端
│   │   ├── openai_client.py  # OpenAI/DeepSeek 客户端
│   │   ├── prompts.py        # 提示词模板管理
│   │   └── factory.py        # LLM 工厂（支持 YAML 配置）
│   └── qa_chain/             # 问答链模块
│       ├── __init__.py
│       └── chain.py          # 端到端问答链
├── data/                     # 知识库文档存储（不进版本控制）
├── vector_store/             # ChromaDB 持久化存储（不进版本控制）
└── tests/                    # 测试目录
```

## 核心模块详解

### 1. 文档加载模块 (document_loader)

**职责**：解析多种格式文档，输出统一的 `Document` 对象。

```
loader.py
├── BaseParser        # 解析器抽象基类
├── TextParser        # TXT/MD 解析（多编码回退）
├── PDFParser         # PDF 解析（PyPDF2，逐页提取）
├── DocxParser        # Word 解析（python-docx）
└── DocumentLoader    # 主类（批量加载、错误统计）
```

**关键设计**：
- 策略模式：每种格式对应一个 Parser
- 多编码回退：utf-8 → gbk → gb2312 → latin-1
- 批量加载返回 `LoadResult` 统计成功/失败数

### 2. 文本分割模块 (text_splitter)

**职责**：将长文档分割为适合向量化的小块，保持语义完整性。

```
splitter.py
├── SentenceSplitter    # 句子级分割器（中英文混合）
└── SmartTextSplitter   # 智能分割器（句子合并 + 重叠）
```

**关键设计**：
- 按句子边界分割，不在单词中间切断
- 中文按 `。！？；` 分割，英文按 `.!?;` 分割
- 支持配置 `chunk_size` 和 `chunk_overlap`
- 超长句子按词边界二次分割
- 重叠窗口保证上下文连贯

### 3. 向量化模块 (embeddings)

**职责**：将文本转换为向量表示。

```
embedder.py
└── TextEmbedder    # BGE 嵌入模型封装
```

**关键设计**：
- 使用 `shibing624/text2vec-base-chinese` 中文嵌入模型（768 维）
- 延迟加载模型，减少启动时间
- 集成 `SmartTextSplitter` 进行文档分割

### 4. 向量存储模块 (vector_store)

**职责**：管理向量数据库，支持增删改查和相似度检索。

```
store.py
├── VectorStoreProtocol   # 协议类（接口契约）
├── DistanceMetric        # 距离度量枚举
├── SearchResult          # 检索结果数据类
├── ChangeEvent           # 变更事件
├── ChangeType            # 变更类型枚举
└── VectorStore           # ChromaDB 实现
```

**关键设计**：
- 运行时协议类 `VectorStoreProtocol`，解耦接口与实现
- 变更回调机制 `on_change()`，通知 BM25 索引同步
- 支持 cosine/l2/ip 三种距离度量
- 持久化存储到本地目录

### 5. RAG 高级检索模块 (rag_retriever)

**职责**：整合混合检索、重排、上下文管理和引用溯源。

```
rag_retriever/
├── hybrid_retriever.py
│   ├── FusionMethod         # 融合方式（RRF/加权）
│   ├── HybridSearchResult   # 混合检索结果
│   └── HybridRetriever      # 向量 + BM25 混合检索
│
├── reranker.py
│   ├── RerankStrategy       # 重排策略（MMR/交叉编码器/元数据）
│   ├── RerankConfig         # 重排配置
│   └── Reranker             # 结果重排器
│
├── context_manager.py
│   ├── ContextWindow        # 上下文窗口
│   ├── ContextConfig        # 上下文配置
│   └── ContextManager       # Token 估算 + 截断 + 重叠
│
├── citation.py
│   ├── Citation             # 引用信息
│   ├── CitationContext      # 引用上下文
│   └── CitationManager      # 引用溯源管理
│
└── rag_pipeline.py
    ├── RAGConfig            # RAG 全局配置
    ├── RAGResult            # RAG 检索结果
    └── RAGPipeline          # RAG 管线整合
```

**关键设计**：

#### 混合检索 (HybridRetriever)
- 向量检索 + BM25 关键词检索
- RRF（Reciprocal Rank Fusion）分数融合
- BM25 索引持久化（pickle）
- 向量库变更自动同步（增量更新/标记重建）

#### 结果重排 (Reranker)
- MMR：平衡相关性和多样性
- 交叉编码器：高精度重排，可配置最大重排数
- 元数据加权：按文档属性调整分数
- 延迟加载 + 失败回退

#### 上下文管理 (ContextManager)
- 自定义 token 计数器（支持 tiktoken）
- 贪心算法：尽可能多包含文档
- 保留头尾文档策略
- 截断时添加重叠片段

#### 引用溯源 (CitationManager)
- 基于 SHA256 的稳定 ID（内容不变则 ID 不变）
- 来源文件、页码、段落索引追踪
- Markdown/脚注格式输出

### 6. 大模型调用模块 (llm)

**职责**：统一的 LLM 调用接口，支持多提供商。

```
llm/
├── base.py
│   ├── LLMProvider      # 提供商枚举
│   ├── LLMConfig        # 配置数据类
│   ├── LLMResponse      # 响应数据类
│   ├── StreamChunk       # 流式输出块
│   └── BaseLLM          # 抽象基类
│
├── ollama_client.py
│   └── OllamaClient     # Ollama 本地模型
│
├── openai_client.py
│   ├── count_tokens()          # Token 计数
│   ├── count_messages_tokens() # 消息 Token 计数
│   └── OpenAIClient            # OpenAI/DeepSeek
│
├── prompts.py
│   ├── PromptTemplate   # 提示词模板
│   └── PromptManager    # 模板管理器
│
└── factory.py
    ├── load_yaml_config()   # YAML 配置加载
    ├── set_global_config()  # 全局配置
    └── create_llm()         # LLM 工厂方法
```

**关键设计**：

#### 统一接口 (BaseLLM)
```
同步: generate() → stream_generate() → chat() → stream_chat()
异步: agenerate() → astream_generate() → achat() → astream_chat()
健康: health_check() / ahealth_check()
重试: _retry() / _aretry() (指数退避)
```

#### StreamChunk 统一格式
```python
StreamChunk(
    content: str,           # 文本内容
    finish_reason: str,     # 结束原因
    usage: Dict[str, int],  # Token 用量
    metadata: Dict           # 元数据
)
```

#### 提示词模板
- 内置 6 个模板（rag_qa、summarize、extract_keywords 等）
- 自动检测变量 `$variable`
- 变量校验 + 默认值支持

#### Token 计数
- 集成 tiktoken，精确计算 OpenAI 模型 token
- `_truncate_messages()` 自动截断超长消息

### 7. 问答链模块 (qa_chain)

**职责**：端到端问答流程整合。

```
chain.py
└── QAChain
    ├── ingest_document()    # 导入单个文档
    ├── ingest_directory()   # 导入目录
    ├── ask()                # 同步问答
    └── ask_stream()         # 流式问答
```

**流程**：
```
文档导入:
  load() → split() → embed() → add_to_vector_store() → rebuild_bm25()

问答:
  retrieve() → rerank() → build_context() → render_prompt() → llm.chat() → add_citations()
```

## 数据流

```
┌─────────────────────────────────────────────────────────────────┐
│                         文档导入流程                              │
└─────────────────────────────────────────────────────────────────┘

  文档文件                文本块                向量
    │                     │                    │
    ▼                     ▼                    ▼
┌──────────┐        ┌──────────┐        ┌──────────┐
│ Document │───────▶│  Text    │───────▶│ Embedder │
│  Loader  │        │ Splitter │        │ (BGE)    │
└──────────┘        └──────────┘        └──────────┘
                                              │
                                              ▼
                                      ┌──────────────┐
                                      │ VectorStore  │
                                      │ (ChromaDB)   │
                                      └──────────────┘
                                              │
                                              ▼
                                      ┌──────────────┐
                                      │ BM25 Index   │
                                      │ (持久化)      │
                                      └──────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                         问答流程                                  │
└─────────────────────────────────────────────────────────────────┘

  用户问题                                                    回答 + 引用
    │                                                           ▲
    ▼                                                           │
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    │
│ Hybrid   │───▶│ Reranker │───▶│ Context  │───▶│   LLM    │───┘
│Retriever │    │ (MMR)    │    │ Manager  │    │ (Ollama) │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     │
     ├── Vector Search (ChromaDB)
     └── BM25 Search (rank_bm25)
```

## 配置层级

```
优先级: 参数 > 全局配置 > YAML 配置 > config/settings.py 默认值

config/settings.py     # 硬编码默认值
        │
        ▼
llm_config.yaml        # YAML 配置文件（可选）
        │
        ▼
set_global_config()    # 运行时全局配置
        │
        ▼
create_llm(provider=..., model=...)  # 调用参数
```

## 扩展点

| 扩展点 | 方式 |
|--------|------|
| 新文档格式 | 继承 `BaseParser`，注册到 `DocumentLoader` |
| 新嵌入模型 | 修改 `TextEmbedder` 或实现新 Embeddings 类 |
| 新向量库 | 实现 `VectorStoreProtocol` 协议 |
| 新重排策略 | 添加 `RerankStrategy` 枚举，实现对应方法 |
| 新 LLM 提供商 | 继承 `BaseLLM`，添加到 `LLMProvider` 和工厂 |
| 新提示词模板 | `PromptManager.register()` 注册 |

## 外部依赖

| 依赖 | 用途 |
|------|------|
| chromadb | 向量数据库 |
| sentence-transformers | 嵌入模型加载 |
| rank_bm25 | BM25 关键词检索 |
| openai | OpenAI/DeepSeek API |
| tiktoken | Token 精确计数 |
| pypdf2 | PDF 解析 |
| python-docx | Word 解析 |
| streamlit | Web 界面 |
| pyyaml | YAML 配置 |
| httpx | HTTP 客户端（Ollama） |

## 启动方式

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量（可选）
export OPENAI_API_KEY=sk-xxx
export DEEPSEEK_API_KEY=sk-xxx

# 启动 Streamlit 界面
streamlit run app.py
```
