# 知识库问答系统

本地知识库问答系统，支持文档上传、向量化、检索和智能问答。

## 功能特性

- 📄 支持多种文档格式：TXT、MD、PDF、DOCX
- 🔍 智能文本分割和向量化
- 💾 基于 Chroma 的向量存储
- 💬 基于 LLM 的智能问答
- 🖥️ 简洁的 Streamlit 界面

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件并添加 OpenAI API Key：

```
OPENAI_API_KEY=your_api_key_here
```

### 3. 启动应用

```bash
streamlit run app.py
```

## 使用方法

1. 在左侧边栏上传文档
2. 点击"导入选定文档"按钮
3. 在对话框输入问题
4. 查看回答和参考来源

## 项目结构

```
knowledge-base-qa/
├── app.py                 # 主程序入口
├── config/                # 配置文件
├── src/
│   ├── document_loader/   # 文档加载
│   ├── embeddings/        # 文本向量化
│   ├── vector_store/      # 向量存储
│   ├── retriever/         # 检索逻辑
│   ├── llm/               # LLM 调用
│   └── qa_chain/          # 问答链
├── data/                  # 文档存储
└── tests/                 # 测试
```
