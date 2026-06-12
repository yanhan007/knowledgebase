"""项目配置文件"""

import os
from pathlib import Path

# 设置 Hugging Face 镜像源（中国大陆加速）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 文档存储路径
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# 向量数据库配置
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"
VECTOR_STORE_DIR.mkdir(exist_ok=True)

# Embedding 模型配置
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
EMBEDDING_DIMENSION = 768

# 文本分割配置
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# LLM 配置
LLM_PROVIDER = "deepseek"  # openai / deepseek / ollama
LLM_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2000

# Ollama 配置
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2:7b"

# DeepSeek 配置
DEEPSEEK_API_KEY = "sk-6d3f2497e7c143bdbad5b1e548b0f73c"  # 从环境变量 DEEPSEEK_API_KEY 读取
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# OpenAI 配置（兼容 API）
OPENAI_API_KEY = "sk-6d3f2497e7c143bdbad5b1e548b0f73c"  # 从环境变量 OPENAI_API_KEY 读取
OPENAI_BASE_URL = "https://api.deepseek.com/v1"

# 检索配置
TOP_K = 3
