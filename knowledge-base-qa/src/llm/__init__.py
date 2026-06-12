"""大模型调用模块

支持：
- 本地模型（Ollama）
- 云端 API（OpenAI、DeepSeek）
- 统一提示词模板（带变量校验）
- 同步/异步调用
- 流式输出
- 重试机制
- 健康检查
"""

from .base import BaseLLM, LLMConfig, LLMProvider, LLMResponse, StreamChunk
from .factory import create_llm, load_yaml_config, set_global_config
from .prompts import PromptTemplate, PromptManager
from .openai_client import count_tokens, count_messages_tokens

__all__ = [
    # 基础类
    "BaseLLM",
    "LLMConfig",
    "LLMProvider",
    "LLMResponse",
    "StreamChunk",
    # 工厂
    "create_llm",
    "load_yaml_config",
    "set_global_config",
    # 提示词
    "PromptTemplate",
    "PromptManager",
    # 工具
    "count_tokens",
    "count_messages_tokens",
]
