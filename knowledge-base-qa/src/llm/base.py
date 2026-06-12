"""LLM 基础抽象类和数据结构"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """LLM 提供商"""

    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"


@dataclass
class LLMConfig:
    """LLM 配置"""

    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0

    # 提供商特定配置
    api_key: str = ""
    base_url: str = ""

    # Ollama 特定
    ollama_url: str = "http://localhost:11434"


@dataclass
class LLMResponse:
    """LLM 响应"""

    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)


@dataclass
class StreamChunk:
    """流式输出块"""

    content: str
    finish_reason: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLLM(ABC):
    """LLM 基类"""

    def __init__(self, config: LLMConfig):
        self.config = config

    # ==================== 同步接口 ====================

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """生成回答"""
        pass

    @abstractmethod
    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Generator[StreamChunk, None, None]:
        """流式生成回答"""
        pass

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> LLMResponse:
        """对话接口（简化版）"""
        messages = self._build_messages(user_message, system_prompt, history)
        return self._retry(lambda: self.generate(messages, **kwargs))

    def stream_chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> Generator[StreamChunk, None, None]:
        """流式对话接口（简化版）"""
        messages = self._build_messages(user_message, system_prompt, history)
        yield from self.stream_generate(messages, **kwargs)

    # ==================== 异步接口 ====================

    @abstractmethod
    async def agenerate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """异步生成回答"""
        pass

    @abstractmethod
    async def astream_generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """异步流式生成回答"""
        pass

    async def achat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> LLMResponse:
        """异步对话接口"""
        messages = self._build_messages(user_message, system_prompt, history)
        return await self._aretry(lambda: self.agenerate(messages, **kwargs))

    async def astream_chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """异步流式对话接口"""
        messages = self._build_messages(user_message, system_prompt, history)
        async for chunk in self.astream_generate(messages, **kwargs):
            yield chunk

    # ==================== 健康检查 ====================

    @abstractmethod
    def health_check(self) -> bool:
        """检查服务是否可用"""
        pass

    @abstractmethod
    async def ahealth_check(self) -> bool:
        """异步检查服务是否可用"""
        pass

    # ==================== 工具方法 ====================

    def _build_messages(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """构建消息列表"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _merge_kwargs(self, **kwargs) -> Dict[str, Any]:
        """合并默认参数和额外参数"""
        defaults = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        defaults.update(kwargs)
        return defaults

    def _retry(self, fn, *args, **kwargs):
        """带重试的同步调用"""
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"调用失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}, "
                        f"{delay}s 后重试"
                    )
                    time.sleep(delay)
        raise last_error

    async def _aretry(self, fn, *args, **kwargs):
        """带重试的异步调用"""
        import asyncio
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"异步调用失败 (尝试 {attempt + 1}/{self.config.max_retries}): {e}, "
                        f"{delay}s 后重试"
                    )
                    await asyncio.sleep(delay)
        raise last_error
