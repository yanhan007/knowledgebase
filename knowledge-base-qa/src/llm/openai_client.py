"""OpenAI 兼容客户端

支持 OpenAI、DeepSeek 等兼容 API
"""

import logging
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

from openai import AsyncOpenAI, OpenAI, Stream, AsyncStream
from openai.types.chat import ChatCompletionChunk

from src.llm.base import BaseLLM, LLMConfig, LLMResponse, StreamChunk

logger = logging.getLogger(__name__)

# tiktoken 缓存
_encoding_cache: Dict[str, Any] = {}


def _get_encoding(model: str):
    """获取 tiktoken 编码器（带缓存）"""
    if model not in _encoding_cache:
        try:
            import tiktoken
            # 尝试按模型名获取编码器
            try:
                _encoding_cache[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # 回退到 cl100k_base
                _encoding_cache[model] = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("tiktoken 未安装，无法精确计算 token 数")
            _encoding_cache[model] = None
    return _encoding_cache[model]


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """计算文本的 token 数"""
    encoding = _get_encoding(model)
    if encoding is None:
        # 回退估算
        return len(text) // 4
    return len(encoding.encode(text))


def count_messages_tokens(messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo") -> int:
    """计算消息列表的 token 数"""
    encoding = _get_encoding(model)
    if encoding is None:
        total = sum(len(m.get("content", "")) for m in messages)
        return total // 4

    # 每条消息有固定开销
    tokens_per_message = 3
    tokens_per_name = 1

    total = 0
    for message in messages:
        total += tokens_per_message
        for key, value in message.items():
            total += len(encoding.encode(value))
            if key == "name":
                total += tokens_per_name
    total += 3  # 回复的 priming
    return total


class OpenAIClient(BaseLLM):
    """OpenAI 兼容客户端

    支持：
    - OpenAI API
    - DeepSeek API
    - 其他 OpenAI 兼容 API
    - 异步调用
    - token 计数
    - 健康检查
    """

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._client: Optional[OpenAI] = None
        self._async_client: Optional[AsyncOpenAI] = None

    @property
    def client(self) -> OpenAI:
        """延迟初始化同步客户端"""
        if self._client is None:
            api_key = self.config.api_key
            if not api_key:
                raise ValueError(
                    f"API Key 未配置。请设置 {self.config.provider.value.upper()}_API_KEY 环境变量"
                )

            kwargs = {"api_key": api_key}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            kwargs["timeout"] = self.config.timeout

            self._client = OpenAI(**kwargs)
            logger.info(
                f"OpenAI 客户端初始化: "
                f"base_url={self.config.base_url or 'default'}, "
                f"model={self.config.model}"
            )
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        """延迟初始化异步客户端"""
        if self._async_client is None:
            api_key = self.config.api_key
            if not api_key:
                raise ValueError(
                    f"API Key 未配置。请设置 {self.config.provider.value.upper()}_API_KEY 环境变量"
                )

            kwargs = {"api_key": api_key}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            kwargs["timeout"] = self.config.timeout

            self._async_client = AsyncOpenAI(**kwargs)
        return self._async_client

    # ==================== 健康检查 ====================

    def health_check(self) -> bool:
        """检查 API 是否可用"""
        try:
            self.client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI 健康检查失败: {e}")
            return False

    async def ahealth_check(self) -> bool:
        """异步检查 API 是否可用"""
        try:
            await self.async_client.models.list()
            return True
        except Exception as e:
            logger.error(f"OpenAI 异步健康检查失败: {e}")
            return False

    # ==================== token 计数 ====================

    def count_tokens(self, text: str) -> int:
        """计算文本 token 数"""
        return count_tokens(text, self.config.model)

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """计算消息 token 数"""
        return count_messages_tokens(messages, self.config.model)

    def _truncate_messages(
        self, messages: List[Dict[str, str]], max_tokens: int
    ) -> List[Dict[str, str]]:
        """截断消息以适应 token 限制"""
        total = self.count_messages_tokens(messages)
        if total <= max_tokens:
            return messages

        # 保留 system 消息
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        system_tokens = self.count_messages_tokens(system_msgs)
        available = max_tokens - system_tokens

        # 从最新的消息开始保留
        truncated = []
        used_tokens = 0
        for msg in reversed(other_msgs):
            msg_tokens = self.count_messages_tokens([msg])
            if used_tokens + msg_tokens > available:
                break
            truncated.insert(0, msg)
            used_tokens += msg_tokens

        return system_msgs + truncated

    # ==================== 同步接口 ====================

    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """生成回答"""
        params = self._merge_kwargs(**kwargs)

        # 截断消息以适应 token 限制
        messages = self._truncate_messages(messages, params.get("max_tokens", self.config.max_tokens) * 3)

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                top_p=params.get("top_p"),
                stream=False,
            )

            choice = response.choices[0]
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage=usage,
                finish_reason=choice.finish_reason or "",
                metadata={
                    "id": response.id,
                    "created": response.created,
                },
            )

        except Exception as e:
            logger.error(f"OpenAI API 调用失败: {e}")
            raise

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Generator[StreamChunk, None, None]:
        """流式生成回答"""
        params = self._merge_kwargs(**kwargs)

        try:
            stream: Stream[ChatCompletionChunk] = (
                self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=params.get("temperature"),
                    max_tokens=params.get("max_tokens"),
                    top_p=params.get("top_p"),
                    stream=True,
                    stream_options={"include_usage": True},
                )
            )

            for chunk in stream:
                if not chunk.choices:
                    # 可能是 usage 信息
                    if chunk.usage:
                        yield StreamChunk(
                            content="",
                            finish_reason="stop",
                            usage={
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens,
                            },
                        )
                    continue

                choice = chunk.choices[0]
                content = choice.delta.content or ""
                finish_reason = choice.finish_reason or ""

                yield StreamChunk(
                    content=content,
                    finish_reason=finish_reason,
                    metadata={
                        "id": chunk.id,
                        "model": chunk.model,
                    },
                )

                if finish_reason:
                    break

        except Exception as e:
            logger.error(f"OpenAI 流式调用失败: {e}")
            raise

    # ==================== 异步接口 ====================

    async def agenerate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """异步生成回答"""
        params = self._merge_kwargs(**kwargs)

        messages = self._truncate_messages(messages, params.get("max_tokens", self.config.max_tokens) * 3)

        try:
            response = await self.async_client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=params.get("temperature"),
                max_tokens=params.get("max_tokens"),
                top_p=params.get("top_p"),
                stream=False,
            )

            choice = response.choices[0]
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage=usage,
                finish_reason=choice.finish_reason or "",
                metadata={
                    "id": response.id,
                    "created": response.created,
                },
            )

        except Exception as e:
            logger.error(f"OpenAI 异步调用失败: {e}")
            raise

    async def astream_generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """异步流式生成回答"""
        params = self._merge_kwargs(**kwargs)

        try:
            stream: AsyncStream[ChatCompletionChunk] = (
                await self.async_client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=params.get("temperature"),
                    max_tokens=params.get("max_tokens"),
                    top_p=params.get("top_p"),
                    stream=True,
                    stream_options={"include_usage": True},
                )
            )

            async for chunk in stream:
                if not chunk.choices:
                    if chunk.usage:
                        yield StreamChunk(
                            content="",
                            finish_reason="stop",
                            usage={
                                "prompt_tokens": chunk.usage.prompt_tokens,
                                "completion_tokens": chunk.usage.completion_tokens,
                                "total_tokens": chunk.usage.total_tokens,
                            },
                        )
                    continue

                choice = chunk.choices[0]
                content = choice.delta.content or ""
                finish_reason = choice.finish_reason or ""

                yield StreamChunk(
                    content=content,
                    finish_reason=finish_reason,
                    metadata={
                        "id": chunk.id,
                        "model": chunk.model,
                    },
                )

                if finish_reason:
                    break

        except Exception as e:
            logger.error(f"OpenAI 异步流式调用失败: {e}")
            raise

    # ==================== 工具方法 ====================

    def list_models(self) -> List[str]:
        """列出可用模型"""
        try:
            models = self.client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []
