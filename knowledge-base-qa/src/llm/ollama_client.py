"""Ollama 客户端"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

import httpx

from src.llm.base import BaseLLM, LLMConfig, LLMResponse, StreamChunk

logger = logging.getLogger(__name__)


class OllamaClient(BaseLLM):
    """Ollama 本地模型客户端"""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = config.ollama_url.rstrip("/")

    def _chat_endpoint(self) -> str:
        return f"{self.base_url}/api/chat"

    def _generate_endpoint(self) -> str:
        return f"{self.base_url}/api/generate"

    def _tags_endpoint(self) -> str:
        return f"{self.base_url}/api/tags"

    def _build_payload(self, messages: List[Dict[str, str]], stream: bool, **kwargs) -> Dict:
        """构建请求 payload"""
        params = self._merge_kwargs(**kwargs)
        return {
            "model": self.config.model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": params.get("temperature", self.config.temperature),
                "num_predict": params.get("max_tokens", self.config.max_tokens),
                "top_p": params.get("top_p", self.config.top_p),
            },
        }

    def _parse_response(self, data: Dict) -> LLMResponse:
        """解析 Ollama 响应为统一格式"""
        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=data.get("model", self.config.model),
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (
                    data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                ),
            },
            finish_reason=data.get("done_reason", "stop"),
            metadata={
                "total_duration": data.get("total_duration", 0),
                "eval_duration": data.get("eval_duration", 0),
            },
        )

    def _parse_stream_line(self, data: Dict) -> StreamChunk:
        """解析流式响应行为统一 StreamChunk"""
        content = data.get("message", {}).get("content", "")
        done = data.get("done", False)

        usage = {}
        if done:
            usage = {
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (
                    data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                ),
            }

        return StreamChunk(
            content=content,
            finish_reason="stop" if done else "",
            usage=usage,
            metadata={
                "model": data.get("model", ""),
                "total_duration": data.get("total_duration", 0),
            },
        )

    # ==================== 健康检查 ====================

    def health_check(self) -> bool:
        """检查 Ollama 服务是否可用"""
        try:
            with httpx.Client(timeout=5) as client:
                response = client.get(self._tags_endpoint())
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama 健康检查失败: {e}")
            return False

    async def ahealth_check(self) -> bool:
        """异步检查 Ollama 服务是否可用"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(self._tags_endpoint())
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama 异步健康检查失败: {e}")
            return False

    # ==================== 同步接口 ====================

    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """生成回答"""
        payload = self._build_payload(messages, stream=False, **kwargs)

        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(self._chat_endpoint(), json=payload)
                response.raise_for_status()
                data = response.json()

            return self._parse_response(data)

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API 错误: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Ollama 调用失败: {e}")
            raise

    def stream_generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Generator[StreamChunk, None, None]:
        """流式生成回答"""
        payload = self._build_payload(messages, stream=True, **kwargs)

        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                with client.stream(
                    "POST", self._chat_endpoint(), json=payload
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            chunk = self._parse_stream_line(data)
                            yield chunk

                            if chunk.finish_reason == "stop":
                                break

                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Ollama 流式调用失败: {e}")
            raise

    # ==================== 异步接口 ====================

    async def agenerate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> LLMResponse:
        """异步生成回答"""
        payload = self._build_payload(messages, stream=False, **kwargs)

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(self._chat_endpoint(), json=payload)
                response.raise_for_status()
                data = response.json()

            return self._parse_response(data)

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama 异步 API 错误: {e.response.status_code}")
            raise
        except Exception as e:
            logger.error(f"Ollama 异步调用失败: {e}")
            raise

    async def astream_generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> AsyncGenerator[StreamChunk, None]:
        """异步流式生成回答"""
        payload = self._build_payload(messages, stream=True, **kwargs)

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                async with client.stream(
                    "POST", self._chat_endpoint(), json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            chunk = self._parse_stream_line(data)
                            yield chunk

                            if chunk.finish_reason == "stop":
                                break

                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            logger.error(f"Ollama 异步流式调用失败: {e}")
            raise

    # ==================== 工具方法 ====================

    def list_models(self) -> List[str]:
        """列出可用模型"""
        try:
            with httpx.Client(timeout=10) as client:
                response = client.get(self._tags_endpoint())
                response.raise_for_status()
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"获取 Ollama 模型列表失败: {e}")
            return []
