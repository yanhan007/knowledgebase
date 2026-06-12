"""LLM 工厂模块

根据配置创建对应的 LLM 实例
支持从全局配置或 YAML 文件加载
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from src.llm.base import BaseLLM, LLMConfig, LLMProvider
from src.llm.ollama_client import OllamaClient
from src.llm.openai_client import OpenAIClient
from config import (
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)

logger = logging.getLogger(__name__)

# 全局配置缓存
_global_config: Optional[Dict[str, Any]] = None


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """从 YAML 文件加载配置"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"从 YAML 加载配置: {config_path}")
    return config or {}


def set_global_config(config: Dict[str, Any]) -> None:
    """设置全局配置"""
    global _global_config
    _global_config = config
    logger.info("全局配置已更新")


def get_global_config() -> Dict[str, Any]:
    """获取全局配置"""
    return _global_config or {}


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    config_path: Optional[str] = None,
    **kwargs,
) -> BaseLLM:
    """创建 LLM 实例

    优先级：参数 > 全局配置 > YAML 配置 > 默认配置

    Args:
        provider: 提供商 (openai / deepseek / ollama)
        model: 模型名称
        temperature: 温度
        max_tokens: 最大 token 数
        api_key: API 密钥
        base_url: API 基础 URL
        config_path: YAML 配置文件路径
        **kwargs: 额外配置

    Returns:
        BaseLLM 实例
    """
    # 加载配置（优先级：YAML > 全局 > 默认）
    yaml_config = {}
    if config_path:
        yaml_config = load_yaml_config(config_path)

    global_config = get_global_config()

    # 合并配置
    llm_config = {
        **yaml_config.get("llm", {}),
        **global_config.get("llm", {}),
    }

    # 参数覆盖
    provider_str = provider or llm_config.get("provider") or LLM_PROVIDER
    provider_enum = LLMProvider(provider_str)

    final_model = model or llm_config.get("model")
    final_temperature = temperature if temperature is not None else llm_config.get("temperature")
    final_max_tokens = max_tokens if max_tokens is not None else llm_config.get("max_tokens")
    final_api_key = api_key or llm_config.get("api_key")
    final_base_url = base_url or llm_config.get("base_url")

    # 构建配置
    config = _build_config(
        provider=provider_enum,
        model=final_model,
        temperature=final_temperature,
        max_tokens=final_max_tokens,
        api_key=final_api_key,
        base_url=final_base_url,
        **kwargs,
    )

    # 创建客户端
    if provider_enum == LLMProvider.OLLAMA:
        llm = OllamaClient(config)
    elif provider_enum in (LLMProvider.OPENAI, LLMProvider.DEEPSEEK):
        llm = OpenAIClient(config)
    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")

    logger.info(f"创建 LLM 实例: {provider_enum.value} - {config.model}")
    return llm


def _build_config(
    provider: LLMProvider,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> LLMConfig:
    """构建 LLM 配置"""

    if provider == LLMProvider.OLLAMA:
        return LLMConfig(
            provider=provider,
            model=model or OLLAMA_MODEL,
            temperature=temperature if temperature is not None else LLM_TEMPERATURE,
            max_tokens=max_tokens or LLM_MAX_TOKENS,
            ollama_url=OLLAMA_BASE_URL,
            **kwargs,
        )

    elif provider == LLMProvider.DEEPSEEK:
        return LLMConfig(
            provider=provider,
            model=model or DEEPSEEK_MODEL,
            temperature=temperature if temperature is not None else LLM_TEMPERATURE,
            max_tokens=max_tokens or LLM_MAX_TOKENS,
            api_key=api_key or DEEPSEEK_API_KEY or os.getenv("DEEPSEEK_API_KEY", ""),
            base_url=base_url or DEEPSEEK_BASE_URL,
            **kwargs,
        )

    elif provider == LLMProvider.OPENAI:
        return LLMConfig(
            provider=provider,
            model=model or LLM_MODEL,
            temperature=temperature if temperature is not None else LLM_TEMPERATURE,
            max_tokens=max_tokens or LLM_MAX_TOKENS,
            api_key=api_key or OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", ""),
            base_url=base_url or OPENAI_BASE_URL,
            **kwargs,
        )

    else:
        raise ValueError(f"不支持的提供商: {provider}")
