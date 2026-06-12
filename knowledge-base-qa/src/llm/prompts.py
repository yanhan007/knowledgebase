"""提示词模板管理

功能：
- 预定义模板
- 自定义模板
- 变量校验与替换
- 默认值支持
- 自动检测变量
"""

import re
import logging
from dataclasses import dataclass, field
from string import Template
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """提示词模板"""

    name: str
    template: str
    description: str = ""
    variables: List[str] = field(default_factory=list)
    defaults: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后自动检测变量"""
        if not self.variables:
            self.variables = self._detect_variables()

    def _detect_variables(self) -> List[str]:
        """从模板中自动检测变量名"""
        # 匹配 $variable 和 ${variable} 格式
        pattern = r'\$\{?([a-zA-Z_][a-zA-Z0-9_]*)\}?'
        matches = re.findall(pattern, self.template)
        # 去重并保持顺序
        seen = set()
        variables = []
        for var in matches:
            if var not in seen:
                seen.add(var)
                variables.append(var)
        return variables

    def validate(self, kwargs: Dict[str, Any]) -> List[str]:
        """校验变量，返回缺失的变量列表"""
        provided = set(kwargs.keys()) | set(self.defaults.keys())
        required = set(self.variables)
        missing = required - provided
        return list(missing)

    def render(self, **kwargs) -> str:
        """渲染模板

        Args:
            **kwargs: 模板变量

        Returns:
            渲染后的字符串

        Raises:
            ValueError: 缺少必需变量
        """
        # 合并默认值
        merged = {**self.defaults, **kwargs}

        # 校验变量
        missing = self.validate(kwargs)
        if missing:
            raise ValueError(
                f"模板 '{self.name}' 缺少必需变量: {missing}\n"
                f"需要: {self.variables}\n"
                f"提供: {list(kwargs.keys())}"
            )

        try:
            tmpl = Template(self.template)
            return tmpl.safe_substitute(**merged)
        except Exception as e:
            logger.error(f"模板 '{self.name}' 渲染失败: {e}")
            raise


class PromptManager:
    """提示词管理器"""

    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._load_builtin_templates()

    def _load_builtin_templates(self) -> None:
        """加载内置模板"""
        # RAG 问答模板
        self.register(
            PromptTemplate(
                name="rag_qa",
                template="""你是一个知识库问答助手。请根据提供的上下文信息回答问题。

规则：
1. 只根据上下文中的信息回答，不要编造内容
2. 如果上下文中没有相关信息，请如实说明
3. 回答要准确、简洁、有条理
4. 适当引用来源编号 [1]、[2] 等

上下文：
$context

问题：$question

请给出准确的回答：""",
                description="RAG 问答模板",
                variables=["context", "question"],
            )
        )

        # 带历史的 RAG 问答模板
        self.register(
            PromptTemplate(
                name="rag_qa_with_history",
                template="""你是一个知识库问答助手。请根据提供的上下文和对话历史回答问题。

规则：
1. 只根据上下文中的信息回答，不要编造内容
2. 考虑对话历史中的上下文
3. 如果上下文中没有相关信息，请如实说明
4. 回答要准确、简洁、有条理

对话历史：
$history

上下文：
$context

问题：$question

请给出准确的回答：""",
                description="带历史的 RAG 问答模板",
                variables=["history", "context", "question"],
            )
        )

        # 总结模板
        self.register(
            PromptTemplate(
                name="summarize",
                template="""请对以下内容进行总结，要求简洁明了，抓住要点。

内容：
$content

总结：""",
                description="文本总结模板",
                variables=["content"],
            )
        )

        # 关键词提取模板
        self.register(
            PromptTemplate(
                name="extract_keywords",
                template="""请从以下文本中提取 $count 个关键词，用逗号分隔。

文本：
$text

关键词：""",
                description="关键词提取模板",
                variables=["text", "count"],
                defaults={"count": 5},
            )
        )

        # 问题生成模板
        self.register(
            PromptTemplate(
                name="generate_questions",
                template="""请根据以下内容生成 $count 个相关问题。

内容：
$content

问题：""",
                description="问题生成模板",
                variables=["content", "count"],
                defaults={"count": 3},
            )
        )

        # 改写模板
        self.register(
            PromptTemplate(
                name="rewrite_query",
                template="""请将以下问题改写为更适合搜索的形式，保持原意但更清晰。

原问题：$question

改写后的问题：""",
                description="查询改写模板",
                variables=["question"],
            )
        )

    def register(self, template: PromptTemplate) -> None:
        """注册模板"""
        self._templates[template.name] = template
        logger.debug(f"注册模板: {template.name}, 变量: {template.variables}")

    def get(self, name: str) -> Optional[PromptTemplate]:
        """获取模板"""
        return self._templates.get(name)

    def render(self, name: str, **kwargs) -> str:
        """渲染模板

        Args:
            name: 模板名称
            **kwargs: 模板变量

        Returns:
            渲染后的字符串

        Raises:
            ValueError: 模板不存在或缺少变量
        """
        template = self.get(name)
        if not template:
            available = list(self._templates.keys())
            raise ValueError(f"模板 '{name}' 不存在，可用模板: {available}")
        return template.render(**kwargs)

    def list_templates(self) -> List[str]:
        """列出所有模板"""
        return list(self._templates.keys())

    def get_template_info(self, name: str) -> Optional[Dict[str, Any]]:
        """获取模板信息"""
        template = self.get(name)
        if not template:
            return None
        return {
            "name": template.name,
            "description": template.description,
            "variables": template.variables,
            "defaults": template.defaults,
            "template": template.template,
        }

    def validate_all(self, kwargs: Dict[str, Any]) -> Dict[str, List[str]]:
        """校验所有模板的变量，返回每个模板缺失的变量"""
        result = {}
        for name, template in self._templates.items():
            missing = template.validate(kwargs)
            if missing:
                result[name] = missing
        return result
