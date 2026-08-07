"""Agentic Skill 与 Components Skill 共享的运行时协议。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class ComponentInvoker(Protocol):
    """向 Agentic workflow 暴露已绑定的组件调用能力。"""

    def has(self, slot: str) -> bool:
        """判断指定槽位是否绑定了至少一个 Component Skill。"""
        ...

    def call(
        self,
        slot: str,
        inputs: Mapping[str, Any],
        *,
        index: int = 0,
    ) -> Mapping[str, Any]:
        """调用槽位中指定序号的组件，并返回结构化执行结果。"""
        ...

    def call_all(
        self,
        slot: str,
        inputs: Mapping[str, Any],
    ) -> Sequence[Mapping[str, Any]]:
        """以相同输入调用槽位中绑定的全部组件。"""
        ...


class ComponentContext(Protocol):
    """向具体 Components Skill 注入外部模型服务。"""

    def call_model(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        """调用冻结的 Executor Model，并返回生成文本。"""
        ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """将一组文本编码成顺序一致的向量。"""
        ...
