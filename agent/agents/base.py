from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig

class BaseAgent(ABC):
    """所有子 Agent 的统一抽象：输入 state，输出“部分 state 更新 dict”。
    约定：
      - 不直接 mutate 传入的 state（除非你非常确定 reducer/merge 逻辑）
      - 返回的 dict 只包含你想更新的键
    """

    name: str

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def invoke(self, state: Dict[str, Any], config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        raise NotImplementedError
