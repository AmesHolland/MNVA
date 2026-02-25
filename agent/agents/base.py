from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from typing import List

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



# 全局通用的文本溯源基类
class Claim(BaseModel):
    statement: str = Field(description="总结的论点、事实判断或事件描述。")
    is_direct_quote: bool = Field(description="如果是直接截取新闻原话为 true，自行归纳总结为 false。")
    source_ids: List[str] = Field(description="支撑该句话的具体新闻 DOC_ID 列表。")
