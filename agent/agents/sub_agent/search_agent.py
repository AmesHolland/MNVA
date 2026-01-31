from __future__ import annotations
from typing import Any, Dict, Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage

from agent.agents.base import BaseAgent
from agent.state import OceanState


class SearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="SearchAgent")

    def invoke(self, state: OceanState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        # TODO: 实现：Weaviate hybrid search + rerank
        # 期望返回：news_candidates(列表)、tool_trace追加、可能的viz_buffer追加
        return {
            "messages": [AIMessage(content="[stub] SearchAgent done 这是检索到了的关于xx的信息")],
            "tool_trace": [{"agent": self.name, "status": "stub"}],
        }
