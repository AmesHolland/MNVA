from __future__ import annotations
from typing import Any, Dict, Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage

from agent.agents.base import BaseAgent
from agent.state import OceanState
from agent.tools.news_manager import get_news_by_id


class TimeEvolutionAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="TimeEvolutionAgent")

    def invoke(self, state: OceanState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        # TODO: 实现：时序新闻整理总结 + 前端可视化 新闻列表 关于这个热点 共有哪些新闻，时间演化的趋势是怎么样的 涉及到了哪些国家

        # topic = state["constraints"]["topic"]
        topic = "deep sea mining" # "深海采矿"

        news_meta_list = state["news_list"]
        news_list = get_news_by_id([news["id"] for news in news_meta_list])

        # 先确认新闻列表是否按时间排序 / 或者进来就直接先按时间排一次序
        # 然后再让llm识别新闻中的 热点/实体
        # 整理出关于这个热点/实体 在时间演变下的变化趋势

        # 期望返回：news_candidates(列表)、tool_trace追加、可能的viz_buffer追加
        return {
            "messages": [AIMessage(content="[stub] TimeEvolutionAgent done")],
            "tool_trace": [{"agent": self.name, "status": "stub"}],
        }