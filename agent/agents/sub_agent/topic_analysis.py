from __future__ import annotations
from typing import Any, Dict, Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage

from agent.agents.base import BaseAgent
from agent.state import OceanState
from agent.tools.news_manager import get_news_by_id


class TopicAnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="TopicAnalysisAgent")

    def invoke(self, state: OceanState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:

        time_span = "October" # "from October to December"
        news_meta_list = state["news_list"]
        news_list = get_news_by_id([news["id"] for news in news_meta_list])
        # TODO: 实现：某段时间内，新闻热点的整理：xx时间-xx时间 共有以下几个热点信息 xxxxxx
        # state:
        # 热点分析（对已经有的新闻进行未知的热点分析某个时间段）：输入 新闻列表
        # 1. 先识别这段时间的热点
        # （热点如何界定？关键词词频？那就需要额外的关键词词频统计工具 同时将提取的关键词进行）
        # 2. 再使用LLM围绕这些新闻进行整理总结
        # 如果新闻列表太多怎么办？分批给大模型？
        # 结果以文本形式传到 messages 的 AIMessage里即可
        return {
            "messages": [AIMessage(content="[stub] TopicAnalysisAgent done")],
            "tool_trace": [{"agent": self.name, "status": "stub"}],
        }
