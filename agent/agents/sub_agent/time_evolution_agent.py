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

# ===================== 最简单的测试逻辑 =====================
def test_time_evolution_agent():
    """测试 TimeEvolutionAgent 的 invoke 方法"""
    # 1. 构造测试用的 state（模拟真实的 OceanState 数据）
    test_state = OceanState({
        "news_list": [{"id": "1"}, {"id": "2"}, {"id": "3"}],  # 模拟新闻元数据列表
        "constraints": {"topic": "deep sea mining"}  # 可选，适配你注释的代码
    })

    # 2. 实例化 agent
    agent = TimeEvolutionAgent()

    # 3. 调用 invoke 方法并捕获结果
    try:
        result = agent.invoke(test_state)
        print("✅ Agent 调用成功！输出结果：")
        # 格式化打印结果，方便查看
        import json
        print(json.dumps(result, indent=2, default=str))  # default=str 处理 AIMessage 序列化
    except Exception as e:
        print(f"❌ Agent 调用失败！错误信息：{str(e)}")

# 执行测试
if __name__ == "__main__":
    test_time_evolution_agent()