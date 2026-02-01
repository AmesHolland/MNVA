from __future__ import annotations
from typing import Any, Dict, Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage

from agent.agents.base import BaseAgent
from agent.state import OceanState
from agent.tools.news_manager import get_news_by_id
from datetime import datetime
from dotenv import load_dotenv
from agent.config.llm_config import llm as global_llm


load_dotenv(dotenv_path="C:\\Users\\win11\\Desktop\\MNVA\\.env")


def sort_news_by_date(news_list: list[dict], recent_first: bool = True) -> list[dict]:
    """
    对新闻列表按发布日期进行排序。
    
    参数:
    - news_list: 包含新闻字典的列表。
    - recent_first: bool, True 表示降序（新->旧），False 表示升序（旧->新）。
    
    返回:
    - 排序后的新闻列表副本。
    """
    sorted_list = sorted(
        news_list, 
        key=lambda x: datetime.strptime(x["publish_date"], "%Y-%m-%d"), 
        reverse=recent_first
    )
    return sorted_list


from pydantic import BaseModel,Field

class TempStruct(BaseModel):
    info:str = Field(...,description="分析得到的信息")




class TimeEvolutionAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="TimeEvolutionAgent")
        #  FIXME: 下一行应被改为真正的结构
        self.llm = global_llm.with_structured_output(TempStruct)

    def invoke(self, state: OceanState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
        # TODO: 实现：时序新闻整理总结 + 前端可视化 新闻列表 关于这个热点 共有哪些新闻，时间演化的趋势是怎么样的 涉及到了哪些国家

        #  FIXME: 下一行应被改为从State里面读取
        time_span = "from October to December"
        #  FIXME: 下一行应被改为: topic = state["constraints"]["topic"]
        topic = "deep sea mining" # "深海采矿"

        news_meta_list = state["news_list"]
        news_list = get_news_by_id([news["id"] for news in news_meta_list])

        # 先确认新闻列表是否按时间排序 / 或者进来就直接先按时间排一次序
        # 然后再让llm识别新闻中的 热点/实体
        # 整理出关于这个热点/实体 在时间演变下的变化趋势
        sorted_news_list = sort_news_by_date(news_list,recent_first=False)
        formatted_news = [f"标题: {news['title']}\n日期: {news['publish_date']}\n内容: {news['content']}" 
                          for news in sorted_news_list]
        # FIXME: 测试用config
        if config == None:
            config = {"configurable":{"thread_id":"001"}}
        
        response = self.llm.invoke(f"""根据热点:{topic},分析下列新闻中的时间范围在{time_span}中的部分。
                         说明在此时间范围内,此热点都涉及了哪些国家、时间演化趋势如何:{"\n\n".join(formatted_news)}""")
        info = response.info
        

        # 结果以文本形式传到 messages 的 AIMessage里即可
        return {
            "messages": [AIMessage(content=info)],
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
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))  # default=str 处理 AIMessage 序列化
    except Exception as e:
        print(f"❌ Agent 调用失败！错误信息：{str(e)}")

# 执行测试
if __name__ == "__main__":
    test_time_evolution_agent()