from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from agent.agents.schemas import Claim
from agent.config.llm_config import llm_qw_quick, llm_qw_thinking
from agent.config.prompt_template import GLOBAL_MONITOR_PROMPT
from typing import List, Optional
from pydantic import BaseModel, Field

from agent.tools.news_manager import get_news_by_id

from pydantic import BaseModel, Field
from typing import List

class TopicNode(BaseModel):
    topic_name: str = Field(description="Name of the clustered topic (e.g., 'Naval Drills', 'Fishery Disputes')")
    description: str = Field(description="A brief 1-sentence explanation of this topic")
    temporal_pattern: str = Field(description="Describe how this topic shifted over time (e.g., 'Continuous low-level reports', 'Sudden burst around mid-month').")
    source_ids: List[str] = Field(description="List of DOC_IDs that belong to this topic. Like ['001']")

class GeoPoint(BaseModel):
    date: str = Field(description="Strict YYYY-MM-DD when this specific event occurred at this location")
    lat: float = Field(description="Latitude of the event location")
    lon: float = Field(description="Longitude of the event location")
    topic_name: str = Field(description="Name of the topic this location belongs to")
    intensity: int = Field(description="Number of related news articles for this location (1-5 scale)")
    summary: str = Field(description="Short summary for the map tooltip")
    source_ids: List[str] = Field(description="List of DOC_IDs that support this geographic event. Like ['001']")

# === 1. 更新 TimePoint 模型以适配 Ridgeline Plot ===
class RidgelinePoint(BaseModel):
    date: str = Field(description="Date in YYYY-MM-DD format")
    topic_name: str = Field(description="Name of the topic")
    count: int = Field(description="Number of articles for this topic on this date")
    source_ids: List[str] = Field(description="List of DOC_IDs published on this date for this topic. Like ['001']")

# === 新增：释放大模型思考能力的洞察模型 ===
class StrategicInsights(BaseModel):
    core_conflict: str = Field(description="Highly analytical summary of the core geopolitical contradictions or friction points.")
    hidden_intentions: str = Field(description="Deep analysis of the underlying strategic motives of the key actors involved, going beyond surface-level news.")
    trend_prediction: str = Field(description="A forward-looking forecast of how this situation is likely to evolve in the short-to-medium term.")

# === 修改：重构 GlobalMonitorOutput ===
class GlobalMonitorOutput(BaseModel):
    # 1. 客观事实层（替代原来的 overview_claims）
    factual_grounding: List[Claim] = Field(description="Objective factual timeline and events extracted directly from the news. MUST be strictly traced to source_ids.")
    # 2. 主观洞察层（新加入的灵魂）
    strategic_insights: StrategicInsights = Field(description="Deep, subjective analytical insights, hidden motives, and predictions. No source_ids required here.")
    # 3. 可视化数据层（保持不变）
    topics: List[TopicNode] = Field(description="List of top 3-5 identified topics with their temporal patterns.")
    geo_data: List[GeoPoint] = Field(description="Data for rendering the dynamic map with time sliders.")
    ridgeline_data: List[RidgelinePoint] = Field(description="Data for rendering the Ridgeline Plot (daily counts per topic).")

# 1. 初始化 LLM
llm = llm_qw_thinking

# 2. 设置 Parser
parser = JsonOutputParser(pydantic_object=GlobalMonitorOutput)

# 3. 构建 Chain
prompt = PromptTemplate(
    template=GLOBAL_MONITOR_PROMPT,
    input_variables=["query", "news_context", "output_language"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

global_monitor_chain = prompt | llm | parser


# 4. 定义 LangGraph 节点函数
def global_monitor_agent(docs, query, blueprint_context="", output_language="English"):
    """
    LangGraph Node: Extracts macro spatiotemporal shifts and topics.
    Expected State Input:
    - query: str
    - retrieved_docs: List[dict] (e.g., [{"DOC_ID": "123", "publish_date": "2026-01-01", "content": "..."}])
    """
    print("\n--- 🌐 GLOBAL MONITOR AGENT: Analyzing Macro Spatiotemporal Shifts ---")

    # query = state.get("query", "")
    # docs = state.get("retrieved_docs", [])

    # 1. 严格格式化上下文，确保 DOC_ID 醒目可见
    formatted_docs = []
    for d in docs:
        doc_id = d.get('DOC_ID', 'UNKNOWN')
        date = d.get('publish_date', 'UNKNOWN_DATE')
        content = d.get('content', '')
        formatted_docs.append(f"[DOC_ID: {doc_id}] - [{date}] {content}")

    news_context = "\n".join(formatted_docs)

    # 2. 调用大模型
    try:
        result = global_monitor_chain.invoke({
            "query": query,
            "news_context": news_context,
            "blueprint_context": blueprint_context,  # 注入 Prompt
            "output_language":output_language
        })
    except Exception as e:
        print(f"❌ Global Monitor 运行出错: {e}")
        # 返回友好的错误状态，避免整个图崩溃
        return {"error": f"Global Monitor execution failed: {str(e)}"}

    print("✅ 分析完成，已生成带溯源的宏观态势数据。")

    # 3. 组装 State 更新字典
    # 这里的结构高度适配你在 Vue 前端的四视图布局
    # 3. 组装 State 更新字典
    return {
        # 将事实与洞察同时返回给下游
        "factual_grounding": result.get('factual_grounding', []),
        "strategic_insights": result.get('strategic_insights', {}),
        "topics_list": result.get('topics', []),

        "visualization_data": {
            "view_type": "global_monitor",
            "geo_dynamic_data": result.get("geo_data", []),
            "ridgeline_data": result.get("ridgeline_data", [])
        }
    }

if __name__ == '__main__':
    query = "US deep-sea mining strategic initiatives and international regulatory responses 2025-10-08 to 2025-12-31"
    news_list = get_news_by_id([])
    result = global_monitor_agent(news_list, query, "美国从10月激进部署与规则博弈，转向12月因挪威暂停及国际压力而陷入治理僵局。")
    print(result)
