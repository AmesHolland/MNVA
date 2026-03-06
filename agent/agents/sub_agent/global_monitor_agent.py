from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from agent.agents.schemas import Claim
from agent.config.llm_config import llm_qw_quick
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
    source_ids: List[str] = Field(description="List of DOC_IDs that belong to this topic")

class GeoPoint(BaseModel):
    date: str = Field(description="Strict YYYY-MM-DD when this specific event occurred at this location")
    lat: float = Field(description="Latitude of the event location")
    lon: float = Field(description="Longitude of the event location")
    topic_name: str = Field(description="Name of the topic this location belongs to")
    intensity: int = Field(description="Number of related news articles for this location (1-5 scale)")
    summary: str = Field(description="Short summary for the map tooltip")
    source_ids: List[str] = Field(description="List of DOC_IDs that support this geographic event")

class TimePoint(BaseModel):
    date: str = Field(description="Date in YYYY-MM-DD format")
    topic_name: str = Field(description="Name of the topic")
    count: int = Field(description="Number of articles for this topic on this date")
    source_ids: List[str] = Field(description="List of DOC_IDs published on this date for this topic")

class GlobalMonitorOutput(BaseModel):
    overview_claims: List[Claim] = Field(description="A macro summary broken down into traceable claims.")
    topics: List[TopicNode] = Field(description="List of top 3-5 identified topics with their temporal patterns.")
    geo_data: List[GeoPoint] = Field(description="Data for rendering the dynamic map with time sliders.")
    trend_data: List[TimePoint] = Field(description="Data for rendering the ThemeRiver/AreaChart.")

# 1. 初始化 LLM
llm = llm_qw_quick

# 2. 设置 Parser
parser = JsonOutputParser(pydantic_object=GlobalMonitorOutput)

# 3. 构建 Chain
prompt = PromptTemplate(
    template=GLOBAL_MONITOR_PROMPT,
    input_variables=["query", "news_context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

global_monitor_chain = prompt | llm | parser


# 4. 定义 LangGraph 节点函数
def global_monitor_agent(docs, query, blueprint_context=""):
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
            "blueprint_context": blueprint_context  # 注入 Prompt
        })
    except Exception as e:
        print(f"❌ Global Monitor 运行出错: {e}")
        # 返回友好的错误状态，避免整个图崩溃
        return {"error": f"Global Monitor execution failed: {str(e)}"}

    print("✅ 分析完成，已生成带溯源的宏观态势数据。")

    # 3. 组装 State 更新字典
    # 这里的结构高度适配你在 Vue 前端的四视图布局
    return {
        "final_answer": result['overview_claims'],
        "topics_list": result['topics'],
        "visualization_data": {
            "view_type": "global_monitor",
            "geo_dynamic_data": result.get("geo_data", []),  # 供带时间滑块的地图使用
            "trend_river_data": result.get("trend_data", [])  # 供主题河流图使用
        },

        # 保留原始解析结果以备他用
        "structured_insight": result
    }

if __name__ == '__main__':
    query = "US deep-sea mining strategic initiatives and international regulatory responses 2025-10-08 to 2025-12-31"
    news_list = get_news_by_id([])
    result = global_monitor_agent(news_list, query, "美国从10月激进部署与规则博弈，转向12月因挪威暂停及国际压力而陷入治理僵局。")
    print(result)

# def global_monitor_node(state):
#     # 1. 获取检索到的新闻 (假设在 search_node 中已经存入 state)
#     # state['retrieved_docs'] = [{'id': 1, 'content': '...', 'date': '...'}, ...]
#     raw_docs = state.get("retrieved_docs", [])
#
#     # 2. 调用 LLM 进行打标 (Tagging)
#     # 这一步只让 LLM 返回分类结果，不让它数数
#     tagging_result = tagging_chain.invoke({"news_list_formatted": format_docs(raw_docs)})
#     # tagging_result = [{'id': 1, 'topic': 'A', 'location': '...'}, ...]
#
#     # 3. 【关键步骤】使用 Python 进行确定性统计 (Deterministic Aggregation)
#
#     # 将 LLM 的结果转为 DataFrame 方便处理
#     df_tags = pd.DataFrame(tagging_result)
#     df_docs = pd.DataFrame(raw_docs)
#
#     # 合并数据 (以 ID 为键)
#     merged_df = pd.merge(df_tags, df_docs, on="id")
#
#     # --- A. 统计 Topic 热度 (Count) ---
#     # 这一步计算出的数值是 100% 准确的
#     topic_counts = merged_df['topic'].value_counts().reset_index()
#     topic_counts.columns = ['topic_name', 'count']
#
#     # --- B. 生成时间演化数据 (Time Evolution) ---
#     # 因为我们在 raw_docs 里有准确的 date，这里直接 groupby 即可
#     # 这样你就无需单独的 Time-Evolution-Agent 了
#     trend_data = merged_df.groupby(['date', 'topic']).size().reset_index(name='count')
#
#     # --- C. 生成地理数据 (Geo Data) ---
#     # 可以取每个 Topic 下出现频率最高的 Location，或者保留所有散点
#     geo_data = merged_df[['lat', 'lon', 'topic', 'title']].to_dict(orient='records')
#
#     # 4. 构建最终输出
#     final_output = {
#         "topics_summary": topic_counts.to_dict(orient='records'),
#         "trend_data": trend_data.to_dict(orient='records'),
#         "geo_data": geo_data,
#         "raw_evidence": merged_df.to_dict(orient='records')  # 用于溯源验证
#     }
#
#     return {"visualization_data": final_output}